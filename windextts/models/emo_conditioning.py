"""Emotion conditioning modules for the GPT-AR stage — pure-torch re-implementation.

Replaces ``indextts/gpt/conformer_encoder.py`` (ConformerEncoder) and
``indextts/gpt/perceiver.py`` (PerceiverResampler) for the IndexTTS-2.5
emotion-reference-audio path:

    emo_audio → w2v-bert[17] → emo_cond_emb [B,T,1024]
        → EmoConformerEncoder (conv2d2 subsample + 4 conformer blocks) [B,T',512]
        → EmoPerceiverEncoder (2 cross-attn layers, 1 latent) [B,1,1024]
        → emovec_layer / emo_layer (in gpt.py) [B,1280]

Numerical contract: aligned to official IndexTTS-2.5 (fp32, same-device) via
``tests/align/test_emo_conditioning_align.py``.

Key implementation notes (grounded in the official sources):
  - ConformerEncoder uses RelPositionMultiHeadedAttention (relative positional
    encoding) — NOT standard MHA. The relative-position score is the sum of
    two bilinear terms ((q+bias_u)·kᵀ + (q+bias_v)·pᵀ), which we express as a
    single SDPA call by concatenating q and k with their position-projected
    counterparts along the head dim (see RelPositionMultiHeadedAttention).
  - macaron_style=False for this config: the weights contain a single
    feed_forward per layer (no feed_forward_macaron), so ff_scale = 1.0.
  - conv2d2 subsampling: Conv2d(1,512,3,3,stride=2) halves T (133→66) and
    halves the 1024 feature dim to 511 → Linear(512*511=261632 → 512).
  - The perceiver is a PerceiverResampler with num_latents=1, dim=1024,
    dim_context=512, heads=4, ff_mult=2, depth=2. Its key padding mask is
    (B, 1+T') — the conformer mask padded with one valid slot for the latent.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EmoConformerEncoder", "EmoPerceiverEncoder"]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    return seq_range_expand >= lengths.unsqueeze(-1)  # [B, max_len]


class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP applied per position (ReLU-free: SiLU in this model)."""

    def __init__(self, idim: int, hidden_units: int, activation: nn.Module):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(xs)))


class ConvolutionModule(nn.Module):
    """Conv1d GLU module from the Conformer paper (kernel 15, LayerNorm)."""

    def __init__(self, channels: int, kernel_size: int = 15, activation: nn.Module = nn.SiLU()):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=padding, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.activation = activation

    def forward(self, x: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
        """x: [B,T,C]; mask_pad: [B,1,T] (True=valid). Returns [B,T,C]."""
        x = x.transpose(1, 2)  # [B,C,T]
        x.masked_fill_(~mask_pad, 0.0)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # GLU
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2)


class RelPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, returned separately for rel-pos attention."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B,T,D] → (x * sqrt(D), pos_emb [1,T,D])."""
        pos_emb = self.pe[:, : x.size(1)]
        return x * self.xscale, pos_emb


class RelPositionMultiHeadedAttention(nn.Module):
    """MHA with relative position encoding (Transformer-XL style, paper 1901.02860).

    score_ij = ((q_i + pos_bias_u) · k_j + (q_i + pos_bias_v) · p_j) / sqrt(d_k)
    where p is the position-projected encoding. The two bilinear terms are
    folded into one SDPA call via head-dim concatenation:
        q' = cat([q+bias_u, q+bias_v], -1),  k' = cat([k, p], -1)
        q'·k'ᵀ = (q+bias_u)·kᵀ + (q+bias_v)·pᵀ        (exact)
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        # init is overwritten by weight loading; keep sane defaults anyway
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """query/key/value: [B,T,D]; mask: [B,1,T] (True=valid); pos_emb: [1,T,D]."""
        b, t = query.size(0), query.size(1)
        q = self.linear_q(query).view(b, t, self.h, self.d_k)  # [B,T,h,dk]
        k = self.linear_k(key).view(b, t, self.h, self.d_k)
        v = self.linear_v(value).view(b, t, self.h, self.d_k)

        p = self.linear_pos(pos_emb).view(1, t, self.h, self.d_k)  # [1,T,h,dk]

        q_u = (q + self.pos_bias_u)  # [B,T,h,dk]
        q_v = (q + self.pos_bias_v)
        q_cat = torch.cat([q_u, q_v], dim=-1)  # [B,T,h,2dk]
        k_cat = torch.cat([k, p], dim=-1)      # [B,T,h,2dk]

        q_s = q_cat.transpose(1, 2)  # [B,h,T,2dk]
        k_s = k_cat.transpose(1, 2)
        v_s = v.transpose(1, 2)      # [B,h,T,dk]

        attn_mask = torch.zeros(b, 1, 1, t, device=query.device, dtype=query.dtype)
        attn_mask = attn_mask.masked_fill(~mask.unsqueeze(1), float("-inf"))  # [B,1,1,T]
        out = F.scaled_dot_product_attention(
            q_s, k_s, v_s, attn_mask=attn_mask, scale=1.0 / math.sqrt(self.d_k)
        )  # [B,h,T,dk]

        out = out.transpose(1, 2).contiguous().view(b, t, self.h * self.d_k)
        return self.linear_out(out)


class ConformerEncoderLayer(nn.Module):
    """One Conformer block: pre-norm MHA + conv module + FFN, macaron disabled."""

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        conv_module: nn.Module,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size, eps=1e-5)
        self.ff_scale = 1.0  # macaron_style=False in this checkpoint
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        # multi-headed self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att = self.self_attn(x, x, x, mask, pos_emb)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = self.conv_module(x, mask_pad)
            x = residual + x
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)
        return x


class EmoConformerEncoder(nn.Module):
    """ConformerEncoder(input=1024, output=512, linear=1024, heads=4, blocks=4,
    input_layer='conv2d2', rel_pos). Mirrors the gpt.pth
    ``emo_conditioning_encoder`` submodule key-for-key."""

    def __init__(
        self,
        input_size: int = 1024,
        output_size: int = 512,
        attention_heads: int = 4,
        linear_units: int = 1024,
        num_blocks: int = 4,
        input_layer: str = "conv2d2",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        cnn_module_kernel: int = 15,
    ):
        super().__init__()
        self._output_size = output_size

        # conv2d2 subsampling: Conv2d(1,512,3,3,stride=2) then Linear(261632→512)
        conv_inner = output_size * ((input_size - 1) // 2)  # 512*511 = 261632
        self.embed = nn.Module()
        self.embed.conv = nn.Sequential(
            nn.Conv2d(1, output_size, 3, stride=2),
            nn.ReLU(),
        )
        self.embed.out = nn.Sequential(nn.Linear(conv_inner, output_size))
        self.embed.pos_enc = RelPositionalEncoding(output_size)

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

        activation = nn.SiLU()
        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(attention_heads, output_size, 0.0),
                    PositionwiseFeedForward(output_size, linear_units, activation),
                    ConvolutionModule(output_size, cnn_module_kernel, activation),
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """xs: [B,T,1024]; xs_lens: [B]. Returns (out [B,T',512], mask [B,1,T'])."""
        T = xs.size(1)
        masks = ~_make_pad_mask(xs_lens, T).unsqueeze(1)  # [B,1,T] True=valid

        # subsampling embed
        x = xs.unsqueeze(1)  # [B,1,T,1024]
        x = self.embed.conv(x)  # [B,512,T',511]
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.embed.out(x)  # [B,T',512]
        x, pos_emb = self.embed.pos_enc(x)
        masks = masks[:, :, 2::2]  # [B,1,T']

        chunk_masks = masks
        mask_pad = masks
        for layer in self.encoders:
            x = layer(x, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            x = self.after_norm(x)
        return x, masks


# ---------------------------------------------------------------------------
# perceiver (PerceiverResampler)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """RMSNorm with learnable gamma (official: F.normalize * dim^0.5 * gamma)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class _FeedForward(nn.Module):
    """GEGLU feed-forward: Linear(0) → GEGLU → Linear(2) (official FFN, mult=2).

    Registers the two Linear layers under names ``0`` and ``2`` so the
    checkpoint keys ``layers.{i}.1.0`` / ``layers.{i}.1.2`` match directly.
    """

    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)  # 1024*2*2/3 = 1365
        self.add_module("0", nn.Linear(dim, dim_inner * 2))
        self.add_module("2", nn.Linear(dim_inner, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._modules["0"](x)
        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(gate) * x
        return self._modules["2"](x)


class _Attention(nn.Module):
    """Cross-attention over context, optionally prepending the query (latent)."""

    def __init__(
        self,
        dim: int,
        dim_context: int,
        dim_head: int = 64,
        heads: int = 8,
        cross_attn_include_queries: bool = False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B,N,D]; context: [B,J,Dc]; mask: [B,J'] (True=keep) or [B,1,J']."""
        h = self.heads
        has_context = context is not None
        context = x if context is None else context
        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)  # [B, 1+J, Dc]

        q = self.to_q(x)  # [B,N,Hi]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # [B,J',Hi]
        q = q.view(*q.shape[:2], h, -1).transpose(1, 2)  # [B,h,N,dh]
        k = k.view(*k.shape[:2], h, -1).transpose(1, 2)
        v = v.view(*v.shape[:2], h, -1).transpose(1, 2)

        j = k.size(-2)
        attn_mask = None
        if mask is not None:
            # mask [B,J'] True=keep → additive [B,1,1,J']
            m = mask.bool() if mask.dtype == torch.bool else mask > 0
            attn_mask = torch.zeros(*m.shape[:1], 1, 1, j, device=m.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~m.unsqueeze(1).unsqueeze(1), float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        # transpose first, then derive dims from the transposed tensor
        out = out.transpose(1, 2).contiguous()
        out = out.view(*out.shape[:2], -1)
        return self.to_out(out)


class EmoPerceiverEncoder(nn.Module):
    """PerceiverResampler(dim=1024, dim_context=512, num_latents=1, heads=4,
    ff_mult=2, depth=2). Mirrors the gpt.pth ``emo_perceiver_encoder`` submodule."""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 2,
        dim_context: int = 512,
        num_latents: int = 1,
        dim_head: int = 64,
        heads: int = 4,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.proj_context = nn.Linear(dim_context, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        # Official Attention defaults dim_context=dim; the
                        # context fed here is the proj_context output (1024),
                        # so to_kv is Linear(1024 → dim_inner*2).
                        _Attention(
                            dim=dim, dim_context=dim, dim_head=dim_head,
                            heads=heads, cross_attn_include_queries=True,
                        ),
                        _FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = _RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B,S,512]; mask: [B,1+S] (True=keep, already padded by caller)."""
        batch = x.shape[0]
        x = self.proj_context(x)  # [B,S,1024]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)  # [B,1,1024]
        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)  # [B,1,1024]


# ---------------------------------------------------------------------------
# official get_emo_conditioning composition (used by inference glue)
# ---------------------------------------------------------------------------

def get_emovec(
    conformer: EmoConformerEncoder,
    perceiver: EmoPerceiverEncoder,
    emovec_layer: nn.Linear,
    emo_layer: nn.Linear,
    speech_conditioning_latent: torch.Tensor,  # [B,T,1024] (w2v-bert[17] feat)
    cond_mel_lengths: torch.Tensor,            # [B]
) -> torch.Tensor:
    """Full emo_vec extraction: conformer → perceiver → emovec_layer → emo_layer.

    Returns [B,1280]. Replicates GPT.get_emovec in model_v2.py:827.
    """
    feat = speech_conditioning_latent
    seq, mask = conformer(feat, cond_mel_lengths)  # [B,T',512], [B,1,T']
    conds_mask = F.pad(mask.squeeze(1), (1, 0), value=True)  # [B,1+T']
    conds = perceiver(seq, conds_mask)  # [B,1,1024]
    emo_vec_syn = emovec_layer(conds.squeeze(1))  # [B,1280]
    return emo_layer(emo_vec_syn)  # [B,1280]
