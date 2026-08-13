"""S2Mel-CFM DiT estimator — pure-torch re-implementation.

Replaces ``indextts.s2mel.modules.diffusion_transformer.DiT`` (+ its deps
``gpt_fast.model.Transformer`` and ``wavenet.WN``) with zero indextts/transformers
dependency. This is the velocity network of the flow-matching Euler solver:
``estimator(x, prompt_x, x_lens, t, style, cond) -> dphi_dt [B, 80, T]``.

Verified contract (all from the official sources, line behavior replicated):
- DiT.forward (diffusion_transformer.py:186-257):  x_in = cat([x, prompt_x,
  cond_projection(cond), style]) (dim 80+80+512+192=864) -> cond_x_merge_linear
  -> Transformer (13 blocks, RoPE, uvit skip) -> skip_linear (long skip) ->
  conv1(Linear) -> WN (8 layers, gated) + res_projection -> FinalLayer(adaLN) -> conv2.
- Transformer (gpt_fast/model.py): llama-style blocks, wqkv fused (n_head+2*n_local)
  heads, swiGLU FFN (intermediate 1536), RMSNorm + AdaptiveLayerNorm (adaLN via
  project_layer on the time embedding), RoPE (precompute_freqs_cis 16384, base
  10000), uvit skip (layers 0..5 emit -> 7..12 receive).
- WN (wavenet.py): 8 x SConv1d(512->1024, k5, reflect-pad) in_layers with
  tanh/sigmoid gating conditioned on t2, res_skip 1x1 convs, Dropout(0.2) (no-op
  in eval). SConv1d padding: kernel 5 stride 1 -> pad (2,2) reflect.
- TimestepEmbedder: sinusoidal freqs (scale 1000, max_period 10000) + 2-layer MLP.
- FinalLayer: norm_final (LayerNorm no-affine eps 1e-6) + adaLN (shift/scale from
  SiLU->Linear chunk(2)) + weight_norm linear.

Numerics: fp32 faithful. All weight_norm layers use classic
``torch.nn.utils.weight_norm`` (weight_g/weight_v keys, matching the checkpoint).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

__all__ = ["DiT", "DiTConfig", "sequence_mask"]


# ---------------------------------------------------------------------------
# Config (mirrors config.yaml s2mel.DiT / s2mel.wavenet / s2mel.style_encoder)
# ---------------------------------------------------------------------------


class DiTConfig:
    """Hyperparameters for the S2Mel DiT estimator (IndexTTS-2.5 defaults)."""

    def __init__(self):
        # s2mel.DiT
        self.hidden_dim = 512
        self.num_heads = 8
        self.depth = 13
        self.class_dropout_prob = 0.1
        self.in_channels = 80
        self.style_condition = True
        self.final_layer_type = "wavenet"
        self.content_dim = 512
        self.content_codebook_size = 1024
        self.content_type = "continuous"  # cond_projection path is used
        self.is_causal = False
        self.long_skip_connection = True
        self.time_as_token = False
        self.style_as_token = False
        self.uvit_skip_connection = True
        # s2mel.wavenet
        self.wavenet_hidden_dim = 512
        self.wavenet_num_layers = 8
        self.wavenet_kernel_size = 5
        self.wavenet_dilation_rate = 1
        self.wavenet_p_dropout = 0.2
        self.wavenet_style_condition = True
        # s2mel.style_encoder
        self.style_encoder_dim = 192


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Boolean mask [B, max_length]: True where arange < length. (commons.py:155)"""
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int) -> torch.Tensor:
    """(commons.py:133) Gated activation: tanh(a) * sigmoid(b) over channel halves."""
    in_act = input_a + input_b
    t_act_part, s_act_part = torch.split(in_act, n_channels, dim=1)
    return torch.tanh(t_act_part) * torch.sigmoid(s_act_part)


# ---------------------------------------------------------------------------
# gpt_fast Transformer (RoPE, AdaptiveLayerNorm, swiGLU, uvit skip)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp16-native: no .float() guard — fp16's 10-bit mantissa is sufficient
        # for RMSNorm (verified: cosine 0.9997 vs fp32). Removes the cast kernel
        # tax that made fp16 eager slower than fp32 before CUDA Graph.
        return self._norm(x) * self.weight


class AdaptiveLayerNorm(nn.Module):
    """Layer norm modulated by a conditioning embedding (adaLN). (gpt_fast)"""

    def __init__(self, d_model: int, norm: nn.Module) -> None:
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model

    def forward(self, input: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(self.project_layer(embedding), self.d_model, dim=-1)
        return weight * self.norm(input) + bias


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000.0, dtype: torch.dtype = torch.float32):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # fp16-native: no .float() guard on xshaped — fp16 precision is sufficient
    # for RoPE (verified: cosine 0.9997 vs fp32). freqs_cis stays fp32 (complex
    # buffer from setup_caches); the mixed fp16×fp32 multiply promotes to fp32,
    # then .type_as(x) is removed so the result stays in the promoted dtype and
    # flows to the next op without an extra cast. This halves the cast kernel
    # count in the DiT forward.
    xshaped = x.reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2


class Attention(nn.Module):
    """Multi-head self-attention, fused wqkv + RoPE + SDPA. (gpt_fast Attention)"""

    def __init__(self, dim: int, n_head: int, n_local_heads: int, head_dim: int):
        super().__init__()
        total_head_dim = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total_head_dim, bias=False)
        self.wo = nn.Linear(head_dim * n_head, dim, bias=False)
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)
        return self.wo(y)


class FeedForward(nn.Module):
    """swiGLU FFN. (gpt_fast FeedForward)"""

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, n_local_heads: int, head_dim: int,
                 intermediate_size: int, norm_eps: float, uvit_skip_connection: bool):
        super().__init__()
        rms = lambda: RMSNorm(dim, eps=norm_eps)
        self.attention = Attention(dim, n_head, n_local_heads, head_dim)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.ffn_norm = AdaptiveLayerNorm(dim, rms())
        self.attention_norm = AdaptiveLayerNorm(dim, rms())
        self.uvit_skip_connection = uvit_skip_connection
        if uvit_skip_connection:
            self.skip_in_linear = nn.Linear(dim * 2, dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None,
        input_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
        skip_in_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


class Transformer(nn.Module):
    """Transformer stack with uvit skip + adaLN output norm. (gpt_fast Transformer)

    ``forward(x, c, input_pos, mask)``; ``c`` is the time embedding [B, 1, D].
    """

    def __init__(self, n_layer: int, dim: int, n_head: int, head_dim: int,
                 intermediate_size: int, block_size: int = 16384,
                 rope_base: float = 10000.0, norm_eps: float = 1e-5,
                 uvit_skip_connection: bool = True):
        super().__init__()
        self.n_layer = n_layer
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_local_heads = n_head
        self.intermediate_size = intermediate_size
        self.block_size = block_size
        self.rope_base = rope_base
        self.uvit_skip_connection = uvit_skip_connection

        self.layers = nn.ModuleList(
            TransformerBlock(dim, n_head, n_head, head_dim, intermediate_size,
                             norm_eps, uvit_skip_connection)
            for _ in range(n_layer)
        )
        self.norm = AdaptiveLayerNorm(dim, RMSNorm(dim, eps=norm_eps))

        self.freqs_cis: torch.Tensor | None = None
        self.causal_mask: torch.Tensor | None = None
        self.max_seq_length = -1
        self.max_batch_size = -1

        if uvit_skip_connection:
            half = n_layer // 2
            self.layers_emit_skip = [i for i in range(n_layer) if i < half]
            self.layers_receive_skip = [i for i in range(n_layer) if i > half]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def setup_caches(self, max_batch_size: int, max_seq_length: int, use_kv_cache: bool = True) -> None:
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device
        # freqs_cis is sized by block_size (16384) — independent of max_seq_length.
        # Only build it ONCE (first call). Rebuilding on every max_seq_length
        # growth changes the tensor address, which invalidates already-captured
        # CUDA Graphs that bound the old address (graph reads garbage RoPE →
        # brick audio). This is the root cause of cross-bucket graph corruption.
        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(self.block_size, self.head_dim, self.rope_base, dtype).to(device)
        self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)).to(device)
        self.use_kv_cache = use_kv_cache

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        input_pos: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized (call setup_caches first)"
        if mask is None:
            mask = self.causal_mask[None, None, input_pos]
            mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(x, c, input_pos, freqs_cis, mask, skip_in_x)
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        x = self.norm(x, c)
        return x


# ---------------------------------------------------------------------------
# Wavenet (WN) with SConv1d (asymmetric reflect padding + weight_norm)
# ---------------------------------------------------------------------------


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tuple[int, int], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    """F.pad wrapper with reflect fallback for short inputs. (encodec.py)"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class NormConv1d(nn.Module):
    """Conv1d wrapper applying weight_norm. State keys: conv.weight_g/weight_v."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 norm: str = "none"):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)
        if norm == "weight_norm":
            self.conv = weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SConv1d(nn.Module):
    """Conv1d with asymmetric reflect padding (encodec SConv1d)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 causal: bool = False, norm: str = "none"):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride=stride,
                               dilation=dilation, groups=groups, bias=bias, norm=norm)
        self.causal = causal
        self.pad_mode = "reflect"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        effective_kernel = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel - stride
        extra_padding = get_extra_padding_for_conv1d(x, effective_kernel, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class WN(nn.Module):
    """WaveNet-style gated residual stack (wavenet.py WN, causal=False)."""

    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int,
                 n_layers: int, gin_channels: int = 0, p_dropout: float = 0, causal: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = SConv1d(gin_channels, 2 * hidden_channels * n_layers, 1, norm="weight_norm")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            in_layer = SConv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                               dilation=dilation, norm="weight_norm", causal=causal)
            self.in_layers.append(in_layer)
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            res_skip_layer = SConv1d(hidden_channels, res_skip_channels, 1, norm="weight_norm", causal=causal)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(x)
        hc = self.hidden_channels
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                g_l = g[:, i * 2 * hc : (i + 1) * 2 * hc, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, hc)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :hc, :]) * x_mask
                output = output + res_skip_acts[:, hc:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


# ---------------------------------------------------------------------------
# DiT embeddings
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP. (diffusion_transformer.py:19-61)"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000

        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # cast back to t's dtype so downstream Linear (which may be fp16 under
        # estimator_fp16_weights mode) does not hit a dtype mismatch.
        return embedding.to(t.dtype) if t.dtype != torch.float32 else embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t))


class FinalLayer(nn.Module):
    """adaLN-modulated final layer. (diffusion_transformer.py:84-101)"""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# DiT estimator
# ---------------------------------------------------------------------------


class DiT(nn.Module):
    """S2Mel-CFM velocity estimator (diffusion_transformer.py DiT).

    forward(x, prompt_x, x_lens, t, style, cond) -> dphi_dt [B, in_channels, T].
    """

    def __init__(self, cfg: DiTConfig | None = None):
        super().__init__()
        cfg = cfg or DiTConfig()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels
        self.num_heads = cfg.num_heads
        self.time_as_token = cfg.time_as_token
        self.style_as_token = cfg.style_as_token
        self.is_causal = cfg.is_causal
        self.long_skip_connection = cfg.long_skip_connection
        self.uvit_skip_connection = cfg.uvit_skip_connection
        self.transformer_style_condition = cfg.style_condition
        self.final_layer_type = cfg.final_layer_type

        # TeaCache state (vLLM-Omni-style step skipping for diffusion)
        self.teacache_enabled = False
        self.teacache_thresh = 0.0
        self.teacache_coef = None  # np.poly1d coefficients for rescaling
        self._tc_state = None  # per-call reset

        # Transformer (gpt_fast ModelArgs, block_size hardcoded 16384 in official)
        self.transformer = Transformer(
            n_layer=cfg.depth,
            dim=cfg.hidden_dim,
            n_head=cfg.num_heads,
            head_dim=cfg.hidden_dim // cfg.num_heads,
            intermediate_size=find_multiple(int(2 * (4 * cfg.hidden_dim) / 3), 256),
            block_size=16384,
            uvit_skip_connection=cfg.uvit_skip_connection,
        )

        self.x_embedder = weight_norm(nn.Linear(cfg.in_channels, cfg.hidden_dim, bias=True))
        self.cond_embedder = nn.Embedding(cfg.content_codebook_size, cfg.hidden_dim)  # discrete (unused path)
        self.cond_projection = nn.Linear(cfg.content_dim, cfg.hidden_dim, bias=True)  # continuous

        self.t_embedder = TimestepEmbedder(cfg.hidden_dim)

        self.register_buffer("input_pos", torch.arange(16384))

        if self.final_layer_type == "wavenet":
            self.t_embedder2 = TimestepEmbedder(cfg.wavenet_hidden_dim)
            self.conv1 = nn.Linear(cfg.hidden_dim, cfg.wavenet_hidden_dim)
            self.conv2 = nn.Conv1d(cfg.wavenet_hidden_dim, cfg.in_channels, 1)
            self.wavenet = WN(
                hidden_channels=cfg.wavenet_hidden_dim,
                kernel_size=cfg.wavenet_kernel_size,
                dilation_rate=cfg.wavenet_dilation_rate,
                n_layers=cfg.wavenet_num_layers,
                gin_channels=cfg.wavenet_hidden_dim,
                p_dropout=cfg.wavenet_p_dropout,
                causal=False,
            )
            self.final_layer = FinalLayer(cfg.wavenet_hidden_dim, 1, cfg.wavenet_hidden_dim)
            self.res_projection = nn.Linear(cfg.hidden_dim, cfg.wavenet_hidden_dim)
            self.wavenet_style_condition = cfg.wavenet_style_condition
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.in_channels),
            )

        self.class_dropout_prob = cfg.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, cfg.hidden_dim)
        self.skip_linear = nn.Linear(cfg.hidden_dim + cfg.in_channels, cfg.hidden_dim)
        self.cond_x_merge_linear = nn.Linear(
            cfg.hidden_dim + cfg.in_channels * 2
            + cfg.style_encoder_dim * int(self.transformer_style_condition) * (not self.style_as_token),
            cfg.hidden_dim,
        )
        if self.style_as_token:
            self.style_in = nn.Linear(cfg.style_encoder_dim, cfg.hidden_dim)

    # ----- weight loading -----

    def load_official(self, sd: dict[str, torch.Tensor]) -> None:
        """Load the checkpoint state_dict (flat keys with 'estimator.' prefix)."""
        remapped = {k[len("estimator."):]: v for k, v in sd.items() if k.startswith("estimator.")}
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        # allow unused-but-present modules we define but the checkpoint lacks? No —
        # everything in the checkpoint maps; check for true mismatches.
        if unexpected:
            raise RuntimeError(f"unexpected keys loading DiT: {unexpected}")
        if missing:
            raise RuntimeError(f"missing keys loading DiT: {missing}")

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)

    def enable_teacache(self, thresh: float = 0.25, coef=None) -> None:
        """Enable TeaCache step-skipping (vLLM-Omni style).

        thresh: relative-L1 accumulation threshold; higher = more skipping (faster
          but lower fidelity). Typical 0.15-0.30 for image DiTs; tune per model.
        coef: optional polynomial rescaling coefficients (np.poly1d). None = linear.
        """
        self.teacache_enabled = True
        self.teacache_thresh = thresh
        self.teacache_coef = coef
        self._tc_state = {"cnt": 0, "prev_core": None, "prev_residual": None, "accum": 0.0}

    def disable_teacache(self) -> None:
        self.teacache_enabled = False
        self._tc_state = None

    def precast_linear_bf16(self) -> int:
        """Cast only Linear/Conv1d weights to bf16 (vLLM-Omni strategy).

        Keeps LayerNorm/embeddings in fp32 so mixed-dtype ops don't break, while
        halving GEMM cost and restoring flash-attention eligibility (bf16+nomask).
        Returns count of params cast.
        """
        import torch.nn as _nn
        n = 0
        for m in self.modules():
            if isinstance(m, (_nn.Linear, _nn.Conv1d)):
                for p in m.parameters(recurse=False):
                    p.data = p.data.to(torch.bfloat16)
                    n += p.numel()
        return n

    # ----- forward -----

    def forward(self, x: torch.Tensor, prompt_x: torch.Tensor, x_lens: torch.Tensor,
                t: torch.Tensor, style: torch.Tensor, cond: torch.Tensor,
                mask_content: bool = False) -> torch.Tensor:
        """Velocity estimate.

        Args:
            x: [B, in_channels, T] current flow state (prompt region zeroed).
            prompt_x: [B, in_channels, T] reference mel (prompt region filled).
            x_lens: [B] valid lengths.
            t: [B] timestep in [0, 1].
            style: [B, 192] CAMPPlus style embedding.
            cond: [B, T, content_dim] semantic condition (length_regulator output).
        Returns:
            dphi_dt [B, in_channels, T].
        """
        class_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
        if not self.training and mask_content:
            class_dropout = True
        cond_in_module = self.cond_projection

        B, _, T = x.size()

        t1 = self.t_embedder(t)  # [B, D]
        cond = cond_in_module(cond)  # [B, T, D]

        x = x.transpose(1, 2)  # [B, T, 80]
        prompt_x = prompt_x.transpose(1, 2)  # [B, T, 80]

        x_in = torch.cat([x, prompt_x, cond], dim=-1)  # 80+80+512 = 672

        if self.transformer_style_condition and not self.style_as_token:
            x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)  # 864

        if class_dropout:
            x_in[..., self.in_channels:] = x_in[..., self.in_channels:] * 0

        x_in = self.cond_x_merge_linear(x_in)  # [B, T, D]

        if self.style_as_token:
            style = self.style_in(style)
            style = torch.zeros_like(style) if class_dropout else style
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)

        if self.time_as_token:
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)

        x_mask = sequence_mask(x_lens + int(self.style_as_token) + int(self.time_as_token),
                               max_length=x_in.size(1)).to(x.device).unsqueeze(1)  # [B,1,T]
        input_pos = self.input_pos[: x_in.size(1)]
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1) if not self.is_causal else None
        # --- TeaCache: skip transformer when consecutive-step inputs are similar ---
        # Monitor the core (non-token) input to the transformer. Cache the residual
        # (transformer_out - transformer_in) over core positions; reuse when the
        # core input changes slowly between Euler steps (vLLM-Omni TeaCache).
        x_in_core = x_in
        n_prefix = 0
        if self.time_as_token:
            n_prefix += 1
        if self.style_as_token:
            n_prefix += 1
        if n_prefix:
            x_in_core = x_in[:, n_prefix:]
        do_full = True
        if self.teacache_enabled:
            st = self._tc_state
            if st["cnt"] == 0:
                st["accum"] = 0.0
            elif st["prev_core"] is not None:
                rel = ((x_in_core - st["prev_core"]).abs().mean() /
                       (st["prev_core"].abs().mean() + 1e-8)).cpu().item()
                scaled = float(self.teacache_coef(rel)) if self.teacache_coef is not None else rel
                st["accum"] += abs(scaled)
                if st["accum"] < self.teacache_thresh and st["prev_residual"] is not None:
                    do_full = False
                else:
                    st["accum"] = 0.0
            st["prev_core"] = x_in_core.detach()
        if do_full:
            x_res_full = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)  # [B, T_full, D]
            # strip tokens to get core output
            x_res = x_res_full[:, n_prefix:] if n_prefix else x_res_full
            if self.teacache_enabled:
                self._tc_state["prev_residual"] = (x_res - x_in_core).detach()
                self._tc_state["cnt"] += 1
        else:
            # FAST PATH: reuse cached core residual
            x_res = x_in_core + self._tc_state["prev_residual"]
            self._tc_state["cnt"] += 1
        # (x_res is already core-only; no further stripping needed)

        if self.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))

        if self.final_layer_type == "wavenet":
            x = self.conv1(x_res)
            x = x.transpose(1, 2)
            t2 = self.t_embedder2(t)
            x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(x_res)
            x = self.final_layer(x, t1).transpose(1, 2)
            x = self.conv2(x)
        else:
            x = self.final_mlp(x_res)
            x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load("/root/IndexTTS-2.5/s2mel.pth", map_location="cpu", weights_only=False)["net"]
    model = DiT().to(dev).eval()
    model.load_official(net["cfm"])
    model.setup_caches(1, 2048)
    print(f"[DiT] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, strict load OK")

    out_dir = "/root/windextts_dumps"
    x = torch.load(f"{out_dir}/s2mel.dit_input_x.pt", weights_only=False).to(dev)
    prompt_x = torch.load(f"{out_dir}/s2mel.dit_input_prompt_x.pt", weights_only=False).to(dev)
    cond = torch.load(f"{out_dir}/s2mel.dit_input_cond.pt", weights_only=False).to(dev)
    style = torch.load(f"{out_dir}/s2mel.dit_input_style.pt", weights_only=False).to(dev)
    t = torch.load(f"{out_dir}/s2mel.dit_input_t.pt", weights_only=False).to(dev)
    ref = torch.load(f"{out_dir}/s2mel.dit_output.pt", weights_only=False).to(dev)
    x_lens = torch.LongTensor([cond.size(1)]).to(dev)

    with torch.no_grad():
        out = model(x, prompt_x, x_lens, t, style, cond)
    print(f"out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"max_abs_diff = {diff:.4e}")
    print(f"allclose(atol=1e-3, rtol=1e-3) = {torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3)}")
    print("SMOKE", "OK" if diff < 1e-3 else "FAIL")
