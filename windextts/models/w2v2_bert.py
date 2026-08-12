"""w2v-bert-2.0 conformer encoder — pure-torch re-implementation.

Replaces ``transformers.Wav2Vec2BertModel`` for the IndexTTS-2.5 ref-audio
feature path. Zero transformers/librosa dependency.

Verified contract (all from the official HF modeling_wav2vec2_bert.py, lines
cited inline): 24-layer macaron conformer, hidden 1024, 16 heads (head_size 64),
intermediate 4096, conv_depthwise_kernel 31, ``position_embeddings_type="relative_key"``.
We extract ``hidden_states[17]`` (the 16th encoder layer output; index 0 is the
feature_projection output) — the IndexTTS-2.5 seam (infer_v2_5.py:288).

Key implementation points (each grounded in HF source):
  - feature_projection: LayerNorm(160) -> Linear(160, 1024). (FP:118-129)
  - encoder: NO explicit positional embedding for relative_key (embed_positions=None,
    Wav2Vec2BertEncoder:470-471). Position info lives only in the attention's
    distance_embedding (clamped relative-key bias).
  - attention (relative_key, Attention:333-348):
      scores = Q@K^T / sqrt(d)                                  # standard
      distance = clamp(j - i, -left_max=64, right_max=8)        # [Tq, Tk]
      pos_emb = distance_embedding(distance + 64)               # [Tq, Tk, 64]
      scores += einsum("bhld,lrd->bhlr", Q, pos_emb) / sqrt(d)
      scores += additive_mask
      out = softmax(scores) @ V
    num_positions = 64+8+1 = 73, head_size = 64  ->  distance_embedding [73, 64].
  - macaron layer (EncoderLayer:422-464):
      h = x + 0.5*ffn1(ln1(x))
      h = h + dropout(attn(ln2(h)))
      h = h + conv_module(h)            # conv module does its own internal LN
      h = h + 0.5*ffn2(ln3(h))
      out = final_layer_norm(h)
  - conv module (ConvolutionModule:195-226): internal LN -> pointwise1(1024->2048)
    -> GLU(dim=1) -> **left-pad (kernel-1=30)** causal depthwise(31, groups=1024)
    -> depthwise_layer_norm -> swish -> pointwise2 -> transpose.
    NOTE the depthwise padding is LEFT-only (causal), despite conformer norm —
    this is the HF implementation's exact behavior (ConvolutionModule:213-215).
  - hidden_states list (Encoder:519,540): [0]=feature_projection out,
    [i]=layer(i-1) output for i>=1. So hidden_states[17] = layer 16 output.
  - eval path: apply_spec_augment=False -> _mask_hidden_states is a no-op;
    add_adapter=False, use_intermediate_ffn_before_adapter=False.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Wav2Vec2BertConformer", "Wav2Vec2BertConfig"]


class Wav2Vec2BertConfig:
    """Config values for the IndexTTS-2.5 w2v-bert-2.0 (from its config.json)."""

    hidden_size = 1024
    num_hidden_layers = 24
    num_attention_heads = 16
    intermediate_size = 4096
    conv_depthwise_kernel_size = 31
    feature_projection_input_dim = 160
    layer_norm_eps = 1e-5
    hidden_act = "swish"
    # relative_key positional encoding window (config.json)
    left_max_position_embeddings = 64
    right_max_position_embeddings = 8

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 64

    @property
    def num_positions(self) -> int:
        return self.left_max_position_embeddings + self.right_max_position_embeddings + 1  # 73


# ---------------------------------------------------------------------------
# Sub-modules (faithful copies of HF naming so load_state_dict is strict).
# ---------------------------------------------------------------------------


class Wav2Vec2BertFeedForward(nn.Module):
    """FFN: Linear(hs->is) -> act -> Linear(is->hs). (HF:133-152)"""

    def __init__(self, cfg: Wav2Vec2BertConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output_dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate_dense(x)
        x = F.silu(x)  # hidden_act="swish"
        x = self.output_dense(x)
        return x


class Wav2Vec2BertSelfAttention(nn.Module):
    """Multi-head self-attention with relative-key position bias. (HF:262-360)

    For relative_key: scores = Q@K^T/sqrt(d) + einsum(Q, dist_emb)/sqrt(d) + mask.
    distance = clamp(j-i, -left_max, right_max); pos lookup = distance_embedding(dist+left_max).
    """

    def __init__(self, cfg: Wav2Vec2BertConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_size = cfg.head_size
        self.left_max = cfg.left_max_position_embeddings
        self.right_max = cfg.right_max_position_embeddings
        self.distance_embedding = nn.Embedding(cfg.num_positions, self.head_size)
        self.linear_q = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.linear_k = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.linear_v = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.linear_out = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.linear_q(hidden_states).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.linear_k(hidden_states).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.linear_v(hidden_states).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        # q,k,v: [B, H, T, head_size]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # relative-key position bias: distance = j(key) - i(query), clamped.
        pos_l = torch.arange(T, device=hidden_states.device).view(-1, 1)
        pos_r = torch.arange(T, device=hidden_states.device).view(1, -1)
        distance = torch.clamp(pos_r - pos_l, -self.left_max, self.right_max)  # [Tq, Tk]
        pos_emb = self.distance_embedding(distance + self.left_max)  # [Tq, Tk, head_size]
        pos_emb = pos_emb.to(dtype=q.dtype)
        rel = torch.einsum("bhld,lrd->bhlr", q, pos_emb)  # [B, H, Tq, Tk]
        scores = scores + rel / math.sqrt(self.head_size)

        if attention_mask is not None:
            scores = scores + attention_mask  # additive [B,1,T,T] or broadcastable

        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # [B, H, T, head_size]
        ctx = ctx.transpose(1, 2).reshape(B, T, self.num_heads * self.head_size)
        return self.linear_out(ctx)


class Wav2Vec2BertConvolutionModule(nn.Module):
    """Conformer conv module with causal left-pad. (HF:156-226)"""

    def __init__(self, cfg: Wav2Vec2BertConfig):
        super().__init__()
        hs = cfg.hidden_size
        k = cfg.conv_depthwise_kernel_size
        self.layer_norm = nn.LayerNorm(hs, eps=cfg.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(hs, 2 * hs, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(hs, hs, k, stride=1, padding=0, groups=hs, bias=False)
        self.depthwise_layer_norm = nn.LayerNorm(hs, eps=cfg.layer_norm_eps)
        self.pointwise_conv2 = nn.Conv1d(hs, hs, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conv_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.layer_norm(x)
        # conv_module receives the SAME additive-extended mask naming as attn in HF;
        # but here conv uses a boolean mask (conv_attention_mask = original attention_mask).
        if conv_attention_mask is not None:
            x = x.masked_fill(~conv_attention_mask.bool().unsqueeze(-1), 0.0)
        x = x.transpose(1, 2)  # [B, hs, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # [B, hs, T]
        # causal left-pad (kernel-1) — HF pads left only.
        x = F.pad(x, (self.depthwise_conv.kernel_size[0] - 1, 0))
        x = self.depthwise_conv(x)
        x = self.depthwise_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # [B, T, hs]
        return x


class Wav2Vec2BertEncoderLayer(nn.Module):
    """Macaron conformer block. (HF:397-464)"""

    def __init__(self, cfg: Wav2Vec2BertConfig):
        super().__init__()
        hs = cfg.hidden_size
        self.ffn1_layer_norm = nn.LayerNorm(hs, eps=cfg.layer_norm_eps)
        self.ffn1 = Wav2Vec2BertFeedForward(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(hs, eps=cfg.layer_norm_eps)
        self.self_attn = Wav2Vec2BertSelfAttention(cfg)
        self.conv_module = Wav2Vec2BertConvolutionModule(cfg)
        self.ffn2_layer_norm = nn.LayerNorm(hs, eps=cfg.layer_norm_eps)
        self.ffn2 = Wav2Vec2BertFeedForward(cfg)
        self.final_layer_norm = nn.LayerNorm(hs, eps=cfg.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conv_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. FFN1 (macaron half-residual)
        residual = x
        x = self.ffn1_layer_norm(x)
        x = self.ffn1(x)
        x = residual + x * 0.5

        # 2. self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attention_mask=attention_mask)
        x = residual + x

        # 3. conv module
        residual = x
        x = self.conv_module(x, conv_attention_mask=conv_attention_mask)
        x = residual + x

        # 4. FFN2 (macaron half-residual) + final LN
        residual = x
        x = self.ffn2_layer_norm(x)
        x = self.ffn2(x)
        x = residual + x * 0.5
        x = self.final_layer_norm(x)
        return x


class Wav2Vec2BertFeatureProjection(nn.Module):
    """LayerNorm(160) -> Linear(160, 1024). (HF:118-129)"""

    def __init__(self, cfg: Wav2Vec2BertConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.feature_projection_input_dim, eps=cfg.layer_norm_eps)
        self.projection = nn.Linear(cfg.feature_projection_input_dim, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.layer_norm(x))


class Wav2Vec2BertConformer(nn.Module):
    """Full w2v-bert-2.0 encoder for IndexTTS-2.5.

    Loads the official ``model.safetensors`` (773 keys) with strict=True.
    forward(input_features, attention_mask) -> hidden_states[17] by default
    (the IndexTTS-2.5 seam). Pass ``return_layer=`` for other layers, or
    ``None`` for all 25 hidden states (index 0 = feature_projection out).
    """

    def __init__(self, cfg: Wav2Vec2BertConfig | None = None):
        super().__init__()
        self.cfg = cfg or Wav2Vec2BertConfig()
        self.feature_projection = Wav2Vec2BertFeatureProjection(self.cfg)
        self.encoder_layers = nn.ModuleList(
            [Wav2Vec2BertEncoderLayer(self.cfg) for _ in range(self.cfg.num_hidden_layers)]
        )

    # ----- state_dict adapter: official keys map 1:1 except 'encoder.layers' -----
    # official: encoder.layers.{i}.*  + feature_projection.* + masked_spec_embed
    # We expose encoder_layers (not encoder.layers) so we remap on load/save.

    def _remap_state_dict(self, sd: dict[str, torch.Tensor], reverse: bool = False) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if reverse:
                if k.startswith("encoder_layers."):
                    k = "encoder." + k[len("encoder_layers."):]
                else:
                    continue  # only encoder subkeys when reversing
            else:
                # official 'encoder.layers.{i}.*' -> our 'encoder_layers.{i}.*'
                # (nn.ModuleList exposes keys as encoder_layers.{i}, no '.layers.')
                if k.startswith("encoder.layers."):
                    k = "encoder_layers." + k[len("encoder.layers."):]
                elif k == "masked_spec_embed":
                    continue  # not used in inference (apply_spec_augment=False)
            out[k] = v
        return out

    def load_official(self, sd: dict[str, torch.Tensor]) -> None:
        """Load official HF safetensors state_dict (strict)."""
        remapped = self._remap_state_dict(sd)
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        # feature_projection.* should all load; only masked_spec_embed dropped.
        if unexpected:
            raise RuntimeError(f"unexpected keys loading w2v-bert: {unexpected}")
        if missing:
            raise RuntimeError(f"missing keys loading w2v-bert: {missing}")

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_layer: int | None = 17,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the conformer and return hidden_states[return_layer] (default 17).

        Args:
            input_features: [B, T, 160] from SeamlessM4TFeaturizer.
            attention_mask: [B, T] int/bool (1=valid). If None, all valid.
            return_layer: which hidden state to return. 0=feature_projection out,
                i>=1 = encoder layer (i-1) output. None = return all 25.
        """
        h = self.feature_projection(input_features)  # [B, T, 1024]  == hidden_states[0]
        all_hs: list[torch.Tensor] | None = [] if return_layer is None else None
        if all_hs is not None:
            all_hs.append(h)
        if return_layer == 0:
            return h

        # additive attention mask (HF Encoder:491-499) + boolean conv mask
        attn_mask, conv_mask = None, None
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=h.dtype)) * torch.finfo(h.dtype).min
            attn_mask = attn_mask.expand(-1, 1, attention_mask.shape[1], attention_mask.shape[1])
            conv_mask = attention_mask

        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, attention_mask=attn_mask, conv_attention_mask=conv_mask)
            if all_hs is not None:
                all_hs.append(h)
            elif return_layer == i + 1:
                return h  # hidden_states[return_layer] captured, early exit

        if all_hs is not None:
            return all_hs
        return h  # return_layer == num_hidden_layers (last layer)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")
    import torchaudio
    from windextts.weights import WeightLoader
    from windextts.frontend.audio_utils import SeamlessM4TFeaturizer

    # Smoke: load strict, run on GPU, compare hidden_states[17] against an official
    # fp32 model run on the SAME device (fair parity). (The IndexTTS dumps were taken
    # with use_bf16=True and are NOT a fair target for our fp32 model — use the
    # official model directly.)
    assert torch.cuda.is_available(), "smoke test assumes CUDA"
    dev = "cuda"
    w = WeightLoader()
    model = Wav2Vec2BertConformer().to(dev)
    model.load_official(w.load_w2v_bert())
    model.eval()
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, strict OK")

    fe = SeamlessM4TFeaturizer(device=dev)
    audio, sr = torchaudio.load("/root/WIndexTTS/test.wav")
    a16 = torchaudio.transforms.Resample(sr, 16000)(audio).to(dev)
    inp = fe(a16)
    am = torch.ones(inp.shape[:2], dtype=torch.int32, device=dev)
    with torch.no_grad():
        out = model(inp, am, return_layer=17)
    print(f"hidden_states[17]: {tuple(out.shape)} {out.dtype}")

    from transformers import Wav2Vec2BertModel

    off = Wav2Vec2BertModel.from_pretrained(
        "/root/IndexTTS-2.5/hf_cache/w2v-bert-2.0", local_files_only=True
    ).to(dev).eval()
    with torch.no_grad():
        ref = off(input_features=inp, attention_mask=am, output_hidden_states=True).hidden_states[17]
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"max_abs_diff vs official fp32 (same GPU) = {diff:.3e}")
    print(f"allclose(atol=1e-4, rtol=1e-3) = {torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-3)}")
    print("SMOKE", "OK" if diff < 1e-3 else "FAIL")
