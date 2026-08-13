"""Qwen3 (0.6B) pure-torch inference — for IndexTTS QwenEmotion text→emotion.

A self-contained reimplementation of the Qwen3-ForCausalLM forward, sufficient
to replicate the official ``QwenEmotion.inference`` text-classification output.
No transformers/modelscope dependency; weights loaded via safetensors only.

Architecture (qwen0.6bemo4-merge/config.json):
  - 28 layers, hidden 1024, 16 attention heads, 8 KV heads (GQA, 2:1)
  - head_dim 128, RoPE theta 1e6, RMSNorm eps 1e-6
  - SwiGLU MLP (intermediate 3072)
  - Qwen3-specific: per-head RMSNorm on q and k (q_norm, k_norm)
  - tie_word_embeddings: lm_head reuses embed_tokens.weight

The forward supports both prefill (full sequence) and single-token decode
(KV-cache append), enough for greedy generation of short JSON outputs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# RoPE (rotary position embedding) — matches Qwen3 config (theta=1e6)
# ---------------------------------------------------------------------------

def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Returns (cos, sin) each of shape [max_seq_len, head_dim] — the standard
    GPT-NeoX / LLaMA-style half-rotation layout (first half paired with second
    half, NOT interleaved). Qwen3 uses this layout.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)  # [seq, head_dim/2]
    # duplicate to full head_dim (cos/sin applied to paired halves)
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, n_heads, T, head_dim].

    cos/sin: [T, head_dim]. Uses the rotate-half formulation:
        out = x * cos + rotate_half(x) * sin
    where rotate_half([a,b]) = [-b, a] (second half negated, swapped).
    """
    x1, x2 = x.chunk(2, dim=-1)
    # cos/sin broadcast over heads; match x dtype to avoid promotion
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)  # [1,1,T,head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm (Qwen3 uses eps=1e-6). Computes in fp32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 accumulation for numerical stability (matches transformers)
        orig = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(orig) * self.weight)


class Qwen3Attention(nn.Module):
    """GQA self-attention with RoPE + per-head q/k RMSNorm (Qwen3-specific)."""

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # GQA repeat factor
        q_out = n_heads * head_dim
        kv_out = n_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=False)
        # Qwen3 per-head RMSNorm (applied to each head's head_dim vector)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list[torch.Tensor]] = None,
        cache_pos=0,
        cache_pos_is_tensor: bool = False,
    ) -> torch.Tensor:
        """Attention with optional KV cache.

        cache_pos: int (eager) or 0-d device tensor (CUDA Graph). When tensor,
        KV write uses scatter (index_copy_) and attention uses a full-size
        mask built from cache_pos (graph-safe, no dynamic slicing).
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Qwen3: per-head RMSNorm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if kv_cache is not None:
            kc, vc = kv_cache  # pre-allocated [B,n_kv,max_seq,hd]
            max_seq = kc.size(2)
            if cache_pos_is_tensor:
                # graph path: scatter k/v at [cache_pos, cache_pos+T) via index_copy_
                pos_idx = torch.arange(T, device=x.device) + cache_pos
                kc.index_copy_(2, pos_idx, k)
                vc.index_copy_(2, pos_idx, v)
                # attention over full max_seq with additive mask hiding
                # positions >= cache_pos+T. Build mask without torch.where
                # (not capture-safe with Python -inf literal): use a
                # precomputed arange comparison expressed as a sub/mask.
                k_all = kc
                v_all = vc
                if mask is None:
                    positions = torch.arange(max_seq, device=x.device)
                    valid = cache_pos + T
                    # additive bias: 0 where valid, large-negative elsewhere.
                    # (positions >= valid) * -1e4 avoids torch.where + -inf.
                    attn_bias = ((positions >= valid).to(q.dtype) * -1e4)
                    mask = attn_bias.view(1, 1, 1, max_seq)
            else:
                # eager path: int slice
                cp = int(cache_pos)
                kc[:, :, cp:cp+T] = k
                vc[:, :, cp:cp+T] = v
                k_all = kc[:, :, :cp+T]
                v_all = vc[:, :, :cp+T]
            k = k_all
            v = v_all

        # GQA: repeat k/v to match n_heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # SDPA (flash/mem-efficient path via PyTorch)
        # q: [B, n_heads, T, head_dim], k/v: [B, n_heads, S, head_dim]
        if mask is not None:
            # mask: [1,1,T,S] additive (-inf for masked)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # causal for prefill when no explicit mask passed
            out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP with residuals."""

    def __init__(self, config: dict):
        super().__init__()
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkv = config["num_key_value_heads"]
        hd = config.get("head_dim", hs // nh)
        inter = config["intermediate_size"]
        eps = config.get("rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hs, eps)
        self.self_attn = Qwen3Attention(hs, nh, nkv, hd)
        self.post_attention_layernorm = RMSNorm(hs, eps)
        self.mlp = Qwen3MLP(hs, inter)

    def forward(
        self, x, rope_cos, rope_sin, mask=None, kv_cache=None, cache_pos=0, cache_pos_is_tensor=False
    ) -> torch.Tensor:
        h = x + self.self_attn(
            self.input_layernorm(x), rope_cos, rope_sin, mask, kv_cache, cache_pos, cache_pos_is_tensor
        )
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Qwen3ForCausalLM(nn.Module):
    """Pure-torch Qwen3-ForCausalLM for short greedy generation (QwenEmotion)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hs = config["hidden_size"]
        eps = config.get("rms_norm_eps", 1e-6)
        self.embed_tokens = nn.Embedding(config["vocab_size"], hs)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = RMSNorm(hs, eps)
        self.tie_word_embeddings = config.get("tie_word_embeddings", True)
        # RoPE cache (filled lazily)
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

    @property
    def lm_head(self) -> nn.Linear:
        """Tied weights: lm_head reuses embed_tokens."""
        return self.embed_tokens

    def _ensure_rope(self, seq_len: int, device, dtype):
        if self._rope_cos is None or self._rope_cos.size(0) < seq_len:
            hd = self.config.get("head_dim", self.config["hidden_size"] // self.config["num_attention_heads"])
            theta = self.config.get("rope_theta", 1_000_000.0)
            cos, sin = precompute_rope_cache(hd, max(seq_len, 4096), theta, device, torch.float32)
            self._rope_cos = cos
            self._rope_sin = sin
        self._rope_cos = self._rope_cos.to(device)
        self._rope_sin = self._rope_sin.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[list[torch.Tensor]]] = None,
        cache_pos=0,
        cache_pos_is_tensor: bool = False,
    ) -> torch.Tensor:
        """Forward pass. Returns logits [B, T, vocab].

        kv_caches: list of [k_cache, v_cache] per layer, or None for prefill.
        cache_pos: int (eager) or 0-d device tensor (CUDA Graph decode).
        cache_pos_is_tensor: must be True when cache_pos is a device tensor.
        """
        B, T = input_ids.shape
        device = input_ids.device
        if cache_pos_is_tensor:
            # graph capture: cannot use int() (host sync). Pre-allocate rope
            # to a safe max; actual gather handles the position.
            need = 8192
            self._ensure_rope(need, device, torch.float32)
            self._rope_cos = self._rope_cos.to(device)
            self._rope_sin = self._rope_sin.to(device)
            x = self.embed_tokens(input_ids)
            idx = torch.arange(T, device=device) + cache_pos
            rope_cos = self._rope_cos.index_select(0, idx)
            rope_sin = self._rope_sin.index_select(0, idx)
        else:
            need = cache_pos + T
            self._ensure_rope(need, device, torch.float32)
            self._rope_cos = self._rope_cos.to(device)
            self._rope_sin = self._rope_sin.to(device)
            x = self.embed_tokens(input_ids)
            rope_cos = self._rope_cos[cache_pos : cache_pos + T]
            rope_sin = self._rope_sin[cache_pos : cache_pos + T]

        # causal mask for prefill (T>1). For decode (T==1) with tensor cache_pos,
        # build the KV-validity mask ONCE here (not per-layer) — saves 28× the
        # arange+compare kernel launches per decode step.
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif cache_pos_is_tensor and kv_caches is not None:
            # decode mask: positions >= (cache_pos+T) get -1e4 additive bias.
            # Reuse a static positions buffer if available (graph-friendly).
            max_seq = kv_caches[0][0].size(2)
            positions = torch.arange(max_seq, device=device)
            valid = cache_pos + T
            mask = ((positions >= valid).to(x.dtype) * -1e4).view(1, 1, 1, max_seq)

        for i, layer in enumerate(self.layers):
            kc = kv_caches[i] if kv_caches is not None else None
            x = layer(x, rope_cos, rope_sin, mask, kc, cache_pos, cache_pos_is_tensor)

        x = self.norm(x)
        logits = F.linear(x, self.lm_head.weight)  # [B, T, vocab]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        eos_token_id: int = 151643,
        use_cuda_graph: bool = False,
    ) -> torch.Tensor:
        """Greedy generation. Returns full [B, input_len + gen_len].

        With use_cuda_graph=True, the decode step is captured into a CUDA
        Graph: cache_pos flows through a device tensor (index_select RoPE +
        index_copy_ KV scatter + dynamic mask), eliminating the 85% launch-
        overhead idle (~29ms→~3ms per token, ~2.2s→~0.3s total).
        """
        device = input_ids.device
        B = input_ids.size(0)
        n_kv = self.config["num_key_value_heads"]
        hd = self.config.get("head_dim", 128)
        model_dtype = next(self.parameters()).dtype
        prompt_len = input_ids.size(1)
        total_max = prompt_len + max_new_tokens

        # pre-allocate KV caches to full length (fill mode, graph-safe)
        kv_caches = [
            [torch.zeros(B, n_kv, total_max, hd, device=device, dtype=model_dtype),
             torch.zeros(B, n_kv, total_max, hd, device=device, dtype=model_dtype)]
            for _ in self.layers
        ]

        # --- prefill (eager, one-time) ---
        logits = self.forward(input_ids, kv_caches, cache_pos=0)
        next_token = logits[:, -1, :].argmax(dim=-1)  # [B]
        gen_tokens = [next_token]
        cache_pos = prompt_len

        if not use_cuda_graph or device.type != "cuda":
            # eager decode fallback
            for _ in range(max_new_tokens - 1):
                if next_token.item() == eos_token_id:
                    break
                step_ids = next_token.unsqueeze(1)  # [B,1]
                logits = self.forward(step_ids, kv_caches, cache_pos=cache_pos)
                next_token = logits[:, -1, :].argmax(dim=-1)
                gen_tokens.append(next_token)
                cache_pos += 1
            return torch.cat([input_ids, torch.stack(gen_tokens, dim=1)], dim=1)

        # --- CUDA Graph decode (cache_pos as device tensor) ---
        # All dynamic state flows through device buffers (no host sync):
        #   tok_buf: [B,1] next input token id
        #   pos_buf: [1] device long = current cache_pos
        # The forward does index_select (RoPE) + index_copy_ (KV) + where-mask,
        # all device ops → graph-captureable.
        tok_buf = next_token.unsqueeze(1).clone()
        pos_buf = torch.tensor([cache_pos], device=device, dtype=torch.long)
        out_buf = torch.zeros(B, dtype=torch.long, device=device)

        def decode_step():
            lg = self.forward(tok_buf, kv_caches, cache_pos=pos_buf[0], cache_pos_is_tensor=True)
            out_buf.copy_(lg[:, -1, :].argmax(dim=-1))

        # warmup (advances pos_buf to exercise fill positions) before capture
        for _ in range(3):
            decode_step()
            pos_buf += 1
            tok_buf.copy_(out_buf.unsqueeze(1))
        torch.cuda.synchronize()

        # reset to real starting state, then capture
        pos_buf.fill_(cache_pos)
        tok_buf.copy_(next_token.unsqueeze(1))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_step()

        # replay loop: each replay = one decode step. The per-step .item()
        # eos check IS a host sync, but it enables early termination (emotion
        # JSON ends at eos ~76 tokens, far below max_new_tokens=150), which
        # saves more than the sync costs.
        for _ in range(max_new_tokens - 1):
            graph.replay()
            tok_buf.copy_(out_buf.unsqueeze(1))
            pos_buf += 1
            gen_tokens.append(out_buf.clone())
            if out_buf.item() == eos_token_id:
                break

        return torch.cat([input_ids, torch.stack(gen_tokens, dim=1)], dim=1)


# ---------------------------------------------------------------------------
# Weight loading (safetensors, HF naming)
# ---------------------------------------------------------------------------

def load_qwen3(model_dir: str | Path, device: str = "cuda", dtype: torch.dtype = torch.float16) -> Qwen3ForCausalLM:
    """Load a Qwen3 model from a HF-format directory (config.json + model.safetensors).

    No transformers dependency: config read as plain JSON, weights via safetensors.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    model = Qwen3ForCausalLM(config)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")

    # Map HF parameter names to our module names (they match 1:1 for Qwen3)
    own = dict(model.named_parameters())
    own_buffers = dict(model.named_buffers())
    loaded, skipped = 0, []
    for k, v in state.items():
        # HF keys are prefixed with "model." (e.g. model.embed_tokens.weight,
        # model.layers.0..., model.norm.weight). Our top-level module names
        # match after stripping that prefix.
        target_key = k
        if target_key not in own:
            target_key = k.removeprefix("model.")
        if target_key in own:
            own[target_key].data.copy_(v.to(own[target_key].dtype))
            loaded += 1
        else:
            skipped.append(k)
    assert loaded == len(own), f"loaded {loaded}/{len(own)} params, missing keys. skipped sample: {skipped[:5]}"

    model = model.to(device).to(dtype)
    model.eval()
    return model
