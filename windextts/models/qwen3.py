"""Qwen3-0.6B pure-torch forward for QwenEmotion text→emotion (no transformers).
28 layers, hidden 1024, 16 heads / 8 KV heads (GQA), head_dim 128, RoPE theta
1e6, RMSNorm eps 1e-6, SwiGLU (inter 3072), per-head q/k RMSNorm, tied lm_head.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def precompute_rope_cache(head_dim, max_seq_len, theta=1_000_000.0, device="cpu", dtype=torch.float32):
    # GPT-NeoX half-rotation layout (halves paired, NOT interleaved) — Qwen3 uses this
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len, device=device, dtype=dtype), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # rotate-half: out = x*cos + rotate_half(x)*sin, rotate_half([a,b])=[-b,a]
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)  # [1,1,T,hd] cast avoids promotion
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # fp32 accumulation for stability (matches transformers)
        orig = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(orig) * self.weight


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.n_rep = n_heads // n_kv_heads
        qo, ko = n_heads * head_dim, n_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, qo, bias=False)
        self.k_proj = nn.Linear(hidden_size, ko, bias=False)
        self.v_proj = nn.Linear(hidden_size, ko, bias=False)
        self.o_proj = nn.Linear(qo, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)  # Qwen3 per-head q/k RMSNorm (before RoPE)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, rope_cos, rope_sin, mask=None, kv_cache=None, cache_pos=0, cache_pos_is_tensor=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k), rope_cos, rope_sin)

        if kv_cache is not None:
            kc, vc = kv_cache  # [B, n_kv, max_seq, hd]
            max_seq = kc.size(2)
            if cache_pos_is_tensor:
                # graph path: index_copy_ scatter + full-size additive mask
                # (no dynamic slicing; -1e4 not -inf: capture-safe, no Python literal)
                pos_idx = torch.arange(T, device=x.device) + cache_pos
                kc.index_copy_(2, pos_idx, k)
                vc.index_copy_(2, pos_idx, v)
                k, v = kc, vc  # attend over FULL buffers; mask hides positions >= cache_pos+T
                if mask is None:
                    valid = cache_pos + T
                    mask = ((torch.arange(max_seq, device=x.device) >= valid).to(q.dtype) * -1e4).view(1, 1, 1, max_seq)
            else:
                cp = int(cache_pos)
                kc[:, :, cp:cp + T] = k
                vc[:, :, cp:cp + T] = v
                k, v = kc[:, :, :cp + T], vc[:, :, :cp + T]

        if self.n_rep > 1:  # GQA: repeat k/v to n_heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and T > 1)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hs, nh = config["hidden_size"], config["num_attention_heads"]
        nkv, hd = config["num_key_value_heads"], config.get("head_dim", hs // nh)
        inter, eps = config["intermediate_size"], config.get("rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hs, eps)
        self.self_attn = Qwen3Attention(hs, nh, nkv, hd)
        self.post_attention_layernorm = RMSNorm(hs, eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hs, inter, bias=False)
        self.mlp.up_proj = nn.Linear(hs, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hs, bias=False)

    def forward(self, x, rope_cos, rope_sin, mask=None, kv_cache=None, cache_pos=0, cache_pos_is_tensor=False):
        h = x + self.self_attn(self.input_layernorm(x), rope_cos, rope_sin, mask, kv_cache, cache_pos, cache_pos_is_tensor)
        return h + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))


class Qwen3ForCausalLM(nn.Module):
    """Greedy short-JSON generation for QwenEmotion (prefill + KV decode + CUDA Graph)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config) for _ in range(config["num_hidden_layers"]))
        self.norm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-6))
        self._rope_cos = self._rope_sin = None

    @property
    def lm_head(self):  # tied weights
        return self.embed_tokens

    def _ensure_rope(self, seq_len, device):
        if self._rope_cos is None or self._rope_cos.size(0) < seq_len:
            hd = self.config.get("head_dim", self.config["hidden_size"] // self.config["num_attention_heads"])
            self._rope_cos, self._rope_sin = precompute_rope_cache(
                hd, max(seq_len, 4096), self.config.get("rope_theta", 1_000_000.0), device, torch.float32)
        self._rope_cos = self._rope_cos.to(device)
        self._rope_sin = self._rope_sin.to(device)

    def forward(self, input_ids, kv_caches=None, cache_pos=0, cache_pos_is_tensor=False):
        B, T = input_ids.shape
        device = input_ids.device
        if cache_pos_is_tensor:
            # graph capture: no host sync (int()) allowed — rope pre-allocated to
            # a safe max, positions gathered via index_select from the device pos
            self._ensure_rope(8192, device)
            x = self.embed_tokens(input_ids)
            idx = torch.arange(T, device=device) + cache_pos
            rope_cos = self._rope_cos.index_select(0, idx)
            rope_sin = self._rope_sin.index_select(0, idx)
        else:
            self._ensure_rope(cache_pos + T, device)
            x = self.embed_tokens(input_ids)
            rope_cos = self._rope_cos[cache_pos:cache_pos + T]
            rope_sin = self._rope_sin[cache_pos:cache_pos + T]

        mask = None
        if T > 1:
            mask = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=x.dtype), diagonal=1).unsqueeze(0).unsqueeze(0)
        elif cache_pos_is_tensor and kv_caches is not None:
            # decode mask built once at model level (not 28× per step):
            # positions >= cache_pos+T get -1e4; 0-d cache_pos keeps it device-side
            max_seq = kv_caches[0][0].size(2)
            mask = ((torch.arange(max_seq, device=device) >= cache_pos + T).to(x.dtype) * -1e4).view(1, 1, 1, max_seq)

        for i, layer in enumerate(self.layers):
            x = layer(x, rope_cos, rope_sin, mask, kv_caches[i] if kv_caches is not None else None, cache_pos, cache_pos_is_tensor)

        return F.linear(self.norm(x), self.lm_head.weight)  # [B, T, vocab]

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, eos_token_id=151643, use_cuda_graph=False):
        device = input_ids.device
        B = input_ids.size(0)
        n_kv = self.config["num_key_value_heads"]
        hd = self.config.get("head_dim", 128)
        total_max = input_ids.size(1) + max_new_tokens
        # pre-allocated full-length KV (fill mode, graph-safe)
        kv_caches = [[torch.zeros(B, n_kv, total_max, hd, device=device, dtype=next(self.parameters()).dtype)
                      for _ in range(2)] for _ in self.layers]

        logits = self.forward(input_ids, kv_caches, cache_pos=0)
        next_token = logits[:, -1, :].argmax(dim=-1)
        gen_tokens = [next_token]
        cache_pos = input_ids.size(1)

        if not use_cuda_graph or device.type != "cuda":
            for _ in range(max_new_tokens - 1):
                if next_token.item() == eos_token_id:
                    break
                logits = self.forward(next_token.unsqueeze(1), kv_caches, cache_pos=cache_pos)
                next_token = logits[:, -1, :].argmax(dim=-1)
                gen_tokens.append(next_token)
                cache_pos += 1
            return torch.cat([input_ids, torch.stack(gen_tokens, dim=1)], dim=1)

        # CUDA Graph decode: all dynamic state in device buffers (no host sync);
        # index_select RoPE + index_copy_ KV + device mask are capture-safe.
        tok_buf = next_token.unsqueeze(1).clone()
        pos_buf = torch.tensor([cache_pos], device=device, dtype=torch.long)
        out_buf = torch.zeros(B, dtype=torch.long, device=device)

        def decode_step():
            lg = self.forward(tok_buf, kv_caches, cache_pos=pos_buf[0], cache_pos_is_tensor=True)
            out_buf.copy_(lg[:, -1, :].argmax(dim=-1))

        for _ in range(3):  # warmup: advance pos_buf to exercise fill positions
            decode_step()
            pos_buf += 1
            tok_buf.copy_(out_buf.unsqueeze(1))
        torch.cuda.synchronize()

        pos_buf.fill_(cache_pos)
        tok_buf.copy_(next_token.unsqueeze(1))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_step()

        for _ in range(max_new_tokens - 1):
            graph.replay()
            tok_buf.copy_(out_buf.unsqueeze(1))
            pos_buf += 1
            gen_tokens.append(out_buf.clone())
            if out_buf.item() == eos_token_id:  # host sync, but enables early stop (emotion JSON ~76 tok)
                break

        return torch.cat([input_ids, torch.stack(gen_tokens, dim=1)], dim=1)


def load_qwen3(model_dir, device="cuda", dtype=torch.float16):
    """HF-format dir (config.json + model.safetensors) → Qwen3ForCausalLM, no transformers."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    model = Qwen3ForCausalLM(config)
    own = dict(model.named_parameters())
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    loaded = 0
    for k, v in state.items():
        target = k if k in own else k.removeprefix("model.")  # HF keys carry model. prefix
        if target in own:
            own[target].data.copy_(v.to(own[target].dtype))
            loaded += 1
    assert loaded == len(own), f"loaded {loaded}/{len(own)} params, missing: {[k for k in own if k not in state and ('model.'+k) not in state][:5]}"
    return model.to(device).to(dtype).eval()
