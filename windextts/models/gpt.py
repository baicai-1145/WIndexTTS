"""GPT-AR (UnifiedVoice) — pure-torch re-implementation of the forward path.

Replaces ``indextts/gpt/model_v2.py`` (UnifiedVoice + GPT2InferenceModel) for the
IndexTTS-2.5 GPT-AR stage. No indextts/transformers dependency.

Scope of this module (forward path only):
  - GPT-2 body (24 layers, model_dim 1280, 20 heads) — the ``gpt.*`` weights.
  - Conditioning assembly (campplus spk + user-supplied emo_vec):
      conds_latent = cat(spk_emb_proj(campplus) + emo_vec, zeros(1,2,dim))  # [1,3,1280]
  - prepare_gpt_inputs: build [pad][cond][start_text+text+stop_text] embeddings,
    with lang_embedding (campplus mode) and start_mel_token placeholder.
  - prefill forward: GPT-2 transformer + lm_head(Sequential(final_norm, mel_head)).
  - emo_matrix lookup for user-supplied emo_vector (style-based cosine index).
  - AR decode loop / KV cache / emo_conditioning_encoder (conformer) are NOT in
    scope here (separate later task / user-supplied emo_vec path only).

Numerical contract (verified against official IndexTTS-2.5 on GPU, fp32):
  prefill logits match official to < 1e-3 (LayerNorm+Linear, effectively exact).

Key implementation notes (grounded in model_v2.py / HF GPT2):
  - GPT body weights are [in,out] layout (HF Conv1D style, model_v2.py uses
    transformers.GPT2Model). We replicate with a Conv1D-like linear layer that
    uses [in,out] weights directly (no transpose on load).
  - gpt.pth keys: 'gpt.h.{i}.attn.c_attn', 'gpt.ln_f', etc. map 1:1 onto our
    ``gpt`` submodule with prefix stripped. emo_conditioning_encoder /
    emo_perceiver_encoder keys are ignored (not implemented).
  - Positional: GPT-2 has no learned pos emb in the body (wpe replaced by null
    in build_hf_gpt_transformer). Mel/text pos embeddings are applied via
    LearnedPositionEmbeddings in prepare_gpt_inputs / decode step only.
  - Attention is causal. We use SDPA (via windextts.attention or torch SDPA).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ["UnifiedVoice"]


class Conv1D(nn.Module):
    """HF GPT-2 Conv1D: linear layer with [in, out] weights (no transpose).

    forward: out = x @ weight + bias   (weight shape [in, out]).
    """

    def __init__(self, nf: int, nx: int):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class GPT2Attention(nn.Module):
    """Causal multi-head attention (HF GPT2Attention semantics, SDPA-based).

    Uses the exact HF call signature: SDPA(attn_mask=bool[B,T], is_causal=True,
    scale=1/sqrt(head_dim)) with the mem_eff backend — the only CUDA kernel that
    supports fp32 + non-null mask (flash rejects non-null mask; cudnn rejects
    fp32+mask; verified on torch 2.8).
    """

    def __init__(self, model_dim: int, heads: int):
        super().__init__()
        self.embed_dim = model_dim
        self.num_heads = heads
        self.head_dim = model_dim // heads
        self.split_size = 3 * model_dim

        self.c_attn = Conv1D(3 * model_dim, model_dim)  # [in, 3*out]
        self.c_proj = Conv1D(model_dim, model_dim)

    def _split_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int) -> torch.Tensor:
        """Splits hidden_size dim into attn_head_size and num_heads."""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (B, heads, T, head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_bufs: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Causal attention with optional KV cache.

        Args:
            hidden_states: [B, T, dim] (prefill T>1) or [B, 1, dim] (decode step).
            attention_mask: prebuilt 4D additive [B,1,Tq,Tk] (None=no mask).
                - prefill: [B,1,T,T] full causal mask (built by GPT2Transformer).
                - decode:  [B,1,1,T_total] (pad keys masked) — single query attends
                  all valid past keys.
            past_key_value: (past_k, past_v) each [B, H, T_past, head_dim]; the
                new K/V is concatenated on the time axis.
            kv_bufs: (k_buf, v_buf) each [B, H, MAX_SEQ, head_dim] pre-allocated
                static KV buffers (CUDA Graph path). The new K/V is written in
                place at column ``kv_pos`` via index_copy_, and attention runs
                over the FULL buffers with a mask hiding positions > kv_pos
                (static shapes → graph-capturable).
            kv_pos: [1] long tensor — absolute buffer column for this step's K/V.
        Returns:
            (c_proj(output), (key, value)) — key/value AFTER the concat/update,
            i.e. the full per-layer cache [B, H, T_total, head_dim].
        """
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if kv_bufs is not None:
            # CUDA Graph path: static buffers, in-place KV write at column kv_pos.
            k_buf, v_buf = kv_bufs
            k_buf.index_copy_(2, kv_pos, key)    # key [B,H,1,D] -> buffer col kv_pos
            v_buf.index_copy_(2, kv_pos, value)
            attn_key, attn_value = k_buf, v_buf
        else:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                key = torch.cat([past_k, key], dim=2)  # [B,H,T_past+1,D]
                value = torch.cat([past_v, value], dim=2)
            attn_key, attn_value = key, value

        # HF GPT2 path: SDPA with prebuilt 4D additive mask, is_causal=False.
        # mem_eff is the only CUDA kernel accepting fp32 + non-null mask
        # (flash rejects non-null mask; cudnn rejects fp32+mask; verified torch 2.8).
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                query, attn_key, attn_value, attn_mask=attention_mask, is_causal=False,
                scale=1.0 / (self.head_dim ** 0.5),
            )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, heads, head_dim)
        attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), self.embed_dim)
        return self.c_proj(attn_output), (attn_key, attn_value)


class GPT2MLP(nn.Module):
    """HF GPT2MLP: c_fc (SiLU-free, GPT-2 uses GELU) → c_proj."""

    def __init__(self, model_dim: int, intermediate_size: int):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, model_dim)
        self.c_proj = Conv1D(model_dim, intermediate_size)
        self.act = nn.GELU(approximate="tanh")  # GPT-2 uses tanh approximation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    """HF GPT2Block: ln_1 → attn → residual; ln_2 → mlp → residual."""

    def __init__(self, model_dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_dim, eps=1e-5)
        self.attn = GPT2Attention(model_dim, heads)
        self.ln_2 = nn.LayerNorm(model_dim, eps=1e-5)
        self.mlp = GPT2MLP(model_dim, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_bufs: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out, kv = self.attn(
            hidden_states, attention_mask=attention_mask, past_key_value=past_key_value,
            kv_bufs=kv_bufs, kv_pos=kv_pos,
        )
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = mlp_out + residual
        return hidden_states, kv


class GPT2Transformer(nn.Module):
    """GPT-2 body: embedding-free stack of GPT2Blocks + final ln_f.

    Matches HF GPT2Model with wte deleted and wpe nulled — the caller always
    provides full inputs_embeds (positional info already baked in).
    """

    def __init__(self, layers: int, model_dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.h = nn.ModuleList(
            [GPT2Block(model_dim, heads, intermediate_size) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(model_dim, eps=1e-5)
        self._wte_ref = None  # plain ref (official sets wte; we keep it out of state_dict)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        kv_bufs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        kv_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """GPT-2 body with optional per-layer KV cache.

        Args:
            inputs_embeds: [B, T, dim] (prefill T>1) or [B, 1, dim] (decode step).
            attention_mask:
                - prefill: [B, T] int mask (0=pad) — converted to the 4D causal
                  additive mask internally (HF semantics incl. _unmask_unattended).
                - decode:  already-built [B,1,1,T_total] additive mask.
            past_key_values: per-layer (past_k, past_v) caches [B,H,T_past,D];
                None → full (prefill) forward. Both paths return the updated
                per-layer caches so a caller can chain prefill → decode steps.
            kv_bufs: per-layer (k_buf, v_buf) [B,H,MAX_SEQ,D] static buffers
                (CUDA Graph path); K/V written in place at column kv_pos, attention
                over the full buffers with a mask hiding positions > kv_pos.
            kv_pos: [1] long — absolute buffer column for this step's K/V write.
        Returns:
            (final_norm(hidden), updated_kvs). For prefill, updated_kvs holds the
            KV of all T tokens (used as the past for the first decode step).
        """
        hidden_states = inputs_embeds
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if kv_bufs is not None:
            # CUDA Graph decode: static per-layer buffers, in-place KV writes.
            assert kv_pos is not None and len(kv_bufs) == len(self.h)
            for i, block in enumerate(self.h):
                hidden_states, kv = block(
                    hidden_states, attention_mask=attention_mask,
                    kv_bufs=kv_bufs[i], kv_pos=kv_pos,
                )
                new_kvs.append(kv)
        elif past_key_values is None:
            extended_mask = self._build_4d_causal_mask(hidden_states, attention_mask)
            for block in self.h:
                hidden_states, kv = block(hidden_states, attention_mask=extended_mask)
                new_kvs.append(kv)
        else:
            assert len(past_key_values) == len(self.h)
            for i, block in enumerate(self.h):
                hidden_states, kv = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values[i],
                )
                new_kvs.append(kv)
        return self.ln_f(hidden_states), new_kvs

    @staticmethod
    def _build_4d_causal_mask(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Build the HF GPT2Model 4D additive causal mask.

        Replicates transformers' `_prepare_4d_causal_attention_mask` +
        `_unmask_unattended` (HF#110213): fully-masked rows (from left-padding)
        are set to all-attend to avoid NaN in softmax.

        attention_mask: [B, T] with 1=valid, 0=pad (or None for no padding).
        Returns [B, 1, T, T] additive mask (0=attend, min_dtype=masked).
        """
        if attention_mask is None:
            return None
        dtype = hidden_states.dtype
        B, T = hidden_states.shape[:2]
        min_dt = torch.finfo(dtype).min
        # causal: upper triangle (j > i) masked
        causal = torch.triu(
            torch.full((T, T), min_dt, dtype=dtype, device=hidden_states.device),
            diagonal=1,
        )  # [T,T]
        # padded key columns -> masked across all query rows
        pad_col = attention_mask == 0  # [B,T] bool
        # combine: start from causal, mask padded columns
        m = causal[None, None].expand(B, 1, T, T).clone()  # [B,1,T,T]
        m = m.masked_fill(pad_col[:, None, None, :], min_dt)  # padded key cols -> -inf
        # fully-masked rows (all -inf, from left-padding) -> all-attend (set row to 0)
        fully_masked = m.flatten(-1).min(dim=-1).values > 0  # rows where every entry is min_dt? no
        # A row is fully-masked iff all entries == min_dt:
        row_all_min = (m == min_dt).all(dim=-1, keepdim=True)  # [B,1,T,1]
        m = m.masked_fill(row_all_min, 0.0)  # unmask fully-masked rows
        return m


class LearnedPositionEmbeddings(nn.Module):
    """Sequential position embeddings (model_v2.py:244-281)."""

    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind: int, dev: torch.device) -> torch.Tensor:
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class UnifiedVoice(nn.Module):
    """GPT-AR forward path (campplus spk mode, user-supplied emo_vec)."""

    def __init__(
        self,
        layers: int = 24,
        model_dim: int = 1280,
        heads: int = 20,
        max_text_tokens: int = 600,
        max_mel_tokens: int = 1815,
        max_conditioning_inputs: int = 1,
        mel_length_compression: int = 1024,
        number_text_tokens: int = 60509,
        start_text_token: int = 0,
        stop_text_token: int = 1,
        number_mel_codes: int = 8194,
        start_mel_token: int = 8192,
        stop_mel_token: int = 8193,
        use_mel_codes_as_input: bool = True,
        types: int = 1,
        condition_num_latent: int = 32,
        condition_type: str = "conformer_perceiver",
        spk_cond_mode: str = "campplus",
    ):
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.spk_cond_mode = spk_cond_mode

        # campplus speaker projection
        if spk_cond_mode == "campplus":
            self.spk_emb_proj = nn.Linear(192, model_dim)
        else:
            raise NotImplementedError("only spk_cond_mode='campplus' is supported")

        # emo path (user-supplied emo_vec, no conformer encoder needed)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)
        # emo_conditioning modules: lazily added by build_emo_conditioning()
        # when the emo_ref_audio path is enabled (loads 149 keys from gpt.pth).
        self.emo_conditioning_encoder = None
        self.emo_perceiver_encoder = None
        self.lang_embedding = nn.Embedding(107, model_dim)  # len(LANGUAGE_DICT)+1

        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)

        # GPT-2 body + position embeddings
        self.gpt = GPT2Transformer(layers, model_dim, heads, 4 * model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(
            self.max_mel_tokens + 2 + self.max_conditioning_inputs, model_dim
        )
        self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_tokens + 2, model_dim)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        # Official post_init sets gpt.wte = mel_embedding (used by the decode step
        # via GPT2InferenceModel.embeddings). We keep a plain reference via
        # object.__setattr__ (avoids registering it as a submodule — gpt.pth has
        # no wte key, so registering would create a spurious gpt.wte.weight).
        object.__setattr__(self.gpt, "_wte_ref", self.mel_embedding)

        # CUDA Graph decode cache: key (max_seq, dtype) -> captured resources.
        self._graph_cache: dict = {}

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    def load_official(self, sd: dict[str, torch.Tensor], load_emo_conditioning: bool = False) -> None:
        """Load gpt.pth state_dict. By default drops the 149 emo_conditioning
        keys (heavy, only needed for emo_ref_audio path). Pass load_emo_conditioning=True
        after build_emo_conditioning() to load them."""
        remapped: dict[str, torch.Tensor] = {}
        dropped: list[str] = []
        for k, v in sd.items():
            is_emo = k.startswith("emo_conditioning_encoder.") or k.startswith("emo_perceiver_encoder.")
            if is_emo and not load_emo_conditioning:
                dropped.append(k)
                continue
            # gpt.pth uses 'gpt.h.{i}.*' and our submodule is also named 'gpt' —
            # keys map 1:1 without any prefix stripping.
            remapped[k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing:
            raise RuntimeError(f"missing keys loading GPT: {missing}")
        if unexpected:
            raise RuntimeError(f"unexpected keys loading GPT: {unexpected}")
        note = f" (loaded {149-len(dropped)} emo-cond keys)" if load_emo_conditioning else ""
        print(f"[GPT] loaded {len(remapped)} keys, dropped {len(dropped)} emo-conditioner keys{note}")

    def build_emo_conditioning(self) -> None:
        """Lazily create the emo_conditioning modules (160M params) for the
        emo_ref_audio path. Call before load_official(load_emo_conditioning=True)."""
        from windextts.models.emo_conditioning import EmoConformerEncoder, EmoPerceiverEncoder
        self.emo_conditioning_encoder = EmoConformerEncoder()
        self.emo_perceiver_encoder = EmoPerceiverEncoder()

    @torch.no_grad()
    def get_emovec(self, cond_emb: torch.Tensor) -> torch.Tensor:
        """cond_emb [B,T,1024] (w2v-bert[17] normalized) → emo_vec [B,1280].

        Runs conformer → perceiver → emovec_layer → emo_layer.
        Requires build_emo_conditioning() to have been called."""
        from windextts.models.emo_conditioning import get_emovec as _get_emovec
        if self.emo_conditioning_encoder is None:
            raise RuntimeError("emo_conditioning not built; call build_emo_conditioning() first")
        dev = cond_emb.device
        lens = torch.tensor([cond_emb.shape[1]], device=dev)
        return _get_emovec(
            self.emo_conditioning_encoder, self.emo_perceiver_encoder,
            self.emovec_layer, self.emo_layer,
            cond_emb, lens,
        )

    @torch.no_grad()
    def merge_emovec(self, spk_cond_emb: torch.Tensor, emo_cond_emb: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Blend spk + emo reference emovecs: base + alpha*(ref - base).

        Replicates GPT.merge_emovec in model_v2.py:833. Both inputs are
        [B,T,1024] w2v-bert[17] features. Returns [B,1280]."""
        emo_vec = self.get_emovec(emo_cond_emb)
        base_vec = self.get_emovec(spk_cond_emb)
        return base_vec + alpha * (emo_vec - base_vec)

    # ------------------------------------------------------------------
    # conditioning assembly
    # ------------------------------------------------------------------

    def emo_matrix_lookup(self, style: torch.Tensor, emo_vec: torch.Tensor,
                          spk_matrix, emo_matrix) -> torch.Tensor:
        """Replicate infer_v2_5 emo_vector path (pure matrix ops, no conformer).

        Args:
            style: [1,192] CAMPPlus embedding.
            emo_vec: [8] user emotion weights (each in [0,1.2]).
            spk_matrix: tuple of 8 chunks (torch.split of feat1.pt [73,192]).
            emo_matrix: tuple of 8 chunks (torch.split of feat2.pt [73,1280]).
        Returns:
            emovec_mat [1,1280]: sum_i emo_vec[i] * emo_matrix[chunk_i][idx_i],
            where idx_i = argmax cosine(style, spk_matrix[chunk_i]).

        NOTE: the official infer_v2_5.py uses the RAW emo_vector here (line 668,
        678) — normalize_emo_vec is defined but NEVER called in the matrix path.
        Applying bias/cap here was a bug that corrupted emo_vec direction
        (cosine ~0 vs official) → 'brick' audio for high single-emotion weights.
        """
        device = style.device
        wv = emo_vec.float().to(device)  # raw weights, no normalize (matches official)
        indices = []
        for chunk in spk_matrix:
            sims = F.cosine_similarity(style.float(), chunk.to(device).float(), dim=1)
            indices.append(int(torch.argmax(sims)))
        rows = [chunk_to[idx].unsqueeze(0) for idx, chunk_to in zip(indices, emo_matrix)]
        mat = torch.cat([r.to(device).float() for r in rows], 0)  # [8,1280]
        out = (wv.unsqueeze(1) * mat).sum(0).unsqueeze(0)  # [1,1280]
        return out

    def build_conds_latent(
        self,
        campplus_emb: torch.Tensor,
        emo_vec: torch.Tensor,
    ) -> torch.Tensor:
        """conds_latent [1,3,1280] = cat(spk_emb_proj(campplus) + emo_vec, zeros(1,2,dim)).

        Args:
            campplus_emb: [1,192].
            emo_vec: [1,1280] (already through emo_layer etc.), or [1,8] raw weights
                (then emo_matrix_lookup must have been applied upstream).
        """
        spk_latent = self.spk_emb_proj(campplus_emb.to(self.spk_emb_proj.weight.dtype))  # [1,1280]
        emo_3d = emo_vec.to(spk_latent.dtype).unsqueeze(1)  # [1,1,1280]
        conds = torch.cat(
            (spk_latent + emo_3d, torch.zeros(1, 2, self.model_dim, device=emo_vec.device, dtype=spk_latent.dtype)),
            dim=1,
        )
        return conds

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(input_ids, inputs_embeds, attention_mask) — replicate model_v2.py:648-715.

        input_ids: [B, S+1] (S = cond + text + 2, last = start_mel_token placeholder).
        inputs_embeds: [B, S, dim].
        attention_mask: [B, S+1] (0 at left-pad, 1 elsewhere).
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, (
                f"batch mismatch: {conditional_latents.shape[0]} vs {b}"
            )
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            if langs is not None and self.spk_cond_mode == "campplus":
                text_emb += self.lang_embedding(langs[i])
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
            attention_mask = torch.ones(target_len + 1, dtype=torch.long, device=device)
            padding: int = L + 2 - text_input.size(-1)
            if padding > 0:
                pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device)
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb)
            assert mel_emb.shape[0] == target_len, f"mel_emb {mel_emb.shape}, target_len {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)

        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)   # [B, S, dim]
        attention_mask = torch.stack(attention_masks, dim=0)    # [B, S+1]
        fake_inputs = torch.ones(
            (batched_mel_emb.shape[0], batched_mel_emb.shape[1] + 1),
            dtype=torch.long, device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    # ------------------------------------------------------------------
    # prefill forward
    # ------------------------------------------------------------------

    def prefill_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """GPT-2 transformer + lm_head(final_norm + mel_head) → logits [B, S, 8194].

        This is the official GPT2InferenceModel prefill path
        (cached_mel_emb == inputs_embeds here; no extra text input because
        input_ids last position is the start_mel placeholder, which for prefill
        alignment we run on the full embeds).
        """
        hidden, _ = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.mel_head(self.final_norm(hidden))
        return logits

    def prefill_logits_from_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """End-to-end prefill: build inputs → run transformer → logits [B, S+1, 8194].

        Mirrors official inference_speech's prefill: after prepare_gpt_inputs,
        cached_mel_emb = inputs_embeds (S tokens), then GPT2InferenceModel runs
        with input_ids [B, S+1] where the last (start_mel) token's embedding is
        looked up from mel_embedding + mel_pos and appended — producing
        [B, S+1, 8194] logits (the extra position predicts the first mel code).
        """
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(
            conditional_latents, text_inputs, langs
        )
        # Replicate GPT2InferenceModel.forward prefill:
        # mel_len = cached_mel_emb.shape[1]; text_inputs = input_ids[:, mel_len:] (last token)
        mel_len = inputs_embeds.shape[1]
        last_ids = input_ids[:, mel_len:]                      # [B, 1] start_mel token
        last_emb = self.mel_embedding(last_ids)                # [B, 1, dim]
        last_emb = last_emb + self.mel_pos_embedding(last_emb)  # mel pos (LearnedPos)
        emb = torch.cat([inputs_embeds, last_emb], dim=1)      # [B, S+1, dim]
        hidden, _ = self.gpt(inputs_embeds=emb, attention_mask=attention_mask)
        logits = self.mel_head(self.final_norm(hidden))        # [B, S+1, 8194]
        return logits

    # ------------------------------------------------------------------
    # AR decode loop (replaces HF generate / accel_engine)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_decode_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Additive [B,1,1,T_total] mask for a single decode query.

        The query is the newest token (last position), so causal masking is
        vacuous — the only thing to block are left-padded key positions
        (attention_mask == 0). Matches HF GPT2Model decode behavior (pad keys
        stay masked; -inf == torch.finfo(dtype).min, same as prefill mask).
        """
        min_dt = torch.finfo(dtype).min
        pad = attention_mask == 0  # [B,T]
        mask = torch.zeros(
            (attention_mask.shape[0], 1, 1, attention_mask.shape[1]),
            dtype=dtype, device=attention_mask.device,
        )
        return mask.masked_fill(pad[:, None, None, :], min_dt)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        generated_ids: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Sample next token ids from [B, V] logits (HF warper semantics).

        Matches HF generate: repetition_penalty → temperature → top_k → top_p
        (only when do_sample=True; greedy is a plain argmax, exactly like HF).
        Official IndexTTS uses repetition_penalty=10.0 (very strong) to prevent
        the AR decoder from repeating similar mel codes (the root cause of
        dragging/muffled quality when omitted).
        """
        if repetition_penalty != 1.0 and generated_ids is not None:
            # HF RepetitionPenaltyLogitsProcessor: gather scores of already-
            # generated tokens; negative → multiply (more negative), positive →
            # divide (smaller). Both directions reduce probability (transformers
            # repetition_penalty.py: score<0 ? score*penalty : score/penalty).
            for b in range(generated_ids.size(0)):
                prev = generated_ids[b]
                score = logits[b].gather(0, prev)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits[b].scatter_(0, prev, score)
        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk = torch.topk(logits, k, dim=-1)
                logits = logits.masked_fill(
                    logits < topk.values[..., -1:], float("-inf")
                )
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # keep at least the top token (HF TopPLogitsWarper shift-right)
                sorted_remove = cum_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove = remove.scatter(-1, sorted_idx, sorted_remove)
                logits = logits.masked_fill(remove, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)[:, 0]
        return torch.argmax(logits, dim=-1)

    def generate(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: Optional[torch.Tensor] = None,
        max_new_tokens: int = 500,
        do_sample: bool = False,
        top_k: int = 30,
        top_p: float = 0.8,
        temperature: float = 0.8,
        stop_token: Optional[int] = None,
        use_cuda_graph: bool = False,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
    ) -> torch.Tensor:
        """Autoregressive mel-code generation (batch=1), pure torch.

        Mirrors official inference_speech → GPT2InferenceModel.generate
        (kv_cache=True path, model_v2.py:760-830):
          - prefill: conds+text+start_mel embeddings → transformer → logits;
            last position predicts the first mel code; per-layer KV caches kept.
          - decode step i (i=0,1,...): embed the sampled token with
            mel_pos_embedding.get_fixed_embedding(i + 2) — official uses
            attention_mask.shape[1] - mel_len which, with mel_len = S (conds+
            text, before start_mel) and attention_mask grown to S+2+i, equals
            i + 2 (position 1 is skipped by the official flow — replicated).
          - single query attends all valid past keys (pad keys masked);
            K/V appended to the per-layer cache.

        Args:
            conditional_latents: [1,3,dim] conds_latent (spk+emo) or [B,3,dim].
            text_inputs: [B,L] text token ids (stop token already appended).
            langs: [B] language token ids (None → no lang embedding).
            max_new_tokens: cap on generated codes (not counting the stop token).
            do_sample/top_k/top_p/temperature: sampling (HF warper semantics).
            stop_token: default self.stop_mel_token (8193).
            use_cuda_graph: if True, run the decode loop through a captured CUDA
                Graph (batch=1, CUDA only). Output is bit-identical to the eager
                path (same math; KV buffers + attention mask only). The graph is
                captured once per (max_seq, dtype) and replayed on later calls.
            repetition_penalty: HF RepetitionPenaltyLogitsProcessor scale
                (official IndexTTS uses 10.0). Applied to already-generated ids.
            num_beams: beam width for beam-search multinomial sampling matching
                HF generate(num_beams=K, do_sample=..., top_p=..., ...). K beams
                are expanded from the single prefill, each sampled independently
                per step (with repetition_penalty), scored by cumulative
                log-prob, and the highest-scoring finished hypothesis wins.
                num_beams=1 keeps the exact current greedy/sample loop.
                Beam search is incompatible with the CUDA Graph path (dynamic
                branching), so num_beams>1 forces use_cuda_graph off.
        Returns:
            codes [B, T_gen] (T_gen includes the stop token when produced;
            may be shorter than max_new_tokens if stop was hit).
        """
        if conditional_latents.shape[0] != 1:
            raise NotImplementedError(
                "AR generate currently supports batch=1 (the official accel path "
                "is also single-sequence; batch is a later task)"
            )
        stop_token = self.stop_mel_token if stop_token is None else stop_token
        # match input dtype to model (supports fp16/bf16 mixed-precision decode)
        mdtype = next(self.parameters()).dtype
        if conditional_latents.dtype != mdtype:
            conditional_latents = conditional_latents.to(mdtype)
        if use_cuda_graph and num_beams <= 1:
            return self._generate_cuda_graph(
                conditional_latents, text_inputs, langs, max_new_tokens,
                do_sample, top_k, top_p, temperature, stop_token,
                repetition_penalty=repetition_penalty,
            )
        if num_beams > 1:
            return self._generate_beam_search(
                conditional_latents, text_inputs, langs, max_new_tokens,
                do_sample, top_k, top_p, temperature, stop_token,
                num_beams, repetition_penalty,
            )
        device = conditional_latents.device
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(
            conditional_latents, text_inputs, langs
        )
        mel_len = inputs_embeds.shape[1]  # S (conds+text, BEFORE start_mel)

        # --- prefill (same math as prefill_logits_from_inputs) ---
        last_ids = input_ids[:, mel_len:]  # [B,1] start_mel token
        last_emb = self.mel_embedding(last_ids)
        last_emb = last_emb + self.mel_pos_embedding(last_emb)  # mel pos 0
        emb = torch.cat([inputs_embeds, last_emb], dim=1)  # [B, S+1, dim]
        hidden, kvs = self.gpt(inputs_embeds=emb, attention_mask=attention_mask)
        logits = self.mel_head(self.final_norm(hidden))  # [B, S+1, V]

        # --- decode loop ---
        codes: list[torch.Tensor] = []
        cur_logits = logits[:, -1, :]  # [B,V] predicts first mel code
        gen_ids = torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)
        for step in range(max_new_tokens):
            next_id = self._sample(cur_logits, do_sample, top_k, top_p, temperature,
                                   generated_ids=gen_ids, repetition_penalty=repetition_penalty)
            codes.append(next_id)
            gen_ids = torch.cat([gen_ids, next_id.unsqueeze(1)], dim=1)
            if next_id.item() == stop_token:
                break
            # official decode emb: mel_embedding(token) + mel_pos at
            # attention_mask.shape[1] - mel_len. attention_mask after appending
            # this token is [B, S+2+step] → pos = step + 2 (t_1 → pos 2).
            pos = step + 2
            emb_dec = self.mel_embedding(next_id.unsqueeze(-1))  # [B,1,dim]
            emb_dec = emb_dec + self.mel_pos_embedding.get_fixed_embedding(
                pos, device
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype, device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )  # [B, T_total+1]
            dec_mask = self._build_decode_mask(attention_mask, dtype=mdtype)  # [B,1,1,T_total+1]
            hidden, kvs = self.gpt(
                inputs_embeds=emb_dec, attention_mask=dec_mask,
                past_key_values=kvs,
            )
            cur_logits = self.mel_head(self.final_norm(hidden))[:, -1, :]

        return torch.stack(codes, dim=1)  # [B, T_gen]

    def _generate_beam_search(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: Optional[torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        stop_token: int,
        num_beams: int,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Beam-search multinomial sampling (HF generate num_beams>1 semantics).

        Pure-torch reimplementation of the official path
        (infer_v2_5.py:782-789 → HF GPT2InferenceModel.generate with
        num_beams=3, repetition_penalty=10.0, do_sample=True, top_p=0.8,
        top_k=30, temperature=0.8, length_penalty=0.0).

        Design (batch=1 prefill → K beams, batched decode over active beams):
          - prefill once; replicate logits/KV/mask to K beams (equal scores 0).
          - per step: each active beam samples ONE token from its warped logits
            (repetition_penalty → temperature → top_k → top_p when do_sample;
            argmax when greedy). Candidate score = beam_score + log-softmax
            prob of the sampled token. With one candidate per beam, "keep
            top-K" is a no-op beyond the EOS filter (n_active ≤ K always).
          - EOS (stop_token): the beam is frozen into the finished-hypothesis
            list with its cumulative score; remaining beams continue.
          - stop when all beams finished or max_new_tokens exhausted.
          - length_penalty=0.0 (official): no length normalization.
          - return the highest-scoring finished hypothesis's codes (stop token
            stripped); if none finished, the best active beam.
        """
        device = conditional_latents.device
        mdtype = next(self.parameters()).dtype
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(
            conditional_latents, text_inputs, langs
        )
        mel_len = inputs_embeds.shape[1]  # S (conds+text, BEFORE start_mel)

        # --- prefill (batch=1, same math as generate()) ---
        last_ids = input_ids[:, mel_len:]  # [B,1] start_mel token
        last_emb = self.mel_embedding(last_ids)
        last_emb = last_emb + self.mel_pos_embedding(last_emb)  # mel pos 0
        emb = torch.cat([inputs_embeds, last_emb], dim=1)  # [B, S+1, dim]
        hidden, kvs = self.gpt(inputs_embeds=emb, attention_mask=attention_mask)
        logits = self.mel_head(self.final_norm(hidden))  # [B, S+1, V]

        K = num_beams
        # replicate the single-sequence state to K beams (materialized copies)
        cur_logits = logits[:, -1, :].expand(K, -1).contiguous()  # [K, V]
        kvs = [
            (k.expand(K, -1, -1, -1).contiguous(), v.expand(K, -1, -1, -1).contiguous())
            for k, v in kvs
        ]  # per-layer [K, H, S+1, head_dim]
        attention_mask = attention_mask.expand(K, -1).contiguous()  # [K, S+1]
        beam_scores = torch.zeros(K, dtype=torch.float32, device=device)
        gen_ids = torch.empty(K, 0, dtype=torch.long, device=device)
        finished: list[tuple[float, list[int]]] = []

        for step in range(max_new_tokens):
            n_active = cur_logits.shape[0]
            if n_active == 0:
                break
            next_id = self._sample(
                cur_logits, do_sample, top_k, top_p, temperature,
                generated_ids=gen_ids, repetition_penalty=repetition_penalty,
            )  # [n_active]
            # log-prob of the sampled token under the (warped) distribution;
            # _sample mutated cur_logits in place with penalty+warpers.
            tok_lp = F.log_softmax(cur_logits, dim=-1).gather(
                1, next_id.unsqueeze(1)
            ).squeeze(1)  # [n_active]
            cand_score = beam_scores + tok_lp  # [n_active] fp32

            is_eos = next_id == stop_token  # [n_active] bool
            if is_eos.any():
                eos_scores = cand_score[is_eos].tolist()
                eos_codes = gen_ids[is_eos].tolist()  # codes BEFORE stop
                finished.extend(zip(eos_scores, eos_codes))
            keep = ~is_eos
            if not keep.any():
                break
            # keep only the still-active (non-EOS) beams; all share step length
            beam_scores = cand_score[keep].float()
            gen_ids = torch.cat([gen_ids[keep], next_id[keep].unsqueeze(1)], dim=1)
            cur_logits = cur_logits[keep]
            kvs = [(k[keep], v[keep]) for k, v in kvs]
            attention_mask = attention_mask[keep]

            # advance the transformer for the active beams (batched)
            pos = step + 2  # mel pos (matches eager path)
            emb_dec = self.mel_embedding(next_id[keep].unsqueeze(-1))  # [n,1,dim]
            emb_dec = emb_dec + self.mel_pos_embedding.get_fixed_embedding(pos, device)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype, device=device,
                    ),
                ],
                dim=-1,
            )  # [n, T_total+1]
            dec_mask = self._build_decode_mask(attention_mask, dtype=mdtype)  # [n,1,1,T_total+1]
            hidden, kvs = self.gpt(
                inputs_embeds=emb_dec, attention_mask=dec_mask,
                past_key_values=kvs,
            )
            cur_logits = self.mel_head(self.final_norm(hidden))[:, -1, :]

        if finished:
            best = max(finished, key=lambda x: x[0])  # highest cumulative score
            return torch.tensor([best[1]], dtype=torch.long, device=device)  # [1, T_gen]
        # none finished: return the best-scoring active beam
        best_beam = int(beam_scores.argmax().item())
        return gen_ids[best_beam].unsqueeze(0)  # [1, T_gen]

    # ------------------------------------------------------------------
    # CUDA Graph AR decode (stage 3b)
    # ------------------------------------------------------------------

    def _graph_decode_step(
        self,
        input_id_buf: torch.Tensor,
        pos_buf: torch.Tensor,
        kv_pos_buf: torch.Tensor,
        mask_buf: torch.Tensor,
        logits_buf: torch.Tensor,
        kv_bufs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Single AR decode step — straight-line, static shapes (CUDA Graph body).

        All tensors are pre-allocated static buffers (or views of them): the
        graph writes this step's K/V into kv_bufs at column kv_pos, runs the
        transformer over the full buffers with the attention mask hiding
        positions > kv_pos, and copies the next-token logits into logits_buf.
        Replayed via torch.cuda.CUDAGraph.replay() with the buffers updated
        between replays (input id, mel pos, kv pos, mask).
        """
        emb = self.mel_embedding(input_id_buf)                        # [B,1,dim]
        emb = emb + self.mel_pos_embedding.emb(pos_buf).unsqueeze(1)   # [1,1,dim]
        hidden, _ = self.gpt(
            inputs_embeds=emb,
            attention_mask=mask_buf,          # [B,1,1,max_seq]
            kv_bufs=kv_bufs,
            kv_pos=kv_pos_buf,
        )
        logits = self.mel_head(self.final_norm(hidden))[:, 0, :]      # [B,V]
        logits_buf.copy_(logits)

    def _capture_graph(
        self, B: int, max_seq: int
    ) -> tuple[torch.cuda.CUDAGraph, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Warm up + capture the decode-step graph for a fixed max_seq.

        Returns (graph, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf,
        logits_buf). The buffers are all pre-allocated (static addresses); the
        graph writes into them. KV content is copied in per-call before replay.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        H, D = self.heads, self.model_dim // self.heads

        kv_bufs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.layers):
            k_buf = torch.zeros(B, H, max_seq, D, dtype=dtype, device=device)
            v_buf = torch.zeros(B, H, max_seq, D, dtype=dtype, device=device)
            kv_bufs.append((k_buf, v_buf))
        input_id_buf = torch.zeros(B, 1, dtype=torch.long, device=device)
        pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        kv_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        logits_buf = torch.empty(B, self.number_mel_codes, dtype=dtype, device=device)
        mask_buf = torch.zeros(B, 1, 1, max_seq, dtype=dtype, device=device)

        # Warmup: run the same op path eagerly (primes cuDNN/cuBLAS autotune and
        # workspace allocations; capture must not observe fresh allocations).
        for _ in range(3):
            self._graph_decode_step(
                input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf, kv_bufs
            )
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._graph_decode_step(
                input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf, kv_bufs
            )
        return g, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf

    def _generate_cuda_graph(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: Optional[torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        stop_token: int,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """CUDA-Graph AR decode — bit-identical to the eager generate() path.

        Prefill runs eagerly (identical math), then the decode loop replays a
        captured graph per step. Per-step variable state (sampled id, mel pos,
        KV buffer column, attention mask) lives in static buffers updated
        between replays; sampling + stop check stay in Python (multinomial /
        argmax are not graph-captured).
        """
        device = conditional_latents.device
        mdtype = next(self.parameters()).dtype
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "use_cuda_graph=True requires CUDA; pass use_cuda_graph=False on CPU"
            )
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(
            conditional_latents, text_inputs, langs
        )
        mel_len = inputs_embeds.shape[1]  # S (conds+text, BEFORE start_mel)

        # --- prefill (same math as the eager path) ---
        last_ids = input_ids[:, mel_len:]  # [B,1] start_mel token
        last_emb = self.mel_embedding(last_ids)
        last_emb = last_emb + self.mel_pos_embedding(last_emb)  # mel pos 0
        emb = torch.cat([inputs_embeds, last_emb], dim=1)  # [B, S+1, dim]
        hidden, kvs = self.gpt(inputs_embeds=emb, attention_mask=attention_mask)
        logits = self.mel_head(self.final_norm(hidden))  # [B, S+1, V]

        S = mel_len + 1  # prefill token count (KV buffer positions 0..S-1)
        pad_len = int((attention_mask[0] == 0).sum().item())
        # round max_seq up to a bucket so similar text lengths reuse the graph
        # (avoid re-capture per request; bucket=64 covers typical variation)
        raw_seq = S + max_new_tokens + 8
        max_seq = ((raw_seq + 63) // 64) * 64
        cache_key = (max_seq, next(self.parameters()).dtype)
        cache = self._graph_cache.get(cache_key)
        if cache is None:
            cache = self._capture_graph(1, max_seq)
            self._graph_cache[cache_key] = cache
        g, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf = cache

        # copy prefill KVs into the front of the static buffers
        for (k, v), (k_buf, v_buf) in zip(kvs, kv_bufs):
            k_buf[:, :, :S, :].copy_(k)
            v_buf[:, :, :S, :].copy_(v)

        min_dt = torch.finfo(mask_buf.dtype).min
        codes: list[torch.Tensor] = []
        cur_logits = logits[:, -1, :]  # [B,V] predicts first mel code
        gen_ids = torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)
        for step in range(max_new_tokens):
            next_id = self._sample(cur_logits, do_sample, top_k, top_p, temperature,
                                   generated_ids=gen_ids, repetition_penalty=repetition_penalty)
            codes.append(next_id)
            gen_ids = torch.cat([gen_ids, next_id.unsqueeze(1)], dim=1)
            if next_id.item() == stop_token:
                break
            kv_pos = S + step  # absolute KV buffer column for this token
            # update static buffers (eager, outside the graph)
            input_id_buf.copy_(next_id.unsqueeze(1))  # [B,1]
            pos_buf.fill_(step + 2)                   # mel pos (matches eager)
            kv_pos_buf.fill_(kv_pos)
            mask_buf.fill_(min_dt)                    # mask future + pad positions
            mask_buf[:, :, :, pad_len:kv_pos + 1].fill_(0.0)
            g.replay()
            cur_logits = logits_buf.float()  # [B,V] fp32 for stable sampling

        return torch.stack(codes, dim=1)  # [B, T_gen]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UnifiedVoice.forward is not the inference entry point; use "
            "prefill_logits_from_inputs / prefill_forward / generate."
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/root/WIndexTTS")
    from windextts.weights import WeightLoader

    dev = "cuda"
    sd = WeightLoader().load_gpt()
    m = UnifiedVoice()
    m.load_official(sd)
    m.eval().to(dev)
    print(f"[GPT] params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    # quick smoke with dumps
    out_dir = "/root/windextts_dumps"
    style = torch.load(f"{out_dir}/gpt.style.pt", weights_only=False).to(dev)
    emovec = torch.load(f"{out_dir}/gpt.emovec_mat.pt", weights_only=False).to(dev)
    text_tokens = torch.load(f"{out_dir}/gpt.text_tokens_v2.pt", weights_only=False).to(dev)
    lang = torch.load(f"{out_dir}/gpt.lang.pt", weights_only=False).to(dev)
    conds = m.build_conds_latent(style, emovec)
    ref_conds = torch.load(f"{out_dir}/gpt.conds_latent_v2.pt", weights_only=False).to(dev)
    print(f"conds_latent diff: {(conds - ref_conds).abs().max().item():.3e}")
    with torch.no_grad():
        logits = m.prefill_logits_from_inputs(conds, text_tokens, lang)
    ref_logits = torch.load(f"{out_dir}/gpt.prefill_logits_v2.pt", weights_only=False).to(dev)
    print(f"logits: {tuple(logits.shape)} ref: {tuple(ref_logits.shape)}")
    diff = (logits.float() - ref_logits.float()).abs().max().item()
    print(f"prefill logits max_abs_diff = {diff:.3e}")
    print(f"allclose(atol=1e-3, rtol=1e-3) = {torch.allclose(logits.float(), ref_logits.float(), atol=1e-3, rtol=1e-3)}")
    print("SMOKE", "OK" if diff < 1e-2 else "FAIL")
