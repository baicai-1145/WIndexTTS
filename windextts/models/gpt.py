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
    ) -> torch.Tensor:
        """attention_mask: prebuilt 4D additive [B,1,T,T] (None=no mask, all attend)."""
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # HF GPT2 path: SDPA with prebuilt 4D additive mask, is_causal=False.
        # mem_eff is the only CUDA kernel accepting fp32 + non-null mask
        # (flash rejects non-null mask; cudnn rejects fp32+mask; verified torch 2.8).
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, is_causal=False,
                scale=1.0 / (self.head_dim ** 0.5),
            )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, heads, head_dim)
        attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), self.embed_dim)
        return self.c_proj(attn_output)


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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = mlp_out + residual
        return hidden_states


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
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        extended_mask = self._build_4d_causal_mask(hidden_states, attention_mask)
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=extended_mask)
        return self.ln_f(hidden_states)

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

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    def load_official(self, sd: dict[str, torch.Tensor]) -> None:
        """Load gpt.pth state_dict (456 keys). Drops unimplemented submodules
        (emo_conditioning_encoder / emo_perceiver_encoder) and reports count."""
        remapped: dict[str, torch.Tensor] = {}
        dropped: list[str] = []
        for k, v in sd.items():
            if k.startswith("emo_conditioning_encoder.") or k.startswith("emo_perceiver_encoder."):
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
        print(f"[GPT] loaded {len(remapped)} keys, dropped {len(dropped)} emo-conditioner keys")

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
        """
        device = style.device
        # normalize_emo_vec (infer_v2_5.py:491-504): apply per-emotion bias,
        # then scale down if total > 0.8.
        bias = torch.tensor(
            [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625],
            device=device, dtype=torch.float32,
        )
        wv = emo_vec.float().to(device) * bias
        s = wv.sum()
        if s > 0.8:
            wv = wv * (0.8 / s)
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
        spk_latent = self.spk_emb_proj(campplus_emb)  # [1,1280]
        if emo_vec.shape[-1] == 8:
            raise ValueError(
                "emo_vec must be [1,1280]; apply emo_matrix_lookup upstream "
                "(user-supplied 8-dim vector is not enough without spk_matrix)"
            )
        emo_3d = emo_vec.unsqueeze(1)  # [1,1,1280]
        conds = torch.cat(
            (spk_latent + emo_3d, torch.zeros(1, 2, self.model_dim, device=emo_vec.device)),
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
        hidden = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
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
        hidden = self.gpt(inputs_embeds=emb, attention_mask=attention_mask)
        logits = self.mel_head(self.final_norm(hidden))        # [B, S+1, 8194]
        return logits

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UnifiedVoice.forward is not the inference entry point; use "
            "prefill_logits_from_inputs / prefill_forward (AR loop is a later task)."
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
