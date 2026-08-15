"""GPT-AR (UnifiedVoice) forward + AR decode, pure-torch. Replaces
indextts/gpt/model_v2.py (UnifiedVoice + GPT2InferenceModel) for IndexTTS-2.5.

Numerical contract (vs official, GPU fp32): prefill logits < 1e-3; greedy AR
codes exact match. GPT body weights are [in,out] (HF Conv1D) — transposed at
load (nn.Linear subclass keeps torchao int4 filter compatibility). Positional
info comes only from mel/text LearnedPositionEmbeddings (body wpe is null).
Attention: causal SDPA, mem_eff backend — the only CUDA kernel accepting
fp32 + non-null mask (flash rejects mask, cudnn rejects fp32+mask, torch 2.8).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ["UnifiedVoice"]


class Conv1D(nn.Linear):
    # HF Conv1D: weight [in,out], y = x@W. Subclassing nn.Linear (weights
    # transposed at load) keeps torchao int4_weight_only isinstance(m, Linear)
    # compatibility.
    def __init__(self, nf, nx):
        super().__init__(nx, nf)  # Conv1D(nf=out, nx=in)
        self.nf, self.nx = nf, nx


class GPT2Attention(nn.Module):
    def __init__(self, model_dim, heads):
        super().__init__()
        self.embed_dim, self.num_heads, self.head_dim = model_dim, heads, model_dim // heads
        self.c_attn = Conv1D(3 * model_dim, model_dim)  # [in, 3*out]
        self.c_proj = Conv1D(model_dim, model_dim)

    def _split_heads(self, x, h, d):
        return x.view(x.size()[:-1] + (h, d)).permute(0, 2, 1, 3)  # [B,h,T,d]

    # hidden [B,T,dim] (prefill) or [B,1,dim] (decode); mask: 4D additive
    # [B,1,Tq,Tk] or None. past_key_value/(k,v) concat on time. kv_bufs+
    # kv_pos: static CUDA-Graph buffers — K/V written at column kv_pos, attention
    # over the FULL buffers with mask hiding positions > kv_pos. Returns
    # (c_proj(out), (k, v)) with the post-update per-layer cache.
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, kv_bufs=None, kv_pos=None):
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
    def __init__(self, model_dim, intermediate_size):
        super().__init__()
        self.c_fc, self.c_proj = Conv1D(intermediate_size, model_dim), Conv1D(model_dim, intermediate_size)
        self.act = nn.GELU(approximate="tanh")  # GPT-2 uses the tanh approximation

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, model_dim, heads, intermediate_size):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(model_dim, eps=1e-5), GPT2Attention(model_dim, heads)
        self.ln_2, self.mlp = nn.LayerNorm(model_dim, eps=1e-5), GPT2MLP(model_dim, intermediate_size)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, kv_bufs=None, kv_pos=None):
        h, kv = self.attn(self.ln_1(hidden_states), attention_mask, past_key_value, kv_bufs, kv_pos)
        h = h + hidden_states
        return h + self.mlp(self.ln_2(h)), kv


class GPT2Transformer(nn.Module):
    # embedding-free block stack + ln_f (HF GPT2Model with wte deleted, wpe null
    # — caller provides full inputs_embeds with positional info baked in).
    def __init__(self, layers, model_dim, heads, intermediate_size):
        super().__init__()
        self.h = nn.ModuleList(GPT2Block(model_dim, heads, intermediate_size) for _ in range(layers))
        self.ln_f = nn.LayerNorm(model_dim, eps=1e-5)
        self._wte_ref = None  # plain ref (official sets wte; kept out of state_dict)

    # prefill: mask=[B,T] int (0=pad) — built into 4D causal internally; decode:
    # mask already [B,1,1,T_total] additive. past_key_values=None → full forward.
    # kv_bufs/kv_pos → CUDA-Graph static-buffer path. Returns (ln_f(h), kvs).
    def forward(self, inputs_embeds, attention_mask=None, past_key_values=None, kv_bufs=None, kv_pos=None):
        h, kvs = inputs_embeds, []
        if kv_bufs is not None:
            assert kv_pos is not None and len(kv_bufs) == len(self.h)
            for i, block in enumerate(self.h):
                h, kv = block(h, attention_mask, kv_bufs=kv_bufs[i], kv_pos=kv_pos)
                kvs.append(kv)
        elif past_key_values is None:
            m = self._build_4d_causal_mask(h, attention_mask)
            for block in self.h:
                h, kv = block(h, m)
                kvs.append(kv)
        else:
            assert len(past_key_values) == len(self.h)
            for i, block in enumerate(self.h):
                h, kv = block(h, attention_mask, past_key_value=past_key_values[i])
                kvs.append(kv)
        return self.ln_f(h), kvs

    @staticmethod
    def _build_4d_causal_mask(hidden_states, attention_mask):
        # HF _prepare_4d_causal_attention_mask + _unmask_unattended (HF#110213):
        # fully-masked rows (left-padding) are set all-attend to avoid softmax NaN.
        if attention_mask is None:
            return None
        dtype, (B, T) = hidden_states.dtype, hidden_states.shape[:2]
        min_dt = torch.finfo(dtype).min
        causal = torch.triu(torch.full((T, T), min_dt, dtype=dtype, device=hidden_states.device), 1)
        m = causal[None, None].expand(B, 1, T, T).clone().masked_fill((attention_mask == 0)[:, None, None, :], min_dt)
        return m.masked_fill((m == min_dt).all(-1, keepdim=True), 0.0)  # [B,1,T,T]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(0, init)

    def forward(self, x):
        return self.emb(torch.arange(x.shape[1], device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class UnifiedVoice(nn.Module):
    # campplus spk mode; user-supplied emo_vec (conformer added lazily).
    def __init__(self, layers=24, model_dim=1280, heads=20, max_text_tokens=600,
                 max_mel_tokens=1815, max_conditioning_inputs=1, number_text_tokens=60509,
                 number_mel_codes=8194):
        super().__init__()
        self.number_text_tokens, self.number_mel_codes = number_text_tokens, number_mel_codes
        self.start_text_token, self.stop_text_token = 0, 1
        self.start_mel_token, self.stop_mel_token = 8192, 8193
        self.layers, self.heads, self.model_dim = layers, heads, model_dim
        self.max_mel_tokens, self.max_text_tokens = max_mel_tokens, max_text_tokens
        self.max_conditioning_inputs = max_conditioning_inputs
        self.spk_cond_mode = "campplus"
        self.spk_emb_proj = nn.Linear(192, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)
        # emo_conditioning modules: lazily added by build_emo_conditioning()
        # (149 keys from gpt.pth; only needed for emo_ref_audio path).
        self.emo_conditioning_encoder = None
        self.emo_perceiver_encoder = None
        self.lang_embedding = nn.Embedding(107, model_dim)  # len(LANGUAGE_DICT)+1
        self.text_embedding = nn.Embedding(number_text_tokens + 1, model_dim)
        self.mel_embedding = nn.Embedding(number_mel_codes, model_dim)
        self.gpt = GPT2Transformer(layers, model_dim, heads, 4 * model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_tokens + 2 + max_conditioning_inputs, model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_tokens + 2, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, number_text_tokens + 1)
        self.mel_head = nn.Linear(model_dim, number_mel_codes)
        # plain ref (official sets gpt.wte = mel_embedding; kept out of state_dict
        # — gpt.pth has no wte key, registering would create a spurious weight)
        object.__setattr__(self.gpt, "_wte_ref", self.mel_embedding)
        self._graph_cache = {}      # (max_seq, dtype) -> captured decode graph
        self._beam_graph_cache = {} # (num_beams, max_seq, dtype) -> beam graph

    def load_official(self, sd, load_emo_conditioning=False):
        # gpt.pth 'gpt.h.{i}.*' maps 1:1 onto our 'gpt' submodule. Conv1D weights
        # in the ckpt are [in,out]; nn.Linear wants [out,in] — transposed here.
        # The 149 emo_conditioning keys are dropped unless the conformer was built.
        _C1D = (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")
        remapped, dropped = {}, []
        for k, v in sd.items():
            if (k.startswith("emo_conditioning_encoder.") or k.startswith("emo_perceiver_encoder.")) and not load_emo_conditioning:
                dropped.append(k)
                continue
            if any(k.endswith(s) for s in _C1D) and v.dim() == 2:
                v = v.t().contiguous()  # [in,out] -> [out,in]
            remapped[k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"loading GPT: missing={missing[:5]} unexpected={unexpected[:5]}")
        note = f" (loaded {149 - len(dropped)} emo-cond keys)" if load_emo_conditioning else ""
        print(f"[GPT] loaded {len(remapped)} keys, dropped {len(dropped)} emo-conditioner keys{note}")

    def build_emo_conditioning(self):
        # 160M params; call before load_official(load_emo_conditioning=True)
        from windextts.models.emo_conditioning import EmoConformerEncoder, EmoPerceiverEncoder
        self.emo_conditioning_encoder = EmoConformerEncoder()
        self.emo_perceiver_encoder = EmoPerceiverEncoder()

    @torch.no_grad()
    def get_emovec(self, cond_emb):
        # cond_emb [B,T,1024] (w2v-bert[17] normalized) → emo_vec [B,1280]
        # via conformer → perceiver → emovec_layer → emo_layer
        from windextts.models.emo_conditioning import get_emovec as _get_emovec
        if self.emo_conditioning_encoder is None:
            raise RuntimeError("call build_emo_conditioning() first")
        return _get_emovec(self.emo_conditioning_encoder, self.emo_perceiver_encoder,
                           self.emovec_layer, self.emo_layer,
                           cond_emb, torch.tensor([cond_emb.shape[1]], device=cond_emb.device))

    @torch.no_grad()
    def merge_emovec(self, spk_cond_emb, emo_cond_emb, alpha=1.0):
        # base + alpha*(ref - base); inputs are [B,T,1024] w2v-bert[17] feats
        emo_vec, base_vec = self.get_emovec(emo_cond_emb), self.get_emovec(spk_cond_emb)
        return base_vec + alpha * (emo_vec - base_vec)

    def emo_matrix_lookup(self, style, emo_vec, spk_matrix, emo_matrix):
        # infer_v2_5 emo_vector path: emovec = sum_i w_i * emo_matrix[i][argmax
        # cosine(style, spk_matrix[i])]. style [1,192]; emo_vec [8] weights;
        # matrices = torch.split of feat1.pt [73,192] / feat2.pt [73,1280].
        # PITFALL: official uses the RAW emo_vector here — normalize_emo_vec is
        # defined but NEVER called in this path; applying bias/cap corrupted the
        # emo_vec direction (cosine ~0) → 'brick' audio at high weights.
        wv = emo_vec.float().to(style.device)
        idx = [int(torch.argmax(F.cosine_similarity(style.float(), c.to(style.device).float(), dim=1)))
               for c in spk_matrix]
        mat = torch.cat([emo_matrix[i][j].unsqueeze(0).to(style.device).float()
                         for i, j in enumerate(idx)], 0)  # [8,1280]
        return (wv.unsqueeze(1) * mat).sum(0).unsqueeze(0)  # [1,1280]

    def build_conds_latent(self, campplus_emb, emo_vec):
        # cat(spk_emb_proj(campplus) + emo_vec, zeros(1,2,dim)) -> [1,3,1280]
        spk = self.spk_emb_proj(campplus_emb.to(self.spk_emb_proj.weight.dtype))
        return torch.cat((spk + emo_vec.to(spk.dtype).unsqueeze(1),
                          torch.zeros(1, 2, self.model_dim, device=emo_vec.device, dtype=spk.dtype)), 1)

    def prepare_gpt_inputs(self, conditional_latents, text_inputs, langs=None):
        # [pad][cond][start_text+text+stop_text] embeddings (model_v2.py:648-715).
        # Returns (input_ids [B,S+1] last=start_mel placeholder, embeds [B,S,dim],
        # attention_mask [B,S+1] with 0 at left-pad).
        b, L, device = *text_inputs.shape[:2], text_inputs.device
        single = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single:
            assert conditional_latents.shape[0] == b
        target_len = conditional_latents.shape[1] + L + 2
        embs, masks = [], []
        for i in range(b):
            t = text_inputs[i][(text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)]
            t = F.pad(F.pad(t, (1, 0), value=self.start_text_token), (0, 1), value=self.stop_text_token)
            te = self.text_embedding(t) + self.text_pos_embedding.emb(torch.arange(t.size(-1), device=device))
            if langs is not None:
                te = te + self.lang_embedding(langs[i])
            am = torch.ones(target_len + 1, dtype=torch.long, device=device)
            parts = [conditional_latents.squeeze(0) if single else conditional_latents[i], te]
            pad = L + 2 - t.size(-1)
            if pad > 0:  # left-pad [pad][cond][text]
                parts.insert(0, torch.zeros(pad, conditional_latents.size(-1), dtype=te.dtype, device=device))
                am[:pad] = 0
            embs.append(torch.cat(parts))
            masks.append(am)
        embs = torch.stack(embs)      # [B, S, dim]
        masks = torch.stack(masks)    # [B, S+1]
        ids = torch.ones(embs.shape[0], embs.shape[1] + 1, dtype=torch.long, device=device)
        ids[:, -1] = self.start_mel_token
        return ids, embs, masks

    def prefill_forward(self, inputs_embeds, attention_mask):
        hidden, _ = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self.mel_head(self.final_norm(hidden))

    def prefill_logits_from_inputs(self, conditional_latents, text_inputs, langs=None):
        # prepare_gpt_inputs -> append mel_embedding(start_mel) + mel_pos -> body
        # -> lm_head. [B, S+1, 8194]; the extra position predicts the first code.
        ids, embeds, mask = self.prepare_gpt_inputs(conditional_latents, text_inputs, langs)
        last = self.mel_embedding(ids[:, embeds.shape[1]:])    # [B,1,dim] start_mel
        last = last + self.mel_pos_embedding(last)             # mel pos (LearnedPos)
        hidden, _ = self.gpt(inputs_embeds=torch.cat([embeds, last], 1), attention_mask=mask)
        return self.mel_head(self.final_norm(hidden))

    # ------------------------------------------------------------------
    # AR decode loop (replaces HF generate / accel_engine)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_decode_mask(attention_mask, dtype=torch.float32):
        # single decode query: causal masking vacuous; only left-pad keys blocked
        b, t = attention_mask.shape
        return torch.zeros(b, 1, 1, t, dtype=dtype, device=attention_mask.device).masked_fill(
            (attention_mask == 0)[:, None, None, :], torch.finfo(dtype).min)

    @staticmethod
    def _sample(logits, do_sample, top_k, top_p, temperature, generated_ids=None, repetition_penalty=1.0):
        # HF generate warper order: repetition_penalty → temperature → top_k →
        # top_p (sampling only; greedy = plain argmax). Official uses rep=10.0 —
        # without it the AR decoder repeats mel codes (dragging/muffled quality).
        if repetition_penalty != 1.0 and generated_ids is not None:
            # HF semantics: score<0 ? score*penalty : score/penalty (both reduce prob)
            for b in range(generated_ids.size(0)):
                s = logits[b].gather(0, generated_ids[b])
                logits[b].scatter_(0, generated_ids[b],
                                   torch.where(s < 0, s * repetition_penalty, s / repetition_penalty))
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k and top_k > 0:
            logits = logits.masked_fill(logits < torch.topk(logits, min(top_k, logits.size(-1)), -1).values[..., -1:], float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sl, si = torch.sort(logits, descending=True, dim=-1)
            rm = torch.cumsum(F.softmax(sl, -1), -1) > top_p
            rm[..., 1:] = rm[..., :-1].clone()  # keep the top token (HF shift-right)
            rm[..., 0] = False
            logits = logits.masked_fill(torch.zeros_like(logits, dtype=torch.bool).scatter(-1, si, rm), float("-inf"))
        return torch.multinomial(F.softmax(logits, -1), 1)[:, 0]

    # AR mel-code generation (batch=1), pure torch; mirrors official
    # inference_speech → GPT2InferenceModel.generate (kv_cache=True path).
    # Decode step i embeds its token at mel pos i+2 (official computes
    # attention_mask.shape[1] - mel_len, which reduces to i+2; pos 1 skipped).
    # use_cuda_graph: decode through a captured graph — bit-identical output,
    # captured once per (max_seq,dtype) bucket. num_beams>1: K independent
    # samplers from one prefill, cumulative log-prob winner (official semantics;
    # graph variant keeps a FIXED K batch — EOS beams keep feeding stop).
    # Returns codes [B, T_gen] (stop token included when produced).
    # Shared prefill for all four decode paths: prepare inputs, append
    # start_mel emb + mel pos, run the body, return (S, attention_mask, kvs,
    # first-step logits [1,V]).
    def _prefill(self, conditional_latents, text_inputs, langs):
        ids, embeds, attention_mask = self.prepare_gpt_inputs(conditional_latents, text_inputs, langs)
        S = embeds.shape[1]  # conds+text, BEFORE start_mel
        last = self.mel_embedding(ids[:, S:])
        emb = torch.cat([embeds, last + self.mel_pos_embedding(last)], 1)  # [B,S+1,dim]
        hidden, kvs = self.gpt(inputs_embeds=emb, attention_mask=attention_mask)
        return S, attention_mask, kvs, self.mel_head(self.final_norm(hidden))[:, -1]

    def _eager_step(self, next_id, step, attention_mask, kvs, mdtype, K_sel=None):
        # eager decode of one token: mel_embedding + mel pos step+2 (header note)
        nid = next_id if K_sel is None else next_id[K_sel]
        e = self.mel_embedding(nid.unsqueeze(-1)) + self.mel_pos_embedding.get_fixed_embedding(step + 2, next_id.device)
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=next_id.device)], -1)
        hidden, kvs = self.gpt(inputs_embeds=e, attention_mask=self._build_decode_mask(attention_mask, mdtype), past_key_values=kvs)
        return attention_mask, kvs, self.mel_head(self.final_norm(hidden))[:, -1]

    def generate(self, conditional_latents, text_inputs, langs=None, max_new_tokens=500,
                 do_sample=False, top_k=30, top_p=0.8, temperature=0.8, stop_token=None,
                 use_cuda_graph=False, repetition_penalty=1.0, num_beams=1):
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
        a = (conditional_latents, text_inputs, langs, max_new_tokens,
             do_sample, top_k, top_p, temperature, stop_token)
        if use_cuda_graph:
            if num_beams > 1:
                return self._generate_beam_graph(*a, num_beams, repetition_penalty)
            return self._generate_cuda_graph(*a, repetition_penalty=repetition_penalty)
        if num_beams > 1:
            return self._generate_beam_search(*a, num_beams, repetition_penalty)
        device = conditional_latents.device
        S, attention_mask, kvs, cur_logits = self._prefill(conditional_latents, text_inputs, langs)

        codes, gen_ids = [], torch.empty(1, 0, dtype=torch.long, device=device)
        for step in range(max_new_tokens):
            next_id = self._sample(cur_logits, do_sample, top_k, top_p, temperature, gen_ids, repetition_penalty)
            codes.append(next_id)
            gen_ids = torch.cat([gen_ids, next_id.unsqueeze(1)], 1)
            if next_id.item() == stop_token:
                break
            attention_mask, kvs, cur_logits = self._eager_step(next_id, step, attention_mask, kvs, mdtype)
        return torch.stack(codes, 1)  # [B, T_gen]

    # HF generate num_beams>1 semantics (official: beams=3, rep=10.0, sampled).
    # One prefill → K beams; each active beam samples ONE token per step; score
    # = cumsum log-softmax of the sampled token; EOS freezes a beam into the
    # finished list; highest cumulative score wins (length_penalty=0.0).
    def _generate_beam_search(self, conditional_latents, text_inputs, langs, max_new_tokens,
                              do_sample, top_k, top_p, temperature, stop_token, num_beams,
                              repetition_penalty=1.0):
        device = conditional_latents.device
        mdtype = next(self.parameters()).dtype
        S, attention_mask, kvs, cur_logits = self._prefill(conditional_latents, text_inputs, langs)

        K = num_beams
        cur_logits = cur_logits.expand(K, -1).contiguous()  # [K,V]
        kvs = [(k.expand(K, -1, -1, -1).contiguous(), v.expand(K, -1, -1, -1).contiguous()) for k, v in kvs]
        attention_mask = attention_mask.expand(K, -1).contiguous()
        beam_scores = torch.zeros(K, dtype=torch.float32, device=device)
        gen_ids = torch.empty(K, 0, dtype=torch.long, device=device)
        finished = []

        for step in range(max_new_tokens):
            if cur_logits.shape[0] == 0:
                break
            next_id = self._sample(cur_logits, do_sample, top_k, top_p, temperature, gen_ids, repetition_penalty)
            # _sample mutated cur_logits in place (penalty+warpers) — score under that distribution
            tok_lp = F.log_softmax(cur_logits, -1).gather(1, next_id.unsqueeze(1)).squeeze(1)
            cand_score = beam_scores + tok_lp  # fp32

            is_eos = next_id == stop_token
            if is_eos.any():
                finished.extend(zip(cand_score[is_eos].tolist(), gen_ids[is_eos].tolist()))
            keep = ~is_eos
            if not keep.any():
                break
            beam_scores = cand_score[keep].float()
            gen_ids = torch.cat([gen_ids[keep], next_id[keep].unsqueeze(1)], 1)
            cur_logits = cur_logits[keep]
            kvs = [(k[keep], v[keep]) for k, v in kvs]
            attention_mask = attention_mask[keep]
            attention_mask, kvs, cur_logits = self._eager_step(next_id, step, attention_mask, kvs, mdtype, K_sel=keep)

        if finished:
            return torch.tensor([max(finished, key=lambda x: x[0])[1]], dtype=torch.long, device=device)
        return gen_ids[int(beam_scores.argmax())].unsqueeze(0)

    # Beam search with the decode step in a CUDA Graph: batch FIXED at K — no
    # KV reordering/removal; EOS beams freeze (score+codes recorded) and keep
    # feeding the stop token so graph shapes never change.
    def _generate_beam_graph(self, conditional_latents, text_inputs, langs, max_new_tokens,
                             do_sample, top_k, top_p, temperature, stop_token, num_beams,
                             repetition_penalty=1.0):
        device = conditional_latents.device
        mdtype = next(self.parameters()).dtype
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("use_cuda_graph=True requires CUDA")
        S, attention_mask, kvs, cur_logits = self._prefill(conditional_latents, text_inputs, langs)

        K = num_beams
        S = S + 1  # prefill token count (KV buffer positions 0..S-1)
        pad_len = int((attention_mask[0] == 0).sum().item())
        raw_seq = S + max_new_tokens + 8
        max_seq = ((raw_seq + 63) // 64) * 64
        cache_key = (K, max_seq, mdtype)
        cache = self._beam_graph_cache.get(cache_key)
        if cache is None:
            cache = self._capture_graph(K, max_seq)
            self._beam_graph_cache[cache_key] = cache
        g, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf = cache

        # copy prefill KVs into the front of the static buffers (K replicates)
        for (k, v), (k_buf, v_buf) in zip(kvs, kv_bufs):
            k_buf[:, :, :S, :].copy_(k.expand(K, -1, -1, -1))
            v_buf[:, :, :S, :].copy_(v.expand(K, -1, -1, -1))

        min_dt = torch.finfo(mask_buf.dtype).min
        cur_logits = cur_logits.expand(K, -1).contiguous().float()  # [K,V]
        beam_scores = torch.zeros(K, dtype=torch.float32, device=device)
        gen_ids = torch.empty(K, 0, dtype=torch.long, device=device)
        finished: list[tuple[float, list[int]]] = []
        done = torch.zeros(K, dtype=torch.bool, device=device)
        stop_t = torch.tensor([stop_token], dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            next_id = self._sample(
                cur_logits, do_sample, top_k, top_p, temperature,
                generated_ids=gen_ids, repetition_penalty=repetition_penalty,
            )  # [K]
            tok_lp = F.log_softmax(cur_logits, dim=-1).gather(
                1, next_id.unsqueeze(1)
            ).squeeze(1)  # [K]
            cand_score = beam_scores + tok_lp  # [K] fp32

            is_eos = (next_id == stop_token) & ~done  # newly finished beams
            if is_eos.any():
                eos_scores = cand_score[is_eos].tolist()
                eos_codes = gen_ids[is_eos].tolist()  # codes BEFORE stop
                finished.extend(zip(eos_scores, eos_codes))
                done = done | is_eos
                if done.all():
                    break
            # freeze done beams' scores; active beams accumulate
            beam_scores = torch.where(done, beam_scores, cand_score)
            # gen_ids for active beams only (done rows frozen, never re-read)
            gen_ids = torch.cat([gen_ids, next_id.unsqueeze(1)], dim=1)
            # feed: done beams keep feeding stop_token (logits ignored)
            feed = torch.where(done, stop_t.expand(K), next_id)
            kv_pos = S + step
            input_id_buf.copy_(feed.unsqueeze(1))  # [K,1]
            pos_buf.fill_(step + 2)
            kv_pos_buf.fill_(kv_pos)
            mask_buf.fill_(min_dt)
            mask_buf[:, :, :, pad_len:kv_pos + 1].fill_(0.0)
            g.replay()
            cur_logits = logits_buf.float()  # [K,V] fp32 for stable sampling

        if finished:
            best = max(finished, key=lambda x: x[0])  # highest cumulative score
            return torch.tensor([best[1]], dtype=torch.long, device=device)  # [1, T_gen]
        # none finished: return the best-scoring active beam
        best_beam = int(beam_scores.argmax().item())
        return gen_ids[best_beam].unsqueeze(0)  # [1, T_gen]

    def _graph_decode_step(self, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf, kv_bufs):
        # straight-line static-shape graph body: K/V written at column kv_pos,
        # attention over FULL buffers (mask hides > kv_pos), logits -> buffer
        emb = self.mel_embedding(input_id_buf) + self.mel_pos_embedding.emb(pos_buf).unsqueeze(1)
        hidden, _ = self.gpt(inputs_embeds=emb, attention_mask=mask_buf, kv_bufs=kv_bufs, kv_pos=kv_pos_buf)
        logits_buf.copy_(self.mel_head(self.final_norm(hidden))[:, 0])

    def _capture_graph(self, B, max_seq):
        # returns (graph, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf)
        p = next(self.parameters())
        H, D = self.heads, self.model_dim // self.heads
        kv_bufs = [(torch.zeros(B, H, max_seq, D, dtype=p.dtype, device=p.device),
                    torch.zeros(B, H, max_seq, D, dtype=p.dtype, device=p.device))
                   for _ in range(self.layers)]
        input_id_buf = torch.zeros(B, 1, dtype=torch.long, device=p.device)
        pos_buf = torch.zeros(1, dtype=torch.long, device=p.device)
        kv_pos_buf = torch.zeros(1, dtype=torch.long, device=p.device)
        logits_buf = torch.empty(B, self.number_mel_codes, dtype=p.dtype, device=p.device)
        mask_buf = torch.zeros(B, 1, 1, max_seq, dtype=p.dtype, device=p.device)
        # warmup eagerly 3x: primes cuDNN/cuBLAS autotune + workspace allocs
        # (capture must not observe fresh allocations)
        for _ in range(3):
            self._graph_decode_step(input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf, kv_bufs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._graph_decode_step(input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf, kv_bufs)
        return g, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf

    # CUDA-Graph AR decode — bit-identical to eager generate(). Prefill eager;
    # decode replays the captured graph with static buffers (sampled id, mel pos,
    # KV column, mask updated between replays; sampling stays in Python).
    def _generate_cuda_graph(self, conditional_latents, text_inputs, langs, max_new_tokens,
                             do_sample, top_k, top_p, temperature, stop_token,
                             repetition_penalty=1.0):
        device = conditional_latents.device
        mdtype = next(self.parameters()).dtype
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("use_cuda_graph=True requires CUDA")
        S, attention_mask, kvs, cur_logits = self._prefill(conditional_latents, text_inputs, langs)
        S += 1  # prefill token count (KV buffer positions 0..S-1)
        pad_len = int((attention_mask[0] == 0).sum().item())
        # bucket max_seq to 64 so similar text lengths reuse the graph
        max_seq = ((S + max_new_tokens + 8 + 63) // 64) * 64
        cache_key = (max_seq, mdtype)
        cache = self._graph_cache.get(cache_key)
        if cache is None:
            cache = self._capture_graph(1, max_seq)
            self._graph_cache[cache_key] = cache
        g, kv_bufs, input_id_buf, pos_buf, kv_pos_buf, mask_buf, logits_buf = cache
        for (k, v), (k_buf, v_buf) in zip(kvs, kv_bufs):
            k_buf[:, :, :S].copy_(k)
            v_buf[:, :, :S].copy_(v)

        min_dt = torch.finfo(mask_buf.dtype).min
        codes = []
        gen_ids = torch.empty(1, 0, dtype=torch.long, device=device)
        for step in range(max_new_tokens):
            next_id = self._sample(cur_logits, do_sample, top_k, top_p, temperature, gen_ids, repetition_penalty)
            codes.append(next_id)
            gen_ids = torch.cat([gen_ids, next_id.unsqueeze(1)], 1)
            if next_id.item() == stop_token:
                break
            input_id_buf.copy_(next_id.unsqueeze(1))   # static buffers, outside graph
            pos_buf.fill_(step + 2)
            kv_pos_buf.fill_(S + step)                 # absolute KV column
            mask_buf.fill_(min_dt)                     # mask future + pad cols
            mask_buf[:, :, :, pad_len:S + step + 1].fill_(0.0)
            g.replay()
            cur_logits = logits_buf.float()  # fp32 for stable sampling
        return torch.stack(codes, 1)

    def forward(self, *a, **kw):
        raise NotImplementedError("use prefill_logits_from_inputs / generate")
