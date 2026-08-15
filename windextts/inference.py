"""WIndexTTS inference pipeline — orchestrates the 6-stage IndexTTS-2.5 flow.

End-to-end: text + ref_audio → mel codes (GPT-AR) → mel (S2Mel-CFM) → audio (BigVGAN).

All neural modules are pure-torch re-implementations (see windextts/models/);
the frontend (tokenizer + mel + audio features) is pure-torch/tiktoken. Zero
indextts/transformers/modelscope/whisper/librosa dependency at runtime (the only
external calls are one-time cache builds from the official model, for mel_basis
and SeamlessM4T filters).

Stage flow (infer_v2_5.py):
  ① frontend:  text → text_tokens (tiktoken BPE + <|lang|> prefix)
  ② ref audio: 16kHz → w2v-bert hidden_states[17] (normalized) → spk_cond
                16kHz → CAMPPlus fbank → style [1,192]
                22kHz → mel_fn → ref_mel [1,80,T]
  ③ GPT-AR:    (spk_cond via codec.quantize for training target; at inference
                the codec is only used to decode codes → S_infer)
                conds_latent = spk_proj(style) + emo  →  GPT.generate → codes
  ④ codec.decode(codes) → S_infer [1,2T,1024]
  ⑤ S2Mel:     length_regulate(spk_cond, S_infer) → cat_condition
                CFM.inference(25 steps, cfg=0.7) → mel [1,80,T_target]
  ⑥ BigVGAN:    mel → audio [1,1,T_audio] @ 22050Hz
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windextts.config import Config, load_default_config
from windextts.weights import WeightLoader

# Default output audio config (infer_v2_5.py:514)
OUTPUT_SR = 22050
# ref audio resample targets (infer_v2_5.py:628-629)
REF_SR_W2V = 16000
REF_SR_MEL = 22050
REF_MAX_SECONDS = 15.0


class WIndexTTS:
    # Pure-torch IndexTTS-2.5 pipeline: construct once, then infer() per request.
    def __init__(self, cfg=None, weights_dir=None, device="cuda", dtype=torch.float32,
                 enable_w4a16=False, enable_emo_ref=True, low_vram=False):
        self.cfg = cfg or load_default_config()
        self.device, self.dtype = device, dtype
        self.enable_w4a16 = enable_w4a16  # W4A16 INT4 quantization (GPT only)
        self.low_vram = low_vram
        self.enable_emo_ref = enable_emo_ref and not low_vram
        from windextts.weights import DEFAULT_WEIGHTS_DIR
        self.weights = WeightLoader(weights_dir or DEFAULT_WEIGHTS_DIR)
        self._featurizer = self._mel_fn = self._tokenizer = None
        self.w2v_use_autocast = False  # True when w2v-bert runs fp16
        self._load_modules()

    def _load_modules(self):
        from windextts.models.bigvgan import BigVGAN, BigVGANConfig
        from windextts.models.campplus import CAMPPlus
        from windextts.models.codec import EnhancedCodec
        from windextts.models.gpt import UnifiedVoice
        from windextts.models.length_regulator import InterpolateRegulator
        from windextts.models.s2mel_cfm import S2Mel, S2MelCFM
        from windextts.models.s2mel_dit import DiT
        from windextts.models.w2v2_bert import Wav2Vec2BertConformer

        dev, w = self.device, self.weights

        self.w2v_bert = Wav2Vec2BertConformer().to(dev)
        self.w2v_bert.load_official(w.load_w2v_bert()); self.w2v_bert.eval()
        mean, var = w.load_w2v_stats()
        # stats file stores VARIANCE — using it as std directly shrinks the
        # normalization ~3x and shifts every downstream cond (official: sqrt)
        self.w2v_mean, self.w2v_std = mean.to(dev), torch.sqrt(var).to(dev)

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192).to(dev)
        self.campplus.load_state_dict(w.load_campplus()); self.campplus.eval()

        sc = self.cfg.semantic_codec
        self.codec = EnhancedCodec(
            codebook_size=sc.codebook_size, hidden_size=sc.hidden_size,
            codebook_dim=sc.codebook_dim, vocos_dim=sc.vocos_dim,
            vocos_intermediate_dim=sc.vocos_intermediate_dim, vocos_num_layers=sc.vocos_num_layers,
        ).to(dev)
        self.codec.load_state_dict(w.load_codec()); self.codec.eval()

        self.gpt = UnifiedVoice().to(dev)
        # emo_conditioning (160M) is needed on EVERY path (even calm default
        # mixes conformer emovec_audio). low_vram only skips beam3 graph capture.
        self.gpt.build_emo_conditioning()
        self.gpt.emo_conditioning_encoder = self.gpt.emo_conditioning_encoder.to(dev)
        self.gpt.emo_perceiver_encoder = self.gpt.emo_perceiver_encoder.to(dev)
        self.gpt.load_official(w.load_gpt(), load_emo_conditioning=True)
        self.gpt.eval()

        self.spk_matrix = w.load_spk_matrix().to(dev)   # feat1 [73,192]
        self.emo_matrix = w.load_emo_matrix().to(dev)   # feat2 [73,1280]
        self.emo_num = list(self.cfg._raw.get("emo_num", [3, 17, 2, 8, 4, 5, 10, 24]))

        net = w.load_s2mel()
        lr = self.cfg.s2mel.length_reg
        self.length_regulator = InterpolateRegulator(
            channels=lr.channels, sampling_ratios=lr.sampling_ratios, is_discrete=lr.is_discrete,
            in_channels=lr.in_channels, codebook_size=lr.content_codebook_size).to(dev)
        self.length_regulator.load_official(net["length_regulator"]); self.length_regulator.eval()

        self.dit = DiT().to(dev)
        self.dit.load_official(net["cfm"]); self.dit.eval()
        cfm = S2MelCFM(self.dit, in_channels=self.cfg.s2mel.dit.in_channels).to(dev).eval()
        self.s2mel = S2Mel(self.length_regulator, cfm).to(dev).eval()

        bcfg = BigVGANConfig.from_json(Path(self.weights.dir) / "hf_cache" / "bigvgan" / "config.json")
        self.bigvgan = BigVGAN(bcfg).to(dev)
        self.bigvgan.load_official(w.load_bigvgan()); self.bigvgan.eval()
        # Flatten weight_norm -> plain weights: kills the _forward_pre_hook per
        # conv (~149ms host dispatch bubble; profiler: 91% of BigVGAN's gaps)
        self.bigvgan.remove_weight_norm()

        print(f">> WIndexTTS loaded all modules on {dev}")
        # S2Mel CUDA Graph also for fp32 — verified bit-identical to eager
        self.s2mel_use_graph = True

        # fp16 fast path — verified stable (0/30 brick stress) with graphs on.
        # (Historic root causes fixed & documented in s2mel_cfm.py: dt_buf GC,
        # keep_mask reflect-pad leakage.)
        if self.dtype == torch.float16:
            # w2v-bert fp16 (cosine 0.99997, saves 1.16GB; runs once, no AR
            # error cascade)
            self.w2v_bert.to(torch.float16)
            self.w2v_mean, self.w2v_std = self.w2v_mean.half(), self.w2v_std.half()
            self.gpt.to(torch.float16)
            self.bigvgan.to(torch.float16)
            self.codec.to(torch.float16)          # decode feeds length_regulator
            self.length_regulator.to(torch.float16)
            self.s2mel.cfm.estimator.to(torch.float16)
            self.s2mel.cfm.estimator_fp16_weights = True
            self.s2mel_use_graph = True
            mode = "fp16"
            if self.enable_w4a16:
                # torchao int4 tinygemm requires bf16; GPT-AR greedy identical
                # in bf16 vs fp16 (101/101). DiT/BigVGAN stay fp16 (bf16
                # dtype-mix issues).
                self.gpt.to(torch.bfloat16)
                self._apply_w4a16()
                mode = "W4A16 (GPT int4 bf16 + DiT/BigVGAN fp16)"
            print(f">> GPT-AR + BigVGAN + S2Mel-DiT {mode} (CUDA Graph enabled)")

        # ref-audio feature cache: (path, mtime) → skips w2v/campplus/mel recompute
        self._ref_cache = {}
        self._normalizer = None
        self._qwen_emo = None

    def _ensure_normalizer(self):
        """Lazily build the TextNormalizer (heavy: loads NeMo TN grammars)."""
        if self._normalizer is None:
            from windextts.frontend.normalizer import TextNormalizer
            self._normalizer = TextNormalizer()
        return self._normalizer

    def _ensure_qwen_emo(self):
        """Lazily load the QwenEmotion text→emotion predictor (~1.2GB, fp16)."""
        if self._qwen_emo is None:
            import os
            # qwen0.6bemo4-merge lives directly under the model dir (config.yaml:
            # qwen_emo_path). Hardcoded name since it's fixed for IndexTTS-2.5.
            emo_dir = os.path.join(str(self.weights.dir), "qwen0.6bemo4-merge")
            if not os.path.isdir(emo_dir):
                raise FileNotFoundError(
                    f"QwenEmotion model not found at {emo_dir}. emo_text requires "
                    "the qwen0.6bemo4-merge directory in the model path."
                )
            from windextts.models.qwen_emotion import QwenEmotion
            self._qwen_emo = QwenEmotion(emo_dir, device=self.device, dtype=torch.float16)
            print(">> QwenEmotion loaded (pure-torch Qwen3, text→emotion)")
        return self._qwen_emo

    def _apply_w4a16(self, group_size: int = 128) -> None:
        """Quantize GPT transformer body nn.Linear weights to INT4 (W4A16).

        Uses torchao int4_weight_only (tinygemm kernel, bf16 activations).
        GPT-AR greedy codes are identical in bf16 vs fp16 (verified 101/101),
        so the bf16 cast is lossless; INT4 adds small quantization error.
        DiT/BigVGAN are NOT quantized (kept fp16).

        Optional dependency: torchao. If unavailable, logs and no-ops.
        """
        try:
            from torchao.quantization import int4_weight_only, quantize_
        except ImportError:
            print(">> W4A16: torchao not installed, skipping quantization")
            return
        import torch.nn as nn_torch
        nn_lin = nn_torch.Linear
        # Quantize all nn.Linear in the GPT transformer body (gpt.gpt.h.*.c_attn/
        # c_proj/c_fc, plus emo/spk projections if Linear).
        n_before = sum(1 for m in self.gpt.gpt.modules() if isinstance(m, nn_lin))
        quantize_(
            self.gpt.gpt,
            int4_weight_only(group_size=group_size),
            filter_fn=lambda mod, _fqn: isinstance(mod, nn_lin),
        )
        # Also quantize the lm_head (mel_head) and emo/spk Linear if present.
        for attr in ("mel_head", "final_norm"):
            pass  # final_norm is LayerNorm (skip); mel_head quantized below
        if isinstance(self.gpt.mel_head, nn_lin):
            quantize_(
                self.gpt.mel_head,
                int4_weight_only(group_size=group_size),
                filter_fn=lambda mod, _fqn: isinstance(mod, nn_lin),
            )
        n_after = sum(1 for m in self.gpt.gpt.modules()
                      if isinstance(m, nn_lin) and "Int4" in type(getattr(m.weight, "__class__", type(None))).__name__)
        print(f">> W4A16: quantized GPT body ({n_before} nn.Linear, int4 tinygemm, group={group_size})")

    def warmup(self):
        # Pre-capture CUDA Graphs + prime cuDNN autotune (shifts ~1s cold-start
        # from the first infer() to load time).
        # NOTE: DiT bf16 autocast is NET SLOWER at batch=1 (207ms vs 175ms fp32):
        # autocast's per-op dispatch exceeds the kernel saving when host-bound.
        # (bf16+graph would recover it, but graph bucketing has numerics issues.)
        import torchaudio
        dev = self.device
        # dummy ref audio (1s silence) to populate caches + capture graphs
        dummy = torch.zeros(1, 16000, device=dev)
        a16 = dummy
        a22 = torch.zeros(1, 22050, device=dev, dtype=torch.float32)
        with torch.no_grad():
            spk = self.extract_spk_cond(a16)
            style = self.extract_style(a16)
            refmel = self.mel_fn(a22)
            emo = self.build_emo_vec(style, spk)
            conds = self.gpt.build_conds_latent(style, emo)
        tt = torch.tensor([[1, 2, 3, 1]], device=dev, dtype=torch.int)
        # low_vram bucket alignment: a full-length segment (120 text tokens incl.
        # stop) + 390 max mel tokens + 8 slack → max_seq 576, the same bucket the
        # capped runtime requests will hit.
        tt_long = torch.ones(1, 120, device=dev, dtype=torch.int)
        from windextts.frontend.tokenizer import lang_to_token
        lang = torch.LongTensor([lang_to_token("ZH")]).to(dev)
        with torch.no_grad():
            use_cg = dev != "cpu"
            # beam3 graph always (production quality config). low_vram: beam3
            # only, at the bucket the capped 390 max_mel_tokens hits (max_seq
            # 576 = 390 + 124 + 8); greedy graph skipped (0.09GB saved).
            warm_tokens = 390 if self.low_vram else 720
            codes = self.gpt.generate(
                conds, tt_long if self.low_vram else tt, lang,
                max_new_tokens=warm_tokens, do_sample=True,
                top_k=30, top_p=0.8, temperature=0.8,
                stop_token=self.cfg.gpt.stop_mel_token, use_cuda_graph=use_cg,
                repetition_penalty=10.0, num_beams=3,
            )
            if not self.low_vram:
                # greedy graph too (do_sample=False callers), at the typical
                # runtime bucket (~192): every decode step's attention/mask
                # kernels scan the FULL KV pool, so an oversized bucket (768)
                # makes each replay ~4x slower. Longer requests capture their own.
                self.gpt.generate(
                    conds, tt, lang, max_new_tokens=150, do_sample=False,
                    stop_token=self.cfg.gpt.stop_mel_token, use_cuda_graph=use_cg,
                )
            s = self.codec.decode(codes[:, :-1] if codes[0, -1] == self.cfg.gpt.stop_mel_token else codes)
            mel = self.s2mel.inference(spk, s, refmel, style, n_timesteps=12)
            bg_dtype = next(self.bigvgan.parameters()).dtype
            _ = self.bigvgan(mel.to(bg_dtype))
        torch.cuda.synchronize()

    @property
    def featurizer(self):
        if self._featurizer is None:
            from windextts.frontend.audio_utils import SeamlessM4TFeaturizer
            self._featurizer = SeamlessM4TFeaturizer(device=self.device)
        return self._featurizer

    @property
    def mel_fn(self):
        if self._mel_fn is None:
            from windextts.frontend.mel import MelSpectrogram
            self._mel_fn = MelSpectrogram(device=self.device)
        return self._mel_fn

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from windextts.frontend.tokenizer import build_tokenizer
            self._tokenizer = build_tokenizer(model_dir=str(self.weights.dir))
        return self._tokenizer

    def _load_audio(self, path, max_seconds=REF_MAX_SECONDS):
        audio, sr = torchaudio.load(path)
        return (audio[:, : int(max_seconds * sr)], sr)  # truncate

    @torch.no_grad()
    def extract_spk_cond(self, audio_16k):
        # w2v-bert hidden_states[17] normalized -> [B,T,1024]. low_vram: weights
        # streamed CPU->GPU only while encoding (~0.5s); same-ref hits _ref_cache.
        if self.low_vram and next(self.w2v_bert.parameters()).device.type == "cpu":
            self.w2v_bert.to(self.device)
        inp, am = self.featurizer(audio_16k.to(self.device), return_mask=True)
        # am: official semantics — a stacked tail frame over a padded row gets
        # mask 0 (indices % stride == 1 rule). Matters: masked positions alter
        # conformer attention and hidden_states[17] for EVERY frame (maxdiff ~11
        # with an all-ones mask vs official).
        wb = self.w2v_bert.feature_projection.layer_norm.weight.dtype
        out = (self.w2v_bert(inp.to(wb), am, return_layer=17) - self.w2v_mean) / self.w2v_std
        if self.low_vram:
            self.w2v_bert.to("cpu")  # free 1.16GB immediately
            torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def extract_style(self, audio_16k):
        # CAMPPlus [1,192]; fbank mean-subtracted per column
        f = torchaudio.compliance.kaldi.fbank(audio_16k.to(self.device), num_mel_bins=80, dither=0, sample_frequency=REF_SR_W2V)
        return self.campplus((f - f.mean(0, keepdim=True)).unsqueeze(0))

    def build_emo_vec(self, style, spk_cond, emo_vector=None):
        # matrix+conformer blend (infer_v2_5.py:757-764):
        #   emovec = emovec_mat(RAW w) + (1 - sum(RAW w)) * get_emovec(spk)
        # The complement term is load-bearing: omitting it lost the speaker's
        # base characteristics -> 'brick' audio at high single-emotion weights.
        if emo_vector is None:
            emo_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]  # calm default
        w = torch.tensor(emo_vector, device=self.device, dtype=torch.float32)
        dt = self.gpt.emovec_layer.weight.dtype
        mat = self.gpt.emo_matrix_lookup(
            style, w, torch.split(self.spk_matrix, self.emo_num), torch.split(self.emo_matrix, self.emo_num))
        # matrix path stays RAW (feat2.pt is already in target space); only the
        # conformer emovec passes through emovec_layer+emo_layer (in get_emovec)
        return mat.to(dt) + float((1.0 - w.sum()).clamp(min=0.0)) * self.gpt.get_emovec(spk_cond.to(dt))

    @torch.no_grad()
    def build_emo_vec_full(self, style, spk_cond, emo_vector, emo_ref_path, emo_alpha):
        # All three control modes (infer_v2_5.py:757-768). Conformer path ALWAYS
        # runs; the matrix lookup only mixes in when emo_vector is explicit.
        #   emo_ref_path: emo_cond from that audio; else emo_cond = spk_cond
        #   (official doubles the spk ref as emo ref, infer_v2_5.py:692).
        #   emo_vector given: mat(RAW w) + (1-sum) * conformer.
        #   else: pure conformer (official default; a calm matrix fallback here
        #   diverged conds_latent and shifted the whole GPT trajectory).
        if emo_ref_path is None:
            emo_cond = spk_cond
        else:
            ea, esr = self._load_audio(emo_ref_path, REF_MAX_SECONDS)
            emo_cond = self.extract_spk_cond(torchaudio.transforms.Resample(esr, REF_SR_W2V)(ea))
        dt = self.gpt.emovec_layer.weight.dtype
        emovec_audio = self.gpt.merge_emovec(spk_cond.to(dt), emo_cond.to(dt), alpha=emo_alpha)
        if emo_vector is None:
            return emovec_audio
        w = torch.tensor(emo_vector, device=self.device, dtype=torch.float32)
        mat = self.gpt.emo_matrix_lookup(
            style, w, torch.split(self.spk_matrix, self.emo_num), torch.split(self.emo_matrix, self.emo_num))
        return mat.to(dt) + float((1.0 - w.sum()).clamp(min=0.0)) * emovec_audio

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    # Zero-shot voice cloning. emo_vector: 8 weights [happy,angry,sad,afraid,
    # disgusted,melancholic,surprised,calm]; emo_text overrides it (QwenEmotion);
    # emo_ref_path: emotion reference audio. max_mel_tokens None = auto per
    # segment (token count x language ratio x2 headroom +8 — keeps the GPT graph
    # KV pool tight since per-step kernels scan the FULL pool). Long text is
    # segmented (max_text_tokens_per_segment) and joined with interval silence.
    # Returns (sr, audio [samples,]) @22050Hz.
    @torch.no_grad()
    def infer(self, spk_audio_prompt, text, lang="ZH", emo_vector=None, emo_text=None,
              emo_ref_path=None, emo_alpha=1.0, duration_factor=1.0, do_sample=True,
              top_p=0.8, top_k=30, temperature=0.8, max_mel_tokens=None, cfm_steps=15,
              cfg_rate=0.7, text_normalization=True, max_text_tokens_per_segment=120,
              interval_silence_ms=200, repetition_penalty=10.0, num_beams=3):
        dev = self.device

        # text→mel ratio per language, MEASURED (Tatoeba 50 sents × 99 langs,
        # greedy): ceil(median×2) capped 14 — see scripts/measure_lang_ratios.py.
        # Script-family grouping visible: South Asian 4-6x, CJK/syllabic 6-12x,
        # Cyrillic 8-10x, Latin 11-14x. Unmeasured fall back to 14 (safest).
        _MEL_RATIO = {
            "AF": 13, "AM": 7, "AR": 14, "AS": 5, "AZ": 10, "BA": 6, "BE": 8,
            "BG": 9, "BN": 5, "BO": 4, "BR": 13, "BS": 13, "CA": 14, "CS": 11,
            "CY": 11, "DA": 13, "DE": 13, "EL": 7, "EN": 14, "ES": 14, "ET": 14,
            "EU": 13, "FA": 11, "FI": 11, "FO": 13, "FR": 14, "GL": 14, "GU": 4,
            "HA": 12, "HAW": 10, "HE": 9, "HI": 6, "HR": 12, "HT": 14, "HU": 12,
            "HY": 6, "ID": 13, "IS": 10, "IT": 13, "JA": 12, "JW": 10, "KA": 6,
            "KK": 6, "KM": 6, "KN": 6, "KO": 10, "LA": 14, "LB": 12, "LN": 11,
            "LO": 4, "LT": 12, "LV": 10, "MG": 14, "MI": 11, "MK": 8, "ML": 4,
            "MN": 6, "MR": 5, "MS": 13, "MT": 11, "MY": 6, "NE": 5, "NL": 14,
            "NN": 14, "NO": 14, "OC": 14, "PA": 4, "PL": 11, "PS": 9, "PT": 14,
            "RO": 11, "RU": 9, "SA": 6, "SI": 6, "SK": 12, "SL": 13, "SN": 13,
            "SO": 13, "SQ": 11, "SR": 10, "SU": 12, "SV": 13, "SW": 11, "TA": 6,
            "TE": 5, "TG": 7, "TH": 6, "TK": 11, "TL": 14, "TR": 12, "TT": 8,
            "UK": 9, "UR": 10, "UZ": 12, "VI": 10, "YI": 6, "YO": 10, "ZH": 13,
            "YUE": 13, "MINNAN": 13, "WUYU": 13,  # CJK family (inferred from ZH)
        }
        if self.low_vram:
            # 390 mel tokens ≈ 7.8s/segment; budget shrinks to match (ZH 13:
            # 30 text × 13 = 390) so the warmup-captured beam3 bucket (576) is
            # reused at runtime without truncation.
            if max_mel_tokens is not None:
                max_mel_tokens = min(max_mel_tokens, 390)
            max_text_tokens_per_segment = min(max_text_tokens_per_segment, 30)

        # --- text normalization (G2P: digits→words, punctuation, names) ---
        if text_normalization:
            norm = self._ensure_normalizer()
            text = norm.normalize(text)

        # --- emo_text → emo_vector (via pure-torch QwenEmotion) ---
        if emo_text is not None:
            qe = self._ensure_qwen_emo()
            emo_vector = qe.inference(emo_text)

        # --- long-text segmentation (split by punctuation + token budget) ---
        from windextts.frontend.segmenter import split_text_by_tokens
        lang_prefix = f"<|{lang.lower()}|> "
        enc = lambda s: self.tokenizer.encode(s, allowed_special="all")
        segments = split_text_by_tokens(text, enc, max_tokens=max_text_tokens_per_segment, lang_prefix=lang_prefix)

        def _seg_mel_cap(seg: str) -> int:
            # Per-segment mel-token cap from the segment's ACTUAL token count:
            # ratio * n_tokens * 2 (headroom for prosody/emotion drift keeps a
            # segment un-truncated) + 8. Keeps the GPT graph KV pool tight
            # (the graph's per-step attention/mask kernels scan the FULL pool,
            # so an oversized cap slows every decode step).
            if max_mel_tokens is not None:
                return max_mel_tokens
            ratio = _MEL_RATIO.get(lang.upper(), 14)
            n_tok = len(enc(lang_prefix + seg))
            cap = max(int(n_tok * ratio * 2) + 8, 64)
            if self.low_vram:
                cap = min(cap, 390)
            return cap

        if len(segments) == 1:
            return self._infer_single(
                spk_audio_prompt, segments[0], lang, emo_vector,
                emo_ref_path, emo_alpha,
                duration_factor, do_sample, top_p, top_k, temperature,
                _seg_mel_cap(segments[0]), cfm_steps, cfg_rate,
                repetition_penalty, num_beams,
            )

        # multi-segment: synthesize each, join with silence
        wavs = []
        for seg in segments:
            sr, wav = self._infer_single(
                spk_audio_prompt, seg, lang, emo_vector,
                emo_ref_path, emo_alpha,
                duration_factor, do_sample, top_p, top_k, temperature,
                _seg_mel_cap(seg), cfm_steps, cfg_rate,
                repetition_penalty, num_beams,
            )
            wavs.append(wav)
        silence = torch.zeros(int(OUTPUT_SR * interval_silence_ms / 1000))
        parts = []
        for i, w in enumerate(wavs):
            parts.append(w)
            if i < len(wavs) - 1:
                parts.append(silence)
        return OUTPUT_SR, torch.cat(parts)

    def _infer_single(
        self,
        spk_audio_prompt: str,
        text: str,
        lang: str,
        emo_vector: list[float] | None,
        emo_ref_path: str | None,
        emo_alpha: float,
        duration_factor: float,
        do_sample: bool,
        top_p: float,
        top_k: int,
        temperature: float,
        max_mel_tokens: int,
        cfm_steps: int,
        cfg_rate: float,
        repetition_penalty: float = 10.0,
        num_beams: int = 3,
    ) -> tuple[int, torch.Tensor]:
        """Synthesize a single (already normalized, short) text segment."""
        dev = self.device
        if self.low_vram and num_beams > 3:
            # beam3 is the official quality config and still fits the budget;
            # only wider beams (4+) would blow it.
            num_beams = 3
        # --- ref audio features (cached by path+mtime; stage 5 overlap/cache) ---
        import os
        cache_key = None
        try:
            cache_key = (spk_audio_prompt, os.path.getmtime(spk_audio_prompt))
        except OSError:
            pass
        cached = self._ref_cache.get(cache_key) if cache_key else None
        if cached is None:
            audio, sr = self._load_audio(spk_audio_prompt, REF_MAX_SECONDS)
            a16 = torchaudio.transforms.Resample(sr, REF_SR_W2V)(audio)
            a22 = torchaudio.transforms.Resample(sr, REF_SR_MEL)(audio).to(dev).float()
            spk_cond = self.extract_spk_cond(a16)
            style = self.extract_style(a16)
            ref_mel = self.mel_fn(a22)
            if cache_key is not None:
                self._ref_cache[cache_key] = (spk_cond, style, ref_mel)
        else:
            spk_cond, style, ref_mel = cached

        # --- text tokens ---
        lang_prefix = f"<|{lang.lower()}|> "
        toks = self.tokenizer.encode(lang_prefix + text, allowed_special="all")
        text_tokens = torch.IntTensor(toks).unsqueeze(0).to(dev)
        text_tokens = F.pad(text_tokens, (0, 1), value=1)  # stop_text
        from windextts.frontend.tokenizer import lang_to_token
        lang_id = torch.LongTensor([lang_to_token(lang)]).to(dev)

        # --- emo vec assembly ---
        # Three paths (priority: emo_ref_path > emo_vector > calm default):
        #  (1) emo_ref_path: conformer path — merge_emovec(spk, emo_ref) then
        #      optionally add matrix correction if emo_vector also given.
        #      Replicates infer_v2_5.py:757-764.
        #  (2) emo_vector only: matrix path (build_emo_vec).
        #  (3) neither: calm default vector.
        emo_vec = self.build_emo_vec_full(
            style, spk_cond, emo_vector, emo_ref_path, emo_alpha
        )  # [1,1280]

        # --- GPT conditioning + AR decode ---
        conds_latent = self.gpt.build_conds_latent(style, emo_vec)  # [1,3,1280]
        # CUDA Graph is now supported for beam search too (static batch K,
        # fixed KV buffers — no beam removal/reordering in the graph loop).
        use_cg = self.device != "cpu"
        codes = self.gpt.generate(
            conds_latent, text_tokens, lang_id,
            max_new_tokens=max_mel_tokens, do_sample=do_sample,
            top_k=top_k, top_p=top_p, temperature=temperature,
            stop_token=self.cfg.gpt.stop_mel_token,
            use_cuda_graph=use_cg,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )  # [1, T_codes]

        # strip stop token if present
        if codes[0, -1].item() == self.cfg.gpt.stop_mel_token:
            codes = codes[:, :-1]

        # --- codec.decode → S_infer ---
        s_infer = self.codec.decode(codes)  # [1, 2*T, 1024]

        # --- S2Mel-CFM → mel ---
        mel = self.s2mel.inference(
            spk_cond, s_infer, ref_mel, style,
            duration_factor=duration_factor, n_timesteps=cfm_steps,
            inference_cfg_rate=cfg_rate,
            use_graph=getattr(self, "s2mel_use_graph", False),
        )  # [1, 80, T_target]

        # --- BigVGAN → audio ---
        # cast mel to BigVGAN's compute dtype (weight_norm makes .weight report
        # fp32 even when params are fp16; use conv_pre.bias which reflects truth)
        bg_dtype = next(self.bigvgan.parameters()).dtype
        audio_out = self.bigvgan(mel.to(bg_dtype))  # [1, 1, T_audio]
        audio_out = audio_out.squeeze(0).squeeze(0).clamp(-1, 1).cpu()
        return OUTPUT_SR, audio_out

    def infer_from_codes(
        self,
        spk_audio_prompt: str,
        text: str,
        lang: str,
        codes: torch.Tensor,
        cfm_steps: int = 15,
        cfg_rate: float = 0.7,
        duration_factor: float = 1.0,
        use_graph: bool | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Synthesize audio from externally supplied GPT mel codes.

        Diagnostic seam: skips win GPT decode entirely (codes come from
        outside — e.g. official IndexTTS), then runs the win downstream
        (codec.decode → S2Mel-CFM → BigVGAN). Used for A/B isolation of the
        GPT stage from the rest of the pipeline.

        Args:
            codes: [1, T] or [T] int tensor, WITHOUT the stop token
                (stop token, if present at the end, is stripped).
        Returns:
            (sample_rate, audio [samples,])
        """
        dev = self.device
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        codes = codes.to(dev).long()
        if codes[0, -1].item() == self.cfg.gpt.stop_mel_token:
            codes = codes[:, :-1]

        with torch.no_grad():
            # --- ref audio features (same path as _infer_single) ---
            import os
        cache_key = None
        try:
            cache_key = (spk_audio_prompt, os.path.getmtime(spk_audio_prompt))
        except OSError:
            pass
        with torch.no_grad():
            cached = self._ref_cache.get(cache_key) if cache_key else None
            if cached is None:
                audio, sr = self._load_audio(spk_audio_prompt, REF_MAX_SECONDS)
                a16 = torchaudio.transforms.Resample(sr, REF_SR_W2V)(audio)
                a22 = torchaudio.transforms.Resample(sr, REF_SR_MEL)(audio).to(dev).float()
                spk_cond = self.extract_spk_cond(a16)
                style = self.extract_style(a16)
                ref_mel = self.mel_fn(a22)
                if cache_key is not None:
                    self._ref_cache[cache_key] = (spk_cond, style, ref_mel)
            else:
                spk_cond, style, ref_mel = cached

            # --- text tokens (needed for spk conditioning path consistency) ---
            lang_prefix = f"<|{lang.lower()}|> "
            toks = self.tokenizer.encode(lang_prefix + text, allowed_special="all")
            text_tokens = torch.IntTensor(toks).unsqueeze(0).to(dev)
            text_tokens = F.pad(text_tokens, (0, 1), value=1)  # stop_text
            emo_vec = self.build_emo_vec_full(style, spk_cond, None, None, 1.0)
            conds_latent = self.gpt.build_conds_latent(style, emo_vec)

            # --- codec.decode → S_infer ---
            s_infer = self.codec.decode(codes)  # [1, 2*T, 256]

            # --- S2Mel-CFM → mel ---
            mel = self.s2mel.inference(
                spk_cond, s_infer, ref_mel, style,
                duration_factor=duration_factor, n_timesteps=cfm_steps,
                inference_cfg_rate=cfg_rate,
                use_graph=(getattr(self, "s2mel_use_graph", False)
                           if use_graph is None else use_graph),
            )

            # --- BigVGAN → audio ---
            bg_dtype = next(self.bigvgan.parameters()).dtype
            audio_out = self.bigvgan(mel.to(bg_dtype))
            audio_out = audio_out.squeeze(0).squeeze(0).clamp(-1, 1).cpu()
            return OUTPUT_SR, audio_out

if __name__ == "__main__":
    # smoke: end-to-end inference (requires GPT AR generate to be implemented)
    import soundfile as sf

    tts = WIndexTTS(device="cuda")
    sr, wav = tts.infer(
        spk_audio_prompt="/root/WIndexTTS/test.wav",
        text="大家好，这是一个测试。",
        lang="ZH",
    )
    out_path = "/root/windextts_dumps/inference_smoke.wav"
    sf.write(out_path, wav.numpy(), sr)
    print(f">> wrote {out_path} ({wav.numel()/sr:.2f}s @ {sr}Hz)")
