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
    """Pure-torch IndexTTS-2.5 inference pipeline.

    Construct once (loads all weights + modules), then call ``infer()`` per request.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        weights_dir: str | Path = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg or load_default_config()
        self.device = device
        self.dtype = dtype
        from windextts.weights import DEFAULT_WEIGHTS_DIR
        self.weights = WeightLoader(weights_dir or DEFAULT_WEIGHTS_DIR)

        # lazy-built frontend caches (built on first use if missing)
        self._featurizer = None
        self._mel_fn = None
        self._tokenizer = None

        self._load_modules()

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------

    def _load_modules(self) -> None:
        from windextts.models.bigvgan import BigVGAN, BigVGANConfig
        from windextts.models.campplus import CAMPPlus
        from windextts.models.codec import EnhancedCodec
        from windextts.models.gpt import UnifiedVoice
        from windextts.models.length_regulator import InterpolateRegulator
        from windextts.models.s2mel_cfm import S2Mel, S2MelCFM
        from windextts.models.s2mel_dit import DiT
        from windextts.models.w2v2_bert import Wav2Vec2BertConformer

        dev = self.device
        w = self.weights

        # w2v-bert conformer
        self.w2v_bert = Wav2Vec2BertConformer().to(dev)
        self.w2v_bert.load_official(w.load_w2v_bert())
        self.w2v_bert.eval()
        mean, var = w.load_w2v_stats()
        self.w2v_mean = mean.to(dev)
        self.w2v_std = var.to(dev)

        # CAMPPlus
        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192).to(dev)
        self.campplus.load_state_dict(w.load_campplus())
        self.campplus.eval()

        # EnhancedCodec
        sc = self.cfg.semantic_codec
        self.codec = EnhancedCodec(
            codebook_size=sc.codebook_size, hidden_size=sc.hidden_size,
            codebook_dim=sc.codebook_dim, vocos_dim=sc.vocos_dim,
            vocos_intermediate_dim=sc.vocos_intermediate_dim, vocos_num_layers=sc.vocos_num_layers,
        ).to(dev)
        self.codec.load_state_dict(w.load_codec())
        self.codec.eval()

        # GPT-AR (UnifiedVoice)
        self.gpt = UnifiedVoice().to(dev)
        # Build + load emo_conditioning (160M) for the emo_ref_audio path.
        # This adds ~1.2GB to the GPT footprint but enables reference-audio
        # emotion extraction (official emo_audio_prompt feature).
        self.gpt.build_emo_conditioning()
        self.gpt.emo_conditioning_encoder = self.gpt.emo_conditioning_encoder.to(dev)
        self.gpt.emo_perceiver_encoder = self.gpt.emo_perceiver_encoder.to(dev)
        self.gpt.load_official(w.load_gpt(), load_emo_conditioning=True)
        self.gpt.eval()

        # emo matrices
        self.spk_matrix = w.load_spk_matrix().to(dev)   # feat1 [73,192]
        self.emo_matrix = w.load_emo_matrix().to(dev)   # feat2 [73,1280]
        self.emo_num = list(self.cfg._raw.get("emo_num", [3, 17, 2, 8, 4, 5, 10, 24]))

        # S2Mel: length_regulator + DiT + CFM
        net = w.load_s2mel()
        lr_cfg = self.cfg.s2mel.length_reg
        self.length_regulator = InterpolateRegulator(
            channels=lr_cfg.channels, sampling_ratios=lr_cfg.sampling_ratios,
            is_discrete=lr_cfg.is_discrete, in_channels=lr_cfg.in_channels,
            codebook_size=lr_cfg.content_codebook_size,
        ).to(dev)
        self.length_regulator.load_official(net["length_regulator"])
        self.length_regulator.eval()

        self.dit = DiT().to(dev)
        self.dit.load_official(net["cfm"])
        self.dit.eval()

        cfm = S2MelCFM(self.dit, in_channels=self.cfg.s2mel.dit.in_channels).to(dev).eval()
        self.s2mel = S2Mel(self.length_regulator, cfm).to(dev).eval()

        # BigVGAN
        bcfg = BigVGANConfig.from_json(Path(self.weights.dir) / "hf_cache" / "bigvgan" / "config.json")
        self.bigvgan = BigVGAN(bcfg).to(dev)
        self.bigvgan.load_official(w.load_bigvgan())
        self.bigvgan.eval()
        # Flatten weight_norm -> plain weights (official path). Eliminates the
        # _forward_pre_hook on every conv, removing ~149ms of host dispatch
        # bubble (profiler: 91% of BigVGAN's large gaps were conv1d dispatch).
        n = self.bigvgan.remove_weight_norm()
        # weight_norm params are fp32 here; if BigVGAN later cast to fp16,
        # the flattened weight goes along with the cast.

        print(f">> WIndexTTS loaded all modules on {dev}")
        # apply per-module precision overrides (mixed-precision fast paths)
        # All three modules run fp16 — verified stable (0/30 brick in stress
        # test across multiple texts/seeds) with CUDA Graph enabled.
        #
        # S2Mel CUDA Graph is ENABLED (s2mel_use_graph=True) after fixing the
        # root-cause bug (dt_buf GC): the Euler timestep buffer `dt_buf` was a
        # local variable not retained by the graph cache, so Python GC freed it
        # and PyTorch's caching allocator reused its memory — the captured graph
        # then read garbage dt values on replay, producing brick/clipped audio.
        # Fix: store dt_buf in the cache dict to keep a Python reference alive.
        # Additional defensive fix: keep_mask zeroes prompt+padding regions each
        # Euler step to prevent WN conv reflect-pad leakage.
        if self.dtype == torch.float16:
            self.gpt.to(torch.float16)
            self.bigvgan.to(torch.float16)
            self.s2mel.cfm.estimator.to(torch.float16)
            self.s2mel.cfm.estimator_fp16_weights = True
            self.s2mel_use_graph = True  # CUDA Graph enabled (dt_buf bug fixed)
            print(">> GPT-AR + BigVGAN + S2Mel-DiT fp16 (CUDA Graph enabled)")

        # ref-audio feature cache: keyed by (path, mtime) → avoids recomputing
        # w2v/campplus/mel when the same ref is reused across requests (stage 5).
        self._ref_cache: dict = {}
        # lazy-loaded text-processing modules (only built on first use)
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

    def warmup(self) -> None:
        """Pre-capture CUDA Graphs + prime cuDNN autotune with dummy data.

        Shifts the ~1s cold-start cost (graph capture + autotune) from the
        first infer() call to model-load time. Call once after __init__ if
        you care about first-request latency.
        """
        # NOTE: DiT bf16 autocast was tested profiler-free and is NET SLOWER
        # for single-request inference (207ms vs 175ms fp32). bf16 kernels are
        # faster but autocast's per-op dispatch overhead exceeds the kernel
        # saving at batch=1 (the host becomes the bottleneck). Keep S2Mel fp32.
        # (bf16+graph would recover it, but graph bucketing has numerics issues.)
        # self.s2mel.cfm.estimator_autocast_dtype = torch.bfloat16
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
        from windextts.frontend.tokenizer import lang_to_token
        lang = torch.LongTensor([lang_to_token("ZH")]).to(dev)
        with torch.no_grad():
            use_cg = dev != "cpu"
            codes = self.gpt.generate(
                conds, tt, lang, max_new_tokens=720, do_sample=False,
                stop_token=self.cfg.gpt.stop_mel_token, use_cuda_graph=use_cg,
            )
            # also pre-capture the beam-search graph (default production path:
            # num_beams=3 → static batch K=3 CUDA Graph)
            self.gpt.generate(
                conds, tt, lang, max_new_tokens=720, do_sample=True,
                top_k=30, top_p=0.8, temperature=0.8,
                stop_token=self.cfg.gpt.stop_mel_token, use_cuda_graph=use_cg,
                repetition_penalty=10.0, num_beams=3,
            )
            s = self.codec.decode(codes[:, :-1] if codes[0, -1] == self.cfg.gpt.stop_mel_token else codes)
            self.s2mel.cfm.estimator.enable_teacache(thresh=0.25)
            mel = self.s2mel.inference(spk, s, refmel, style, n_timesteps=12)
            bg_dtype = next(self.bigvgan.parameters()).dtype
            _ = self.bigvgan(mel.to(bg_dtype))
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # frontend (lazy)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # audio loading + feature extraction
    # ------------------------------------------------------------------

    def _load_audio(self, path: str, max_seconds: float = REF_MAX_SECONDS) -> tuple[torch.Tensor, int]:
        """Load + truncate ref audio to max_seconds. Returns (audio[sr], sr)."""
        audio, sr = torchaudio.load(path)
        # truncate
        max_samples = int(max_seconds * sr)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        return audio, sr

    @torch.no_grad()
    def extract_spk_cond(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """w2v-bert hidden_states[17] normalized → [B, T, 1024]."""
        inp = self.featurizer(audio_16k.to(self.device))
        am = torch.ones(inp.shape[:2], dtype=torch.int32, device=self.device)
        feat = self.w2v_bert(inp, am, return_layer=17)
        return (feat - self.w2v_mean) / self.w2v_std

    @torch.no_grad()
    def extract_style(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """CAMPPlus speaker/style embedding → [1, 192]."""
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device), num_mel_bins=80, dither=0, sample_frequency=REF_SR_W2V
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus(feat.unsqueeze(0))

    # ------------------------------------------------------------------
    # emo vector handling (infer_v2_5.py:672-678 + normalize_emo_vec)
    # ------------------------------------------------------------------

    def build_emo_vec(self, style: torch.Tensor, spk_cond: torch.Tensor, emo_vector: list[float] | None = None) -> torch.Tensor:
        """Build the emo_vec [1,1280] for GPT conditioning.

        Full official formula (infer_v2_5.py:757-764):
            emovec_audio = merge_emovec(spk, spk, alpha=1.0)  # conformer path
            emovec = emovec_mat + (1 - sum(normalize(w))) * emovec_audio

        The (1-sum)*emovec_audio correction was previously omitted (caused
        'brick' output when a single emotion weight was high — without the
        spk-base emovec term, the conditioning lost the speaker's base
        characteristics). Now restored via the conformer path.
        """
        if emo_vector is None:
            emo_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]  # calm default
        emo_vec_raw = torch.tensor(emo_vector, device=self.device, dtype=torch.float32)
        spk_chunks = tuple(torch.split(self.spk_matrix, self.emo_num))
        emo_chunks = tuple(torch.split(self.emo_matrix, self.emo_num))
        emovec_mat_raw = self.gpt.emo_matrix_lookup(
            style, emo_vec_raw, spk_chunks, emo_chunks
        )  # [1,1280] (raw weights, no normalize; NOT through emo_layer)
        # conformer path: emovec_audio from spk's own audio (alpha=1, pure spk)
        gpt_dtype = self.gpt.emovec_layer.weight.dtype
        emovec_audio = self.gpt.get_emovec(spk_cond.to(gpt_dtype))  # [1,1280] (through emo_layer)
        # official line 769: complement = 1 - sum(RAW weight_vector), clamped >= 0
        complement = float((1.0 - emo_vec_raw.sum()).clamp(min=0.0))
        # emovec_mat is used RAW (no emo_layer) — matches official: the matrix
        # (feat2.pt) is already in target space; only emovec_audio (conformer)
        # passes through emovec_layer+emo_layer inside get_emovec.
        return emovec_mat_raw.to(gpt_dtype) + complement * emovec_audio

    @torch.no_grad()
    def build_emo_vec_full(
        self,
        style: torch.Tensor,
        spk_cond: torch.Tensor,
        emo_vector: list[float] | None,
        emo_ref_path: str | None,
        emo_alpha: float,
    ) -> torch.Tensor:
        """Unified emo_vec [1,1280] assembly covering all three control modes.

        Mirrors infer_v2_5.py:672-778 priority: emo_ref_path (conformer) >
        emo_vector (matrix) > calm default.

        - emo_ref_path given: emovec_audio = merge_emovec(spk, emo_ref, alpha)
          via the 160M conformer. If emo_vector also given, add the matrix
          correction: emovec_mat + (1-sum(w))*emovec_audio (official line 764).
          Otherwise emovec_audio alone is the final emo_vec.
        - emo_ref_path None: pure matrix path (build_emo_vec).
        """
        if emo_ref_path is None:
            return self.build_emo_vec(style, spk_cond, emo_vector)

        # --- conformer path: extract emo_cond_emb from the reference audio ---
        import torchaudio as _ta
        emo_audio, emo_sr = self._load_audio(emo_ref_path, REF_MAX_SECONDS)
        emo_16k = _ta.transforms.Resample(emo_sr, REF_SR_W2V)(emo_audio)
        emo_cond = self.extract_spk_cond(emo_16k)  # [1,T,1024]

        # cast to GPT dtype (conformer runs in fp16 on the fast path)
        gpt_dtype = self.gpt.emovec_layer.weight.dtype
        emovec_audio = self.gpt.merge_emovec(
            spk_cond.to(gpt_dtype), emo_cond.to(gpt_dtype), alpha=emo_alpha
        )  # [1,1280] (already through emo_layer inside get_emovec)

        if emo_vector is None:
            return emovec_audio

        # matrix correction: emovec_mat + (1 - sum(weights)) * emovec_audio
        emo_vec_raw = torch.tensor(emo_vector, device=self.device, dtype=torch.float32)
        spk_chunks = tuple(torch.split(self.spk_matrix, self.emo_num))
        emo_chunks = tuple(torch.split(self.emo_matrix, self.emo_num))
        emovec_mat = self.gpt.emo_matrix_lookup(
            style, emo_vec_raw, spk_chunks, emo_chunks
        )  # [1,1280] (raw weights, no normalize)
        # official line 769: complement = 1 - sum(RAW weight_vector), clamped >= 0
        complement = float((1.0 - emo_vec_raw.sum()).clamp(min=0.0))
        # emovec_mat used RAW (no emo_layer) — matches official.
        return emovec_mat.to(gpt_dtype) + complement * emovec_audio

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        spk_audio_prompt: str,
        text: str,
        lang: str = "ZH",
        emo_vector: list[float] | None = None,
        emo_text: str | None = None,
        emo_ref_path: str | None = None,
        emo_alpha: float = 1.0,
        duration_factor: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        max_mel_tokens: int | None = None,
        cfm_steps: int = 12,
        cfg_rate: float = 0.7,
        teacache_thresh: float = 0.25,
        text_normalization: bool = True,
        max_text_tokens_per_segment: int = 120,
        interval_silence_ms: int = 200,
        repetition_penalty: float = 10.0,
        num_beams: int = 3,
    ) -> tuple[int, torch.Tensor]:
        """Zero-shot voice cloning.

        Args:
            spk_audio_prompt: path to reference audio (any sr).
            text: text to synthesize (raw user text; normalized if
                text_normalization=True).
            lang: ZH / EN / JA / ...
            emo_vector: 8-dim emotion weights [happy,angry,sad,afraid,
                disgusted,melancholic,surprised,calm], or None for calm.
            emo_text: free-text emotion description (e.g. "很开心的语气").
                If given, overrides emo_vector via QwenEmotion prediction.
            duration_factor: scales target length (1.72 * factor).
            text_normalization: run G2P text normalization (digits→words,
                punctuation, etc.) before tokenization.
            max_text_tokens_per_segment: split long text into segments of at
                most this many tokens; segments are synthesized separately and
                concatenated with interval_silence_ms of silence between them.
            max_mel_tokens: max mel codes GPT may emit per segment. If None
                (recommended), auto-derived as max_text_tokens_per_segment * 6
                (text→mel ratio ~5.5x measured) so the limits stay coupled and a
                segment is never truncated. Set explicitly to override.
            repetition_penalty: HF repetition-penalty scale (official 10.0).
            num_beams: GPT-AR beam width (official 3). Beam search now runs
                through the CUDA-Graph decode path too (static batch K, fixed
                KV buffers — no beam removal/reordering in the graph loop), so
                the official quality configuration is no longer slow.
        Returns:
            (sample_rate, audio [samples,]) at 22050 Hz mono.
        """
        dev = self.device

        # --- auto-couple max_mel_tokens to max_text_tokens_per_segment ---
        # text→mel ratio is language-dependent (measured, beam3 do_sample):
        #   ZH ~5.5x, JA ~5.2x, EN ~6.7-10.6x (English BPE tokens carry more
        #   phonetic content per token). Use a language-specific multiplier
        # so a segment is never truncated regardless of language.
        _MEL_RATIO = {"ZH": 6, "JA": 6, "EN": 11, "KO": 8, "YUE": 6}
        if max_mel_tokens is None:
            max_mel_tokens = max_text_tokens_per_segment * _MEL_RATIO.get(lang.upper(), 12)

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

        if len(segments) == 1:
            return self._infer_single(
                spk_audio_prompt, segments[0], lang, emo_vector,
                emo_ref_path, emo_alpha,
                duration_factor, do_sample, top_p, top_k, temperature,
                max_mel_tokens, cfm_steps, cfg_rate, teacache_thresh,
                repetition_penalty, num_beams,
            )

        # multi-segment: synthesize each, join with silence
        wavs = []
        for seg in segments:
            sr, wav = self._infer_single(
                spk_audio_prompt, seg, lang, emo_vector,
                emo_ref_path, emo_alpha,
                duration_factor, do_sample, top_p, top_k, temperature,
                max_mel_tokens, cfm_steps, cfg_rate, teacache_thresh,
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
        teacache_thresh: float,
        repetition_penalty: float = 10.0,
        num_beams: int = 3,
    ) -> tuple[int, torch.Tensor]:
        """Synthesize a single (already normalized, short) text segment."""
        dev = self.device
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

        # --- S2Mel-CFM → mel (TeaCache: skip redundant DiT steps) ---
        est = self.s2mel.cfm.estimator
        if teacache_thresh > 0 and not getattr(est, "teacache_enabled", False):
            est.enable_teacache(thresh=teacache_thresh)
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
