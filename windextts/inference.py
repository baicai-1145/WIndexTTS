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
        self.gpt.load_official(w.load_gpt())
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

        print(f">> WIndexTTS loaded all modules on {dev}")
        # apply per-module precision overrides (mixed-precision fast paths)
        # GPT-AR decode is matmul-bound: fp16 gives ~1.8x with 100% greedy match.
        # S2Mel/BigVGAN stay fp32 (bf16/fp16 degrade quality or weren't validated).
        if self.dtype == torch.float16:
            self.gpt.to(torch.float16)
            print(">> GPT-AR cast to fp16 (mixed precision)")

        # ref-audio feature cache: keyed by (path, mtime) → avoids recomputing
        # w2v/campplus/mel when the same ref is reused across requests (stage 5).
        self._ref_cache: dict = {}

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

    def build_emo_vec(self, style: torch.Tensor, emo_vector: list[float] | None = None) -> torch.Tensor:
        """Build the emo_vec [1,1280] for GPT conditioning via emo_matrix_lookup.

        Uses the matrix path (emovec_mat only). The (1-sum)*emovec(audio) correction
        term from infer_v2_5.py:764 requires the merge_emovec conformer (not yet
        ported); omitted here as a first-version simplification (dominant term is
        emovec_mat when weights are concentrated).
        """
        if emo_vector is None:
            emo_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]  # calm default
        emo_vec_raw = torch.tensor(emo_vector, device=self.device, dtype=torch.float32)
        spk_chunks = tuple(torch.split(self.spk_matrix, self.emo_num))
        emo_chunks = tuple(torch.split(self.emo_matrix, self.emo_num))
        emovec_mat = self.gpt.emo_matrix_lookup(
            style, emo_vec_raw, spk_chunks, emo_chunks
        )  # [1,1280] (normalize_emo_vec applied inside)
        # final emo_layer projection; cast to GPT dtype (fp16 mixed-precision path)
        return self.gpt.emo_layer(emovec_mat.to(self.gpt.emo_layer.weight.dtype))

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
        duration_factor: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        max_mel_tokens: int = 1000,
        cfm_steps: int = 25,
        cfg_rate: float = 0.7,
        teacache_thresh: float = 0.15,
    ) -> tuple[int, torch.Tensor]:
        """Zero-shot voice cloning.

        Args:
            spk_audio_prompt: path to reference audio (any sr).
            text: text to synthesize (already normalized; complex G2P is TODO).
            lang: ZH / EN / JA / ...
            emo_vector: 8-dim emotion weights, or None for calm.
            duration_factor: scales target length (1.72 * factor).
        Returns:
            (sample_rate, audio [samples,]) at 22050 Hz mono.
        """
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

        # --- emo vec (matrix path; audio-term omitted, see build_emo_vec) ---
        emo_vec = self.build_emo_vec(style, emo_vector)  # [1,1280]

        # --- GPT conditioning + AR decode ---
        conds_latent = self.gpt.build_conds_latent(style, emo_vec)  # [1,3,1280]
        use_cg = self.device != "cpu"
        codes = self.gpt.generate(
            conds_latent, text_tokens, lang_id,
            max_new_tokens=max_mel_tokens, do_sample=do_sample,
            top_k=top_k, top_p=top_p, temperature=temperature,
            stop_token=self.cfg.gpt.stop_mel_token,
            use_cuda_graph=use_cg,
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
        )  # [1, 80, T_target]

        # --- BigVGAN → audio ---
        audio_out = self.bigvgan(mel)  # [1, 1, T_audio]
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
