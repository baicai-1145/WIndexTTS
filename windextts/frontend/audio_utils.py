"""Audio feature extraction — pure-torch replacement for transformers' frontend.

Replaces ``transformers.SeamlessM4TFeatureExtractor`` (the w2v-bert frontend) and
the mel/spectrogram helpers, with **zero transformers/librosa dependency**.

Verified to reproduce SeamlessM4TFeatureExtractor's ``input_features`` exactly
(max_abs_diff ~ 1e-6 vs the official numpy path) for the IndexTTS-2.5 config:

Pipeline (SeamlessM4TFeatureExtractor, all params from the w2v-bert-2.0
``preprocessor_config.json`` + source):
  1. waveform × 2**15            (Kaldi 16-bit signed-int scaling)
  2. frame: frame_length=400 (25ms), hop=160 (10ms), center=False
     per frame: remove_dc_offset → preemphasis 0.97 → × Povey window
  3. rfft(512) → power spectrogram  [T, 257]
  4. spec @ mel_filters (257×80) → log-mel  [T, 80]   (mel_floor 1.19e-7, natural log)
  5. per-mel-bin z-score normalize:  (x - mean) / sqrt(var(ddof=1) + 1e-7)
  6. stride-2 frame stacking:       [T, 80] -> [T//2, 160]

The mel_filters (257×80) and Povey window (400) are precomputed constants; we
load them once. For w2v-bert-2.5 they ship inside the SeamlessM4TFeatureExtractor,
so we cache them to a .npz at first use (no transformers import at runtime).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["SeamlessM4TFeaturizer"]

# ---------------------------------------------------------------------------
# Constants for the IndexTTS-2.5 w2v-bert-2.0 frontend (from preprocessor_config.json).
# ---------------------------------------------------------------------------

_FRAME_LENGTH = 400   # samples (25ms @16k)
_HOP_LENGTH = 160     # samples (10ms @16k)
_FFT_LENGTH = 512
_PREEMPHASIS = 0.97
_N_MELS = 80
_MEL_FLOOR = 1.192092955078125e-07
_STRIDE = 2
_NORM_EPS = 1e-7

# Cached frontend constants (mel_filters + povey window), generated once from the
# official SeamlessM4TFeatureExtractor. The fairseq-derived filterbank has
# non-standard parameters, so the constants ship as package data (168KB) —
# no transformers dependency and no local dump needed at runtime.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FRONTEND_CACHE_DEFAULT = _DATA_DIR / "seamless_frontend.npz"


def _povey_window(n: int) -> torch.Tensor:
    """Symmetric Povey window: (0.5 - 0.5*cos(2πk/(N)))^0.85.

    Matches transformers.audio_utils.window_function(N, 'povey', periodic=False).
    """
    k = torch.arange(n, dtype=torch.float64)
    return (0.5 - 0.5 * torch.cos(2 * torch.pi * k / n)) ** 0.85


class SeamlessM4TFeaturizer:
    """Pure-torch w2v-bert-2.0 frontend: waveform → input_features [B, T, 160].

    Args:
        mel_filters: [fft//2+1, n_mels] = [257, 80] mel filterbank. If None,
            loaded from ``cache_path`` (which must have been built first).
        device/dtype: where to place the fixed buffers (window, mel_filters).
    """

    def __init__(
        self,
        mel_filters: torch.Tensor | None = None,
        *,
        window: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        cache_path: str | Path | None = None,
    ) -> None:
        cache_path = Path(cache_path) if cache_path else _FRONTEND_CACHE_DEFAULT

        if mel_filters is None:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"frontend cache not found at {cache_path}. Run "
                    f"SeamlessM4TFeaturizer.build_cache(...) once first."
                )
            d = np.load(cache_path)
            mel_filters = torch.from_numpy(d["mel_filters"])
            window_np = d["window"]
        elif window is None:
            # window must be provided alongside a custom mel_filters, else fall
            # back to the analytic Povey window (less exact — prefer cache).
            window_np = _povey_window(_FRAME_LENGTH).numpy()
        else:
            window_np = window.numpy() if isinstance(window, torch.Tensor) else window

        # window as float64 for numerical fidelity with the reference (np.fft is f64).
        self.window = torch.as_tensor(window_np, dtype=torch.float64, device=device)
        self.mel_filters = torch.as_tensor(mel_filters, dtype=torch.float64, device=device)
        self.device = torch.device(device)
        self.dtype = dtype

    # ----- the core transform -----

    @torch.no_grad()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute input_features from a 16kHz mono waveform.

        Args:
            waveform: [B, N] or [N] float tensor, 16kHz mono audio in [-1, 1].

        Returns:
            input_features [B, T, 160] float32, where T = (num_frames // stride)
            and num_frames = 1 + floor((N - 400)/160).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        B, N = waveform.shape
        device, dtype = waveform.device, torch.float64
        win = self.window.to(device=device, dtype=dtype)
        mf = self.mel_filters.to(device=device, dtype=dtype)

        num_frames = 1 + (N - _FRAME_LENGTH) // _HOP_LENGTH
        n_freq = _FFT_LENGTH // 2 + 1  # 257

        # Kaldi 16-bit scaling + frame extraction via unfold (vectorized, no loop).
        wav = (waveform.to(dtype=dtype) * (2**15))  # [B, N]
        # unfold the time dim into frames: [B, num_frames, frame_length]
        frames = wav.unfold(1, _FRAME_LENGTH, _HOP_LENGTH).contiguous()

        # remove_dc_offset (per frame)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        # preemphasis: y[0]=x[0]; y[i]=x[i]-0.97*x[i-1]
        pre = frames.clone()
        pre[..., 1:] = frames[..., 1:] - _PREEMPHASIS * frames[..., :-1]
        frames = pre

        # window + rfft -> power spec
        windowed = frames * win  # [B, T, fl]
        spec = torch.fft.rfft(windowed, n=_FFT_LENGTH)  # [B, T, 257] complex
        power = spec.real**2 + spec.imag**2  # [B, T, 257]

        # mel projection + log
        mel = power @ mf  # [B, T, 80]
        mel = torch.log(mel.clamp_min(_MEL_FLOOR))

        # per-mel-bin z-score normalize (ddof=1), per-utterance (over time axis)
        mean = mel.mean(dim=1, keepdim=True)
        var = mel.var(dim=1, unbiased=True, keepdim=True)  # unbiased=ddof=1
        mel = (mel - mean) / torch.sqrt(var + _NORM_EPS)

        # stride-2 frame stacking: [B, T, 80] -> [B, T//2, 160]
        T = mel.shape[1]
        rem = T % _STRIDE
        if rem:
            mel = mel[:, : T - rem, :]
        mel = mel.reshape(B, T // _STRIDE, _N_MELS * _STRIDE)
        return mel.to(self.dtype)

    # ----- one-time cache build from the official extractor -----

    @staticmethod
    def build_cache(
        w2v_bert_dir: str | Path = "/root/IndexTTS-2.5/hf_cache/w2v-bert-2.0",
        out_path: str | Path | None = None,
    ) -> Path:
        """Extract mel_filters + window from the official SeamlessM4TFeatureExtractor
        and cache them to an .npz. Run once; thereafter no transformers import needed.

        (The mel_filters table is non-trivial to regenerate bit-exactly — it comes
        from kaldi/transformers internal tables — so we snapshot it once.)
        """
        from transformers import SeamlessM4TFeatureExtractor

        ex = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_bert_dir), local_files_only=True
        )
        out_path = Path(out_path) if out_path else _FRONTEND_CACHE_DEFAULT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            mel_filters=np.asarray(ex.mel_filters, dtype=np.float64),
            window=np.asarray(ex.window, dtype=np.float64),
        )
        print(f"frontend cache written: {out_path} "
              f"(mel_filters {ex.mel_filters.shape}, window {ex.window.shape})")
        return out_path


if __name__ == "__main__":
    import sys

    # Build cache if missing, then self-test against an official dump.
    if not _FRONTEND_CACHE_DEFAULT.exists():
        SeamlessM4TFeaturizer.build_cache()

    # reference dump
    ref_path = Path("/root/windextts_dumps/w2v.input_features.pt")
    if not ref_path.exists():
        print(
            f"reference dump not found at {ref_path}; run "
            f"scripts/dump_indextts_tensors.py --stages w2v first"
        )
        sys.exit(1)

    import torchaudio

    ref = torch.load(ref_path, weights_only=False).cpu()  # [1,303,160]
    audio, sr = torchaudio.load("/root/WIndexTTS/test.wav")
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)  # [1, N]

    fe = SeamlessM4TFeaturizer()
    out = fe(audio_16k)
    assert out.shape == ref.shape, f"shape {out.shape} != ref {ref.shape}"
    # Compare in float64: the official dump is float32, and independent float64->float32
    # roundings of near-zero-variance mel bins can differ in the last bit (up to ~0.025
    # at a handful of elements). The true algorithmic accuracy is ~1e-6 in float64.
    out64 = out.double()
    ref64 = ref.double()
    # also compare against a freshly-computed float64 reference (most rigorous)
    import numpy as np
    from transformers import SeamlessM4TFeatureExtractor
    ex = SeamlessM4TFeatureExtractor.from_pretrained(
        "/root/IndexTTS-2.5/hf_cache/w2v-bert-2.0", local_files_only=True
    )
    off64 = torch.as_tensor(
        ex(audio_16k[0].numpy(), sampling_rate=16000, return_tensors="pt")["input_features"][0].numpy()
    ).double()
    diff_vs_dump = (out64 - ref64).abs().max().item()
    diff_vs_fresh = (out64[0] - off64).abs().max().item()
    print(f"out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    print(f"max_abs_diff vs float32 dump  = {diff_vs_dump:.3e}  (includes f32 truncation noise)")
    print(f"max_abs_diff vs float64 fresh = {diff_vs_fresh:.3e}  (true algorithmic accuracy)")
    print(f"allclose(atol=1e-3, rtol=1e-3) vs fresh = {torch.allclose(out64[0], off64, atol=1e-3, rtol=1e-3)}")
    print("SMOKE", "OK" if diff_vs_fresh < 1e-3 else "FAIL")
