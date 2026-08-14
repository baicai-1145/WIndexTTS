"""HiFiGAN/BigVGAN-style mel-spectrogram (the S2Mel prompt-mel frontend).

Replaces ``indextts.s2mel.modules.audio.mel_spectrogram`` with zero-librosa
dependency. The mel filterbank (``mel_basis``) is non-trivial to regenerate
bit-exactly from librosa, so we snapshot it once (like the w2v-bert frontend)
and load from cache.

Verified contract (s2mel/modules/audio.py:45-80):
  y = reflect_pad(y, (n_fft-hop)//2 each side)
  spec = |STFT(y, hann_window)|  (magnitude, sqrt(re^2+im^2+1e-9))
  mel = mel_basis @ spec
  mel = log(clamp(mel, 1e-5))     # spectral_normalize_torch

Config (from s2mel.preprocess_params.spect_params): n_fft=1024, hop=256,
win=1024, n_mels=80, sr=22050, fmin=0, fmax=None, center=False.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

__all__ = ["MelSpectrogram", "build_mel_basis_cache"]

# Defaults from config.yaml s2mel.preprocess_params.spect_params
_N_FFT = 1024
_HOP = 256
_WIN = 1024
_N_MELS = 80
_SR = 22050
_FMIN = 0.0
_FMAX = None  # config "None" string -> None
_CENTER = False
_CLAP = 1e-5

# HiFiGAN mel basis (librosa slaney params, verified bit-identical to the
# official indextts cached basis). Ships as package data (166KB) so no
# librosa/indextts is needed at runtime.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_DEFAULT = _DATA_DIR / "mel_basis_hifigan.pt"


class MelSpectrogram:
    """HiFiGAN mel-spectrogram (the S2Mel prompt-mel path).

    Args:
        mel_basis: [n_mels, n_fft//2+1] = [80, 513] filterbank. If None, loaded
            from cache_path (must be built first via build_mel_basis_cache).
    """

    def __init__(
        self,
        mel_basis: torch.Tensor | None = None,
        *,
        device: str | torch.device = "cpu",
        cache_path: str | Path | None = None,
    ) -> None:
        cache_path = Path(cache_path) if cache_path else _CACHE_DEFAULT
        if mel_basis is None:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"mel_basis cache not found at {cache_path}. Run "
                    f"build_mel_basis_cache() once first."
                )
            mel_basis = torch.load(cache_path, weights_only=False)
        self.mel_basis = mel_basis.to(device)
        self.hann_window = torch.hann_window(_WIN).to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        Args:
            y: [B, T] float audio at 22050 Hz (or [T] -> unsqueezed).
        Returns:
            mel [B, n_mels, T_frames] log-mel.
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
        device = y.device
        # reflect pad (n_fft-hop)//2 each side (matches official padding)
        pad = (_N_FFT - _HOP) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

        spec = torch.stft(
            y,
            _N_FFT,
            hop_length=_HOP,
            win_length=_WIN,
            window=self.hann_window.to(device),
            center=_CENTER,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # magnitude (sqrt(re^2 + im^2 + 1e-9))
        spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9)
        mel = self.mel_basis.to(device) @ spec  # [B, n_mels, T]
        # spectral_normalize: log(clamp(mel, 1e-5))
        mel = torch.log(torch.clamp(mel, min=_CLAP))
        return mel


def build_mel_basis_cache(
    out_path: str | Path | None = None,
) -> Path:
    """Snapshot the librosa mel_basis once (so we never import librosa at runtime).

    Reads it from the official indextts audio module's cached global.
    """
    out_path = Path(out_path) if out_path else _CACHE_DEFAULT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Import the official module just to grab its cached mel_basis (librosa-generated).
    from indextts.s2mel.modules.audio import mel_basis as official_mb  # noqa: WPS433

    # ensure it's populated for our (sr, fmax) by running once
    import torchaudio  # noqa: F401, WPS433
    import torch as _t

    # trigger population
    from indextts.s2mel.modules.audio import mel_spectrogram  # noqa: WPS433

    mel_spectrogram(
        _t.zeros(1, _SR), n_fft=_N_FFT, num_mels=_N_MELS, sampling_rate=_SR,
        hop_size=_HOP, win_size=_WIN, fmin=_FMIN, fmax=_FMAX, center=_CENTER,
    )
    # the key format: "{sr}_{fmax}_{device}", device=cpu here
    key = f"{_SR}_{_FMAX}_cpu"
    mb = official_mb[key]
    torch.save(mb.cpu(), out_path)
    print(f"mel_basis cache written: {out_path} {tuple(mb.shape)}")
    return out_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")

    if not _CACHE_DEFAULT.exists():
        build_mel_basis_cache()

    import torchaudio

    mel_fn = MelSpectrogram(device="cuda")
    audio22 = torch.load("/root/windextts_dumps/frontend.audio_22k.pt", weights_only=False).to("cuda")
    ref = torch.load("/root/windextts_dumps/frontend.mel_fn_output.pt", weights_only=False)

    out = mel_fn(audio22)
    assert out.shape == ref.shape, f"shape {out.shape} != ref {ref.shape}"
    diff = (out.float().cpu() - ref.float().cpu()).abs().max().item()
    print(f"mel_fn out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    print(f"max_abs_diff = {diff:.3e}")
    print(f"allclose(atol=1e-4, rtol=1e-3) = {torch.allclose(out.float().cpu(), ref.float(), atol=1e-4, rtol=1e-3)}")
    print("SMOKE", "OK" if diff < 1e-3 else "FAIL")
