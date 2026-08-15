"""HiFiGAN mel frontend (S2Mel prompt-mel path), zero-librosa.

Contract (s2mel/modules/audio.py:45-80): reflect-pad (n_fft-hop)//2 each side,
|STFT(hann)| = sqrt(re^2+im^2+1e-9), mel = mel_basis @ spec, log(clamp(.,1e-5)).
spect_params: n_fft=1024, hop=256, win=1024, n_mels=80, sr=22050, fmin=0,
center=False. The [80,513] filterbank is snapshotted once (librosa slaney params,
verified bit-identical to the official cached basis) so librosa is never
imported at runtime.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

_DATA = Path(__file__).resolve().parent.parent / "data" / "mel_basis_hifigan.pt"


class MelSpectrogram:
    def __init__(self, mel_basis=None, *, device="cpu", cache_path=None):
        # mel_basis [80,513] filterbank; None -> load cache (build once first)
        if mel_basis is None:
            p = Path(cache_path) if cache_path else _DATA
            if not p.exists():
                raise FileNotFoundError(
                    f"mel_basis cache not found at {p}. Run build_mel_basis_cache() once first."
                )
            mel_basis = torch.load(p, weights_only=False)
        self.mel_basis = mel_basis.to(device)
        self.hann_window = torch.hann_window(1024).to(device)  # win=1024

    @torch.no_grad()
    def __call__(self, y):  # y [B,T] @22k (or [T]) -> log-mel [B,80,T_frames]
        if y.dim() == 1:
            y = y.unsqueeze(0)
        pad = (1024 - 256) // 2  # reflect pad (n_fft-hop)//2 each side
        spec = torch.stft(
            F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1),
            1024, hop_length=256, win_length=1024,
            window=self.hann_window.to(y.device), center=False,
            pad_mode="reflect", normalized=False, onesided=True,
            return_complex=True,
        )
        # spectral_normalize_torch: log(clamp(mel, 1e-5))
        return torch.log(torch.clamp(
            self.mel_basis.to(y.device) @ torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9),
            min=1e-5))


def build_mel_basis_cache(out_path=None):
    """Snapshot the librosa mel_basis once (never import librosa at runtime)."""
    out = Path(out_path) if out_path else _DATA
    out.parent.mkdir(parents=True, exist_ok=True)
    # grab the official module's cached global (librosa-generated); trigger its
    # population for our (sr, fmax) by running one mel_spectrogram call
    from indextts.s2mel.modules.audio import mel_basis as mb, mel_spectrogram
    import torchaudio  # noqa: F401
    mel_spectrogram(
        torch.zeros(1, 22050), n_fft=1024, num_mels=80, sampling_rate=22050,
        hop_size=256, win_size=1024, fmin=0.0, fmax=None, center=False,
    )
    basis = mb["22050_None_cpu"]  # key format "{sr}_{fmax}_{device}", device=cpu
    torch.save(basis.cpu(), out)
    print(f"mel_basis cache written: {out} {tuple(basis.shape)}")
    return out


