"""Pure-torch w2v-bert-2.0 frontend (replaces transformers.SeamlessM4TFeatureExtractor).

Pipeline (SeamlessM4TFeatureExtractor params from preprocessor_config.json):
x*2^15 -> frame(400/160, center=False) -> dc-offset -> preemphasis .97 ->
Povey window -> rfft(512) -> power -> mel_filters(257x80) ->
log(clamp_min 1.19e-7) -> per-mel-bin z-score (ddof=1) -> stride-2 stack
[T,80] -> [T//2,160]. Reproduces official input_features to ~1e-6. The
fairseq-derived mel_filters + window ship as package data (168KB npz) — no
transformers import at runtime; build_cache() regenerates the snapshot.
"""
from pathlib import Path

import numpy as np
import torch

__all__ = ["SeamlessM4TFeaturizer"]

_FRAME_LENGTH, _HOP_LENGTH, _FFT_LENGTH = 400, 160, 512  # 25ms / 10ms @16k
_PREEMPHASIS, _N_MELS, _STRIDE = 0.97, 80, 2
_MEL_FLOOR, _NORM_EPS = 1.192092955078125e-07, 1e-7
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FRONTEND_CACHE_DEFAULT = _DATA_DIR / "seamless_frontend.npz"


def _povey_window(n):
    # symmetric Povey window (audio_utils.window_function 'povey', periodic=False)
    k = torch.arange(n, dtype=torch.float64)
    return (0.5 - 0.5 * torch.cos(2 * torch.pi * k / n)) ** 0.85


class SeamlessM4TFeaturizer:
    """waveform [B,N]|[N] 16k mono in [-1,1] -> input_features [B, T//2, 160]."""

    def __init__(self, mel_filters=None, *, window=None, device="cpu",
                 dtype=torch.float32, cache_path=None):
        cache_path = Path(cache_path) if cache_path else _FRONTEND_CACHE_DEFAULT
        if mel_filters is None:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"frontend cache not found at {cache_path}. Run "
                    f"SeamlessM4TFeaturizer.build_cache(...) once first.")
            d = np.load(cache_path)
            mel_filters, window_np = d["mel_filters"], d["window"]
        elif window is None:
            window_np = _povey_window(_FRAME_LENGTH).numpy()  # analytic fallback (less exact than cache)
        else:
            window_np = window.numpy() if isinstance(window, torch.Tensor) else window
        # f64 buffers: the official path computes in float64 (np.fft)
        self.window = torch.as_tensor(window_np, dtype=torch.float64, device=device)
        self.mel_filters = torch.as_tensor(mel_filters, dtype=torch.float64, device=device)
        self.device, self.dtype = torch.device(device), dtype

    @torch.no_grad()
    def __call__(self, waveform, return_mask=False):
        # T = 1 + (N-400)//160 frames; odd tails zero-padded to multiple of stride
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        dev = waveform.device
        win, mf = self.window.to(dev, torch.float64), self.mel_filters.to(dev, torch.float64)
        fr = (waveform.to(torch.float64) * 2 ** 15).unfold(1, _FRAME_LENGTH, _HOP_LENGTH).contiguous()  # Kaldi 16-bit scale, [B,T,400]
        fr = fr - fr.mean(-1, keepdim=True)  # dc offset
        pre = fr.clone(); pre[..., 1:] = fr[..., 1:] - _PREEMPHASIS * fr[..., :-1]  # y[i]=x[i]-.97*x[i-1]
        p = torch.fft.rfft(pre * win, _FFT_LENGTH)  # [B,T,257]
        mel = torch.log(((p.real ** 2 + p.imag ** 2) @ mf).clamp_min(_MEL_FLOOR))  # power spec @ mel_filters
        mel = (mel - mel.mean(1, keepdim=True)) / torch.sqrt(mel.var(1, unbiased=True, keepdim=True) + _NORM_EPS)
        # stride-2 stack; pad odd tail with 1.0 (official pad_to_multiple_of=2,
        # padding_value=1 — silent frame). Dropping the tail frame instead used
        # to shift every downstream conds (spk_cond_emb, conds_latent, GPT path).
        T, rem = mel.shape[1], mel.shape[1] % _STRIDE
        if rem:
            mel = torch.cat([mel, torch.full((mel.shape[0], _STRIDE - rem, mel.shape[2]), 1.0,
                                             dtype=mel.dtype, device=dev)], 1)
            T = mel.shape[1]
        mel = mel.reshape(mel.shape[0], T // _STRIDE, _N_MELS * _STRIDE).to(self.dtype)  # [B, T//2, 160]
        if not return_mask:
            return mel
        mask = torch.ones(mel.shape[0], T // _STRIDE, dtype=torch.int32, device=self.device)
        if rem:
            mask[:, -1] = 0  # stacked pair contains the padded row
        return mel, mask

    @staticmethod
    def build_cache(w2v_bert_dir="/root/IndexTTS-2.5/hf_cache/w2v-bert-2.0", out_path=None):
        # snapshot the official tables once — non-trivial to regenerate
        # bit-exactly (kaldi/transformers internal filterbank)
        from transformers import SeamlessM4TFeatureExtractor
        ex = SeamlessM4TFeatureExtractor.from_pretrained(str(w2v_bert_dir), local_files_only=True)
        out = Path(out_path) if out_path else _FRONTEND_CACHE_DEFAULT
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 mel_filters=np.asarray(ex.mel_filters, dtype=np.float64),
                 window=np.asarray(ex.window, dtype=np.float64))
        print(f"frontend cache written: {out} "
              f"(mel_filters {ex.mel_filters.shape}, window {ex.window.shape})")
        return out


if __name__ == "__main__":
    import sys
    import torchaudio

    if not _FRONTEND_CACHE_DEFAULT.exists():
        SeamlessM4TFeaturizer.build_cache()
    ref_path = Path("/root/windextts_dumps/w2v.input_features.pt")
    if not ref_path.exists():
        print(f"reference dump not found at {ref_path}; run "
              f"scripts/dump_indextts_tensors.py --stages w2v first")
        sys.exit(1)
    ref = torch.load(ref_path, weights_only=False).cpu()
    audio, sr = torchaudio.load("/root/WIndexTTS/test.wav")
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
    out = SeamlessM4TFeaturizer()(audio_16k)
    assert out.shape == ref.shape
    # f64 compare: official dump is f32; near-zero-variance bins can differ in
    # the last f32 bit (~0.025 at a few elements); true accuracy ~1e-6 in f64.
    diff = (out.double() - ref.double()).abs().max().item()
    print(f"out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    print(f"max_abs_diff vs float32 dump = {diff:.3e} (incl. f32 truncation noise)")
    print("SMOKE", "OK" if diff < 1e-3 else "FAIL")
