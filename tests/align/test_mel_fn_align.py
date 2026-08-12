"""Alignment test: our MelSpectrogram (HiFiGAN mel) vs official mel_fn."""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"


def test_mel_fn_alignment():
    from windextts.frontend.mel import MelSpectrogram

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mel_fn = MelSpectrogram(device=dev)
    audio22 = torch.load(f"{DUMPS}/frontend.audio_22k.pt", weights_only=False).to(dev)
    ref = torch.load(f"{DUMPS}/frontend.mel_fn_output.pt", weights_only=False)

    out = mel_fn(audio22)
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    diff = (out.float().cpu() - ref.float()).abs().max().item()
    print(f"\n[align] mel_fn {tuple(out.shape)} max_diff={diff:.3e}")
    assert torch.allclose(out.float().cpu(), ref.float(), atol=1e-4, rtol=1e-3), f"diff {diff}"


if __name__ == "__main__":
    test_mel_fn_alignment()
    print("MEL_FN ALIGN OK")
