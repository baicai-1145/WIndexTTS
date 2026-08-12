"""Numerical alignment test: our pure-torch BigVGAN vs official index-tts output.

Compares against the dumped official vocoder output:
    /root/windextts_dumps/bigvgan.input_mel.pt   [1,80,523]  (input)
    /root/windextts_dumps/bigvgan.output_wav.pt  [1,1,133888] (official fp32 CUDA output)

Runs on CUDA fp32. BigVGAN is a deep conv network; we expect ~1e-4..1e-3 max
abs diff in fp32.
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"


def test_bigvgan_alignment():
    from windextts.models.bigvgan import BigVGAN, BigVGANConfig
    from windextts.weights import WeightLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = BigVGAN(BigVGANConfig())
    m.load_official(WeightLoader().load_bigvgan())
    m = m.to(dev).eval()

    ref_in = torch.load(f"{DUMPS}/bigvgan.input_mel.pt", weights_only=False).to(dev)
    ref_out = torch.load(f"{DUMPS}/bigvgan.output_wav.pt", weights_only=False).to(dev)

    with torch.no_grad():
        out = m(ref_in)

    assert out.shape == ref_out.shape, f"shape {out.shape} != ref {ref_out.shape}"
    diff = (out.float() - ref_out.float()).abs().max().item()
    mean_diff = (out.float() - ref_out.float()).abs().mean().item()
    rel = diff / (ref_out.float().abs().max().item() + 1e-9)
    print(f"\n[align] out {tuple(out.shape)}")
    print(f"[align] max_abs_diff={diff:.3e}  mean_abs_diff={mean_diff:.3e}  rel={rel:.3e}")
    print(f"[align] allclose(atol=1e-3, rtol=1e-3) = "
          f"{torch.allclose(out.float(), ref_out.float(), atol=1e-3, rtol=1e-3)}")
    assert torch.allclose(out.float(), ref_out.float(), atol=1e-3, rtol=1e-3), (
        f"BigVGAN output not aligned: max_abs_diff {diff}"
    )
    print("BIGVGAN ALIGN OK")


if __name__ == "__main__":
    test_bigvgan_alignment()
