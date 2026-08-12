"""Numerical alignment test: S2Mel-CFM end-to-end (length_regulator + DiT + CFM).

Validates the full S2Mel pipeline assembled from our three pure-torch sub-modules:
  InterpolateRegulator + DiT + CFM Euler solver (25 steps, cfg_rate=0.7).

Strategy: CFM output depends on RNG noise, so we compare against a fixed-seed
(seed=123) official dump. With all three sub-modules bit-aligned individually,
the assembled CFM must reproduce the official mel bit-for-bit.
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"


def _build_s2mel(dev):
    from windextts.config import load_default_config
    from windextts.models.length_regulator import InterpolateRegulator
    from windextts.models.s2mel_cfm import S2Mel, S2MelCFM
    from windextts.models.s2mel_dit import DiT
    from windextts.weights import WeightLoader

    cfg = load_default_config()
    w = WeightLoader()
    net = w.load_s2mel()

    lr = InterpolateRegulator(
        channels=cfg.s2mel.length_reg.channels,
        sampling_ratios=cfg.s2mel.length_reg.sampling_ratios,
        is_discrete=cfg.s2mel.length_reg.is_discrete,
        in_channels=cfg.s2mel.length_reg.in_channels,
        codebook_size=cfg.s2mel.length_reg.content_codebook_size,
    )
    lr.load_official(net["length_regulator"])
    lr = lr.to(dev).eval()

    dit = DiT()
    dit.load_official(net["cfm"])
    dit = dit.to(dev).eval()

    cfm = S2MelCFM(dit, in_channels=cfg.s2mel.dit.in_channels).to(dev).eval()
    return S2Mel(lr, cfm).to(dev).eval()


def test_s2mel_end_to_end_alignment():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    s2mel = _build_s2mel(dev)

    spk_cond = torch.load(f"{DUMPS}/gpt.spk_cond_w2v.pt", weights_only=False).to(dev)
    s_infer = torch.load(f"{DUMPS}/s2mel.S_infer.pt", weights_only=False).to(dev)
    ref_mel = torch.load(f"{DUMPS}/s2mel.ref_mel.pt", weights_only=False).to(dev)
    style = torch.load(f"{DUMPS}/s2mel.style.pt", weights_only=False).to(dev)
    ref = torch.load(f"{DUMPS}/s2mel.cfm_output_mel_seed123.pt", weights_only=False).to(dev)

    # Fixed seed (must match the dump's seed=123).
    torch.cuda.manual_seed(123)
    out = s2mel.inference(spk_cond, s_infer, ref_mel, style, duration_factor=1.0)

    assert out.shape == ref.shape, f"shape {out.shape} != ref {ref.shape}"
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"\n[align] S2Mel end-to-end mel {tuple(out.shape)} max_diff={diff:.3e}")
    print(f"[align] allclose(atol=1e-3, rtol=1e-3) = "
          f"{torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3)}")
    assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3), (
        f"S2Mel CFM not aligned: diff {diff}"
    )


if __name__ == "__main__":
    test_s2mel_end_to_end_alignment()
    print("S2MEL END-TO-END ALIGN OK")
