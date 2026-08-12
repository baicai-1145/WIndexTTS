"""Numerical alignment test: our S2Mel DiT estimator vs official output.

Compares our pure-torch DiT (windextts/models/s2mel_dit.py) against the official
index-tts CFM estimator on a fixed-timestep, fixed-noise single forward step
(the deterministic dphi_dt the Euler solver uses each step).

Reference dumps (from the official fp32 CUDA run, /root/windextts_dumps/):
- s2mel.dit_input_x.pt        [1, 80, 1045]  flow state (prompt region zeroed)
- s2mel.dit_input_prompt_x.pt [1, 80, 1045]  reference mel
- s2mel.dit_input_cond.pt     [1, 1045, 512] length_regulator output
- s2mel.dit_input_style.pt    [1, 192]
- s2mel.dit_input_t.pt        [1]            t = 0.5
- s2mel.dit_output.pt         [1, 80, 1045]  official dphi_dt
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"
S2MEL_PTH = "/root/IndexTTS-2.5/s2mel.pth"


def test_s2mel_dit_alignment():
    import torch

    from windextts.models.s2mel_dit import DiT

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load(S2MEL_PTH, map_location="cpu", weights_only=False)["net"]

    model = DiT().to(dev).eval()
    model.load_official(net["cfm"])
    model.setup_caches(1, 2048)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[align] DiT params: {n_params/1e6:.1f}M, strict load OK")

    x = torch.load(f"{DUMPS}/s2mel.dit_input_x.pt", weights_only=False).to(dev)
    prompt_x = torch.load(f"{DUMPS}/s2mel.dit_input_prompt_x.pt", weights_only=False).to(dev)
    cond = torch.load(f"{DUMPS}/s2mel.dit_input_cond.pt", weights_only=False).to(dev)
    style = torch.load(f"{DUMPS}/s2mel.dit_input_style.pt", weights_only=False).to(dev)
    t = torch.load(f"{DUMPS}/s2mel.dit_input_t.pt", weights_only=False).to(dev)
    ref = torch.load(f"{DUMPS}/s2mel.dit_output.pt", weights_only=False).to(dev)
    x_lens = torch.LongTensor([cond.size(1)]).to(dev)

    with torch.no_grad():
        out = model(x, prompt_x, x_lens, t, style, cond)

    assert out.shape == ref.shape, f"shape {tuple(out.shape)} != ref {tuple(ref.shape)}"
    diff = (out.float() - ref.float()).abs().max().item()
    rel = diff / (ref.float().abs().max().item() + 1e-12)
    print(f"[align] out {tuple(out.shape)}")
    print(f"[align] max_abs_diff={diff:.3e}  rel={rel:.3e}  (ref|max|={ref.float().abs().max().item():.3f})")
    print(f"[align] allclose(atol=1e-3, rtol=1e-3) = "
          f"{torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3)}")

    # Project standard for neural modules.
    assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3), (
        f"S2Mel DiT estimator not aligned: max_abs_diff={diff}"
    )


if __name__ == "__main__":
    test_s2mel_dit_alignment()
    print("S2MEL-DIT ALIGN OK")
