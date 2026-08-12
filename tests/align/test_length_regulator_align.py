"""Numerical alignment test: our InterpolateRegulator vs the official
indextts length_regulator (S2Mel-CFM first stage).

Two continuous-input cases (is_discrete=False path):
  1. S_infer [1,304,1024] -> cond  [1,522,512]   (ylens = int(304*1.72) = 522)
  2. spk_cond_w2v [1,303,1024] -> prompt_condition [1,523,512]  (ylens = 523)

Reference tensors dumped from the official IndexTTS-2.5 run (fp32, CUDA):
  /root/windextts_dumps/s2mel.{S_infer,cond,prompt_condition}.pt
  /root/windextts_dumps/gpt.spk_cond_w2v.pt
"""
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(dev):
    from windextts.models.length_regulator import InterpolateRegulator
    from windextts.weights import WeightLoader

    sd = WeightLoader().load_s2mel()["length_regulator"]
    m = InterpolateRegulator(
        channels=512,
        sampling_ratios=[1, 1, 1, 1],
        is_discrete=False,
        in_channels=1024,
        codebook_size=2048,
    ).to(dev)
    m.load_official(sd)
    return m.eval()


def _check(name, inp, ylens, ref_path):
    m = _load_model(DEV)
    with torch.no_grad():
        out = m(inp, ylens=ylens, n_quantizers=3)[0]
    ref = torch.load(ref_path, weights_only=False).to(DEV)
    assert out.shape == ref.shape, f"shape {out.shape} != ref {ref.shape}"
    d = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-3)
    print(f"[align] {name}: shape={tuple(out.shape)} max_abs_diff={d:.3e} allclose={ok}")
    assert ok, f"{name} not aligned: max_abs_diff={d}"
    return d


if __name__ == "__main__":
    # Case 1: S_infer -> cond (ylens=522)
    S_infer = torch.load(f"{DUMPS}/s2mel.S_infer.pt", weights_only=False).to(DEV)
    ylens = torch.LongTensor([int(S_infer.shape[1] * 1.72)]).to(DEV)  # 522
    d1 = _check("cond", S_infer, ylens, f"{DUMPS}/s2mel.cond.pt")

    # Case 2: spk_cond -> prompt_condition (ylens=523)
    spk_cond = torch.load(f"{DUMPS}/gpt.spk_cond_w2v.pt", weights_only=False).to(DEV)
    ylens2 = torch.LongTensor([523]).to(DEV)
    d2 = _check("prompt_condition", spk_cond, ylens2, f"{DUMPS}/s2mel.prompt_condition.pt")

    print(f"\ncond diff={d1:.3e}, prompt_condition diff={d2:.3e}")
    print("LENGTH_REGULATOR ALIGN OK")
