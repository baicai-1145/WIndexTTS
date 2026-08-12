"""Numerical alignment test: windextts CAMPPlus vs official IndexTTS-2.5.

Reference tensors (dumped by scripts/dump_indextts_tensors.py from the official
index-tts environment, A10G CUDA fp32, /root/WIndexTTS/test.wav):

- /root/windextts_dumps/campplus.fbank_cm.pt   input  [606, 80]  Kaldi fbank @16k,
                                               mean-subtracted (per column)
- /root/windextts_dumps/campplus.spk_emb_192.pt official output [1, 192] float32

Acceptance: torch.allclose(my_out, ref_out, atol=1e-4, rtol=1e-3).

IMPORTANT: the official reference is dumped on **CUDA fp32** (the official
infer_v2_5.py runs campplus in fp32 on GPU — only GPT gets bf16). CUDA fp32 and
CPU fp32 conv/batchnorm kernels differ by ~4e-3 on this deep net, so strict
alignment is only meaningful on CUDA. The strict test therefore runs on CUDA
when available; on CPU we run a loose sanity check (rtol~1%) instead.

Run: /root/index-tts/.venv/bin/python -m pytest tests/align/test_campplus_align.py -s
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from windextts.models.campplus import CAMPPlus  # noqa: E402

WEIGHTS = "/root/IndexTTS-2.5/hf_cache/campplus_cn_common.bin"
DUMP_DIR = "/root/windextts_dumps"

ATOL, RTOL = 1e-4, 1e-3  # contract tolerance
CPU_ATOL, CPU_RTOL = 1e-2, 1e-2  # loose CPU sanity bounds (CUDA-vs-CPU kernel noise)

CUDA_AVAIL = torch.cuda.is_available()


def _load_ref(name: str) -> torch.Tensor:
    return torch.load(os.path.join(DUMP_DIR, name), map_location="cpu")


def _make_model() -> CAMPPlus:
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
    model.eval()
    return model


def test_campplus_state_dict_loads_strict() -> None:
    model = _make_model()
    sd = torch.load(WEIGHTS, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"


def test_campplus_output_aligns_with_official_cuda() -> None:
    """Strict numerical alignment — requires CUDA (matches official dump env)."""
    if not CUDA_AVAIL:
        pytest.skip("CUDA not available; strict alignment requires CUDA")
    model = _make_model().to("cuda")

    fbank = _load_ref("campplus.fbank_cm.pt")  # [606, 80]
    ref_out = _load_ref("campplus.spk_emb_192.pt")  # [1, 192]

    with torch.no_grad():
        my_out = model(fbank.unsqueeze(0).cuda()).cpu()  # [1, 606, 80] -> [1, 192]

    assert my_out.shape == ref_out.shape
    assert my_out.dtype == ref_out.dtype

    close = torch.allclose(my_out, ref_out, atol=ATOL, rtol=RTOL)
    max_abs = (my_out - ref_out).abs().max().item()
    mean_abs = (my_out - ref_out).abs().mean().item()
    print(f"\n[cuda] max_abs_diff = {max_abs:.3e}  mean_abs_diff = {mean_abs:.3e}")
    print(f"[cuda] allclose(atol={ATOL}, rtol={RTOL}) = {close}")
    assert close, (
        f"OUTPUT MISMATCH: max_abs_diff={max_abs:.3e} exceeds "
        f"atol={ATOL}/rtol={RTOL} (mean_abs={mean_abs:.3e})"
    )


def test_campplus_output_cpu_sanity() -> None:
    """Loose CPU sanity check (CUDA-vs-CPU kernel numerics differ ~4e-3)."""
    model = _make_model()

    fbank = _load_ref("campplus.fbank_cm.pt")
    ref_out = _load_ref("campplus.spk_emb_192.pt")

    with torch.no_grad():
        my_out = model(fbank.unsqueeze(0))

    assert my_out.shape == ref_out.shape
    close = torch.allclose(my_out, ref_out, atol=CPU_ATOL, rtol=CPU_RTOL)
    max_abs = (my_out - ref_out).abs().max().item()
    print(f"\n[cpu sanity] max_abs_diff = {max_abs:.3e}")
    assert close, f"CPU sanity failed: max_abs_diff={max_abs:.3e}"


if __name__ == "__main__":
    test_campplus_state_dict_loads_strict()
    print("state_dict strict load OK (937 keys)")
    if CUDA_AVAIL:
        test_campplus_output_aligns_with_official_cuda()
        print("CUDA ALIGN OK")
    else:
        test_campplus_output_cpu_sanity()
        print("CPU sanity OK (no CUDA)")
