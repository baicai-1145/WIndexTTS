#!/usr/bin/env python
"""Run each alignment test in its own process (pytest crashes natively here)."""
import subprocess
import sys
from pathlib import Path

TESTS = ["mel", "featurizer", "fbank", "resample", "normalizer", "campplus", "codec",
         "length_regulator", "w2v_bert", "emo_conditioning", "dit", "s2mel_inference",
         "bigvgan", "gpt_prefill", "gpt_greedy", "gpt_beam3", "e2e", "e2e_fp16"]

sel = sys.argv[1:] or TESTS
mlx_dir = Path(__file__).resolve().parent
root = mlx_dir.parents[2]
fail = []
for t in sel:
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, r'{mlx_dir}');"
         f"from test_align import test_{t}; test_{t}()"],
        cwd=root, capture_output=True, text=True, timeout=1200)
    out = (r.stdout + r.stderr)
    ok = r.returncode == 0
    line = [ln for ln in out.splitlines() if "cosine" in ln or "OK" in ln]
    print(f"{'PASS' if ok else 'FAIL'} test_{t}  {' | '.join(line[-3:])}", flush=True)
    if not ok and "cosine" not in " ".join(line):
        tail = [ln for ln in out.splitlines() if ln.strip()][-4:]
        print("   ", " / ".join(tail), flush=True)
    if not ok:
        fail.append(t)
print(f"\n{len(sel) - len(fail)}/{len(sel)} passed; failed: {fail or 'none'}")
