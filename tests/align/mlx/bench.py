# GPU (Metal) end-to-end benchmark: RTF (greedy + beam3) and peak memory per
# dtype configuration (fp32 / fp16 / w4a16). Uses test.wav ref + Chinese text.
# Run: .venv/bin/python tests/align/mlx/bench.py
import os
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

os.environ.setdefault("WINDEXTTS_WEIGHTS_DIR", "/Volumes/2T/IndexTTS-2.5")
MLX = Path("/Volumes/2T/IndexTTS-2.5-mlx")
sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root for windextts_mlx

TEXT = "你们好，今天我们要讲一个关于人工智能的故事。它正在改变我们生活的方方面面。"


def bench(cfg, greedy_only=False):
    from windextts_mlx.inference import WIndexTTSMLX

    m = WIndexTTSMLX(weights_dir=MLX, dtype=cfg["dtype"], quantize=cfg.get("quantize", False),
                     w2v_fp16=cfg.get("w2v_fp16", False))
    # inference-only peak: load-time peak (mmap preload of the full fp32 set +
    # fp16 conversion doubles) is a one-time cost; torch.max_memory_allocated
    # baselines also measure the inference phase only.
    mx.reset_peak_memory()

    def run(beam):
        t0 = time.perf_counter()
        sr, out = m.infer("/Volumes/2T/WIndexTTS/test.wav", TEXT, do_sample=False, num_beams=beam)
        wall = time.perf_counter() - t0
        dur = out.shape[-1] / sr
        peak = mx.get_peak_memory() / 1e9
        mx.reset_peak_memory()
        return wall, dur, peak

    # warmup (kernel compile + alloc)
    run(1)
    rows = []
    w, d, p = run(1)
    rows.append(("greedy", w, d, p))
    if not greedy_only:
        w, d, p = run(3)
        rows.append(("beam3", w, d, p))
    name = cfg["dtype"] + ("+w4a16" if cfg.get("quantize") else "") + ("+w2v16" if cfg.get("w2v_fp16") else "")
    for mode, w, d, p in rows:
        print(f"[{name:7s}] {mode:6s} wall {w:6.2f}s audio {d:4.2f}s RTF {w / d:5.2f}x  peak {p:5.2f} GB")
    # release the model so a second config in the same process does not double
    # count this config's resident weights in its peak.
    del m


if __name__ == "__main__":
    cfgs = []
    for arg in sys.argv[1:] or ["fp32", "fp16", "w4a16", "fp16-fast"]:
        cfgs.append({"dtype": arg, "quantize": arg == "w4a16",
                    "w2v_fp16": arg == "fp16-fast"})
    for cfg in cfgs:
        bench(cfg)
