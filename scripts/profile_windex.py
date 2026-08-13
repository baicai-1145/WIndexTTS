#!/usr/bin/env python
"""WIndexTTS GPU kernel-level profiler.

Captures a torch.profiler trace of one end-to-end inference, split into the
three neural stages (GPT-AR / S2Mel-CFM / BigVGAN), with CPU+GPU activities.

Outputs:
  - <outdir>/windex_{stage}.json.gz   (chrome/perfetto traces per stage)
  - <outdir>/windex_summary.txt       (top kernels by GPU time per stage)

Usage:
  /root/index-tts/.venv/bin/python scripts/profile_windex.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

sys.path.insert(0, "/root/WIndexTTS")
from windextts.inference import WIndexTTS  # noqa: E402

OUTDIR = Path("/root/windextts_dumps/profiles")
OUTDIR.mkdir(parents=True, exist_ok=True)
TEXT = "人工智能正在改变世界。"
REF = "/root/WIndexTTS/test.wav"


def stage_trace(tts: WIndexTTS, stage_name: str, fn, label: str) -> float:
    """Profile a single stage's callable; write trace + print top kernels."""
    trace_path = OUTDIR / f"windex_{label}.json.gz"
    # warm the stage once (prime cudnn autotune, avoid first-call noise)
    with torch.no_grad():
        try:
            fn(warm=True)
        except Exception:
            pass
    torch.cuda.synchronize()
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
    # save chrome trace
    prof.export_chrome_trace(str(trace_path))
    print(f"\n=== {stage_name} : {dt*1000:.1f}ms  (trace -> {trace_path.name}) ===")
    # top kernels by CUDA self-time
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cuda_time_total", row_limit=12))
    return dt, out


def main():
    print(">> loading model...")
    tts = WIndexTTS(device="cuda", dtype=torch.float16)
    # full warmup
    for _ in range(2):
        tts.infer(REF, "warmup.", "ZH")
    torch.cuda.synchronize()

    # We profile a FULL end-to-end infer as one trace (simplest, captures
    # cross-stage interactions + Python dispatch between stages). Then we also
    # break out per-stage by hooking.
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sr, w = tts.infer(REF, TEXT, "ZH")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f">> e2e steady: mean={statistics.mean(times)*1000:.0f}ms "
          f"min={min(times)*1000:.0f}ms")

    # ---- capture ONE profiled e2e run (the authoritative trace) ----
    trace_path = OUTDIR / "windex_e2e.json.gz"
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            sr, w = tts.infer(REF, TEXT, "ZH")
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    print(f"\n=== E2E trace -> {trace_path.name} ===")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cuda_time_total", row_limit=20))

    # also dump per-stage breakdown via events (group_by is hard; rely on the
    # e2e trace which we can slice manually in Perfetto)
    print(f"\n>> traces in {OUTDIR}")
    print(">> open windex_e2e.json.gz in https://ui.perfetto.dev or chrome://tracing")


if __name__ == "__main__":
    main()
