#!/usr/bin/env python
"""WIndexTTS per-stage capture harness + analysis.

Captures torch.profiler traces for E2E + GPT-AR / S2Mel-CFM / BigVGAN stages,
then runs the windextts.profiler analyzer on each. ALSO runs profiler-free
wall-clock repeats (>=3) for the authoritative latency — profiler wall time is
distorted and diagnostic only.

Usage:
  /root/index-tts/.venv/bin/python scripts/profile_windex_stages.py [--steps N]

Outputs (in /root/windextts_dumps/profiles/):
  - e2e_steps{N}.json.gz, stage_{gpt,s2mel,bigvgan}.json.gz  (chrome traces)
  - per-stage analyzer report + authoritative latency table
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root/WIndexTTS")
from windextts.inference import WIndexTTS  # noqa: E402
from windextts import profiler  # noqa: E402

OUTDIR = Path("/root/windextts_dumps/profiles")
OUTDIR.mkdir(parents=True, exist_ok=True)
REF = "/root/WIndexTTS/test.wav"
TEXT = "人工智能正在改变世界。"


def profiler_free_latency(fn, repeats: int = 5, label: str = "") -> dict:
    """Authoritative latency: NO profiler, >=3 repeats with cuda sync.
    Returns {mean_ms, median_ms, min_ms, max_ms}."""
    # warm once more (steady state)
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    r = dict(mean_ms=statistics.mean(times), median_ms=statistics.median(times),
             min_ms=min(times), max_ms=max(times), n=repeats)
    print(f"    [authoritative] {label}: mean={r['mean_ms']:.0f}ms "
          f"median={r['median_ms']:.0f}ms min={r['min_ms']:.0f}ms "
          f"max={r['max_ms']:.0f}ms (n={repeats}, profiler-free)")
    return r


def capture_trace(tts, label: str, fn, wall_fn) -> tuple[list[dict], float]:
    """Capture a torch.profiler trace for fn. Returns (events, profiler_wall_ms).
    wall_fn() is a lambda that runs the same work profiler-free for the
    authoritative measurement (called separately by the caller)."""
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=True) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
    path = OUTDIR / f"stage_{label}.json.gz"
    prof.export_chrome_trace(str(path))
    return profiler.load_trace(path), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15, help="CFM Euler steps")
    ap.add_argument("--repeats", type=int, default=5,
                    help="profiler-free repeats for authoritative latency")
    args = ap.parse_args()

    print(">> loading model...")
    tts = WIndexTTS(device="cuda", dtype=torch.float16)
    tts.warmup()  # enables DiT bf16 autocast + captures GPT CUDA graph
    for _ in range(2):
        tts.infer(REF, "warmup.", "ZH", cfm_steps=args.steps)
    torch.cuda.synchronize()

    # ---- e2e: authoritative latency first, then one profiled trace ----
    print(">> e2e authoritative latency (profiler-free)...")
    e2e_lat = profiler_free_latency(
        lambda: tts.infer(REF, TEXT, "ZH", cfm_steps=args.steps),
        repeats=args.repeats, label="E2E")

    print(">> capturing e2e trace...")
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=True) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sr, w = tts.infer(REF, TEXT, "ZH", cfm_steps=args.steps)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
    epath = OUTDIR / f"e2e_steps{args.steps}.json.gz"
    prof.export_chrome_trace(str(epath))
    e2e_res = profiler.analyze(f"E2E (steps={args.steps})",
                               profiler.load_trace(epath), wall_ms=wall)

    # ---- per-stage: capture boundary tensors from one real infer ----
    print("\n>> caching intermediates for per-stage isolation...")
    tts.s2mel.cfm.estimator.enable_teacache(thresh=0.15)
    cache = {}
    orig_gpt_gen = tts.gpt.generate
    orig_s2mel = tts.s2mel.inference

    def grab_gpt(conds, text_inputs, langs, *a, **kw):
        cache["gpt_conds"] = conds
        cache["gpt_text"] = text_inputs
        cache["gpt_lang"] = langs
        cache["gpt_kw"] = kw
        r = orig_gpt_gen(conds, text_inputs, langs, *a, **kw)
        cache["gpt_codes"] = r
        return r

    def grab_s2mel(spk, s, refmel, style, **kw):
        cache["spk"] = spk
        cache["s"] = s
        cache["refmel"] = refmel
        cache["style"] = style
        cache["s2mel_kw"] = kw
        return orig_s2mel(spk, s, refmel, style, **kw)

    tts.gpt.generate = grab_gpt
    tts.s2mel.inference = grab_s2mel
    tts.infer(REF, TEXT, "ZH", cfm_steps=args.steps)
    tts.gpt.generate = orig_gpt_gen
    tts.s2mel.inference = orig_s2mel

    # ---- per-stage traces + authoritative latencies ----
    gpt_fn = lambda: tts.gpt.generate(  # noqa: E731
        cache["gpt_conds"], cache["gpt_text"], cache["gpt_lang"], **cache["gpt_kw"])
    s2mel_fn = lambda: tts.s2mel.inference(  # noqa: E731
        cache["spk"], cache["s"], cache["refmel"], cache["style"], **cache["s2mel_kw"])
    mel = tts.s2mel.inference(cache["spk"], cache["s"], cache["refmel"],
                              cache["style"], **cache["s2mel_kw"])
    bg_dtype = next(tts.bigvgan.parameters()).dtype
    bg_fn = lambda: tts.bigvgan(mel.to(bg_dtype))  # noqa: E731

    print(">> capturing GPT-AR...")
    g_ev, g_wall = capture_trace(tts, "gpt", gpt_fn, None)
    g_lat = profiler_free_latency(gpt_fn, repeats=args.repeats, label="GPT-AR")

    print(">> capturing S2Mel-CFM...")
    s_ev, s_wall = capture_trace(tts, "s2mel", s2mel_fn, None)
    s_lat = profiler_free_latency(s2mel_fn, repeats=args.repeats, label="S2Mel-CFM")

    print(">> capturing BigVGAN...")
    b_ev, b_wall = capture_trace(tts, "bigvgan", bg_fn, None)
    b_lat = profiler_free_latency(bg_fn, repeats=args.repeats, label="BigVGAN")

    g = profiler.analyze("GPT-AR", g_ev, wall_ms=g_wall)
    s = profiler.analyze(f"S2Mel-CFM (steps={args.steps})", s_ev, wall_ms=s_wall)
    b = profiler.analyze("BigVGAN", b_ev, wall_ms=b_wall)

    # ---- summary: authoritative table ----
    print(f"\n{'='*74}")
    print(f"SUMMARY (steps={args.steps}) — AUTHORITATIVE (profiler-free) latency")
    print(f"{'stage':18s} {'mean(ms)':>9s} {'min(ms)':>8s} {'gpu-work(ms)':>13s} "
          f"{'idle%':>6s}")
    rows = [
        ("E2E", e2e_lat, e2e_res),
        ("GPT-AR", g_lat, g),
        ("S2Mel-CFM", s_lat, s),
        ("BigVGAN", b_lat, b),
    ]
    for name, lat, res in rows:
        gpu_ms = res.get("gpu_ms", 0) if res else 0
        idle = res.get("idle_pct", 0) if res else 0
        print(f"{name:18s} {lat['mean_ms']:9.0f} {lat['min_ms']:8.0f} "
              f"{gpu_ms:13.0f} {idle:6.1f}")
    print(f"\n  -> GPU-work + idle% are from profiler traces (diagnostic);")
    print(f"  -> mean/min are profiler-free (the authoritative latency).")
    print(f"  -> gap histograms reveal CUDA-Graph-eliminatable micro-dispatch "
          f"overhead.")


if __name__ == "__main__":
    main()
