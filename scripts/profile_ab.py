#!/usr/bin/env python
"""WIndexTTS profiler-free A/B harness — the FINAL word on optimizations.

Two configs selected by --a and --b flags; each is warmed up then measured with
>=5 profiler-free repeats (cuda-synced). Reports mean/median/min/max + delta.
Optionally pairs with a profiler trace capture (--trace) for the diff
explanation.

Config flags (composable):
  --steps N          CFM Euler steps (default 12)
  --bf16 / --fp32    DiT compute dtype (default: follow warmup = bf16 autocast)
  --graph / --eager  DiT CUDA-graph path (default eager)
  --teacache on/off  TeaCache (default on)

Usage:
  /root/index-tts/.venv/bin/python scripts/profile_ab.py \
      --a "fp32-eager" --b "bf16-graph" --trace

Output:
  AUTHORITATIVE A/B table + optional per-config profiler trace analysis.
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
TEXTS = [
    "大家好，这是一个测试。",
    "今天天气真不错。",
    "人工智能正在改变世界。",
    "语音合成技术发展迅速。",
]


def make_config(args) -> dict:
    cfg = dict(steps=args.steps)
    if args.bf16:
        cfg["bf16"] = True
    if args.fp32:
        cfg["bf16"] = False
    if args.graph:
        cfg["graph"] = True
    if args.eager:
        cfg["graph"] = False
    if args.teacache is not None:
        cfg["teacache"] = args.teacache
    return cfg


def apply_config(tts, cfg: dict) -> None:
    """Apply a config to a freshly-loaded model. Returns nothing."""
    if cfg.get("bf16", True):
        tts.s2mel.cfm.estimator_autocast_dtype = torch.bfloat16
    else:
        tts.s2mel.cfm.estimator_autocast_dtype = None
    tts.s2mel_use_graph = bool(cfg.get("graph", False))
    tc = cfg.get("teacache", True)
    tts.s2mel.cfm.estimator.enable_teacache(thresh=0.15) if tc else (
        setattr(tts.s2mel.cfm.estimator, "teacache_enabled", False))
    tts.s2mel.cfm.estimator.teacache_enabled = bool(tc)


def measure(tts, cfg: dict, repeats: int) -> dict:
    """Profiler-free authoritative latency over TEXTS."""
    # warm
    for _ in range(3):
        tts.infer(REF, "warmup 测试。", "ZH", cfm_steps=cfg["steps"])
    torch.cuda.synchronize()
    times = []
    for txt in TEXTS:
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tts.infer(REF, txt, "ZH", cfm_steps=cfg["steps"])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return dict(mean_ms=statistics.mean(times),
                median_ms=statistics.median(times),
                min_ms=min(times), max_ms=max(times), n=len(times))


def capture_trace(tts, cfg: dict, label: str) -> None:
    """One profiled run for diff explanation; analyzes + saves trace."""
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=True) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tts.infer(REF, TEXTS[0], "ZH", cfm_steps=cfg["steps"])
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
    path = OUTDIR / f"ab_{label}.json.gz"
    prof.export_chrome_trace(str(path))
    res = profiler.analyze(f"AB-{label}", profiler.load_trace(path), wall_ms=wall)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline", help="config A label")
    ap.add_argument("--b", default="opt", help="config B label")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--teacache", type=int, default=None, choices=[0, 1])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--trace", action="store_true",
                    help="also capture a profiler trace per config")
    args = ap.parse_args()

    cfg = make_config(args)
    print(f">> config: {cfg}")

    # ---- config A ----
    print(f"\n== config A [{args.a}] ==")
    tts_a = WIndexTTS(device="cuda", dtype=torch.float16)
    tts_a.warmup()
    apply_config(tts_a, cfg)
    lat_a = measure(tts_a, cfg, args.repeats)
    res_a = capture_trace(tts_a, cfg, args.a) if args.trace else None
    print(f"  A {args.a}: mean={lat_a['mean_ms']:.0f}ms "
          f"median={lat_a['median_ms']:.0f}ms min={lat_a['min_ms']:.0f}ms "
          f"max={lat_a['max_ms']:.0f}ms (n={lat_a['n']})")
    del tts_a
    torch.cuda.empty_cache()

    # ---- config B ----
    print(f"\n== config B [{args.b}] ==")
    tts_b = WIndexTTS(device="cuda", dtype=torch.float16)
    tts_b.warmup()
    apply_config(tts_b, cfg)
    lat_b = measure(tts_b, cfg, args.repeats)
    res_b = capture_trace(tts_b, cfg, args.b) if args.trace else None
    print(f"  B {args.b}: mean={lat_b['mean_ms']:.0f}ms "
          f"median={lat_b['median_ms']:.0f}ms min={lat_b['min_ms']:.0f}ms "
          f"max={lat_b['max_ms']:.0f}ms (n={lat_b['n']})")
    del tts_b
    torch.cuda.empty_cache()

    # ---- verdict ----
    delta = lat_a["mean_ms"] - lat_b["mean_ms"]
    pct = delta / lat_a["mean_ms"] * 100 if lat_a["mean_ms"] else 0
    print(f"\n{'='*60}")
    print(f"AUTHORITATIVE A/B ({args.a} vs {args.b}), n={lat_a['n']} each")
    print(f"{'metric':8s} {args.a:>10s} {args.b:>10s} {'delta':>10s}")
    for m in ("mean_ms", "median_ms", "min_ms", "max_ms"):
        print(f"{m:8s} {lat_a[m]:10.0f} {lat_b[m]:10.0f} "
              f"{lat_b[m]-lat_a[m]:+10.0f}")
    print(f"\n  {args.b} is {abs(delta):.0f}ms ({abs(pct):.1f}%) "
          f"{'FASTER' if delta > 0 else 'SLOWER' if delta < 0 else 'EQUAL'} "
          f"than {args.a} by mean")
    if args.trace and res_a and res_b:
        gpu_a, gpu_b = res_a.get("gpu_ms", 0), res_b.get("gpu_ms", 0)
        idle_a, idle_b = res_a.get("idle_pct", 0), res_b.get("idle_pct", 0)
        print(f"  [trace] GPU-work: A={gpu_a:.0f}ms B={gpu_b:.0f}ms | "
              f"idle%: A={idle_a:.1f} B={idle_b:.1f}")
        print(f"  -> GPU-work drop with idle% rise = kernel faster but host "
              f"bubble exposed (add CUDA graph);")
        print(f"  -> GPU-work same but wall faster = launch/dispatch reduction.")


if __name__ == "__main__":
    main()
