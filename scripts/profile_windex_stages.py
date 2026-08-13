#!/usr/bin/env python
"""WIndexTTS per-stage profiler with GPU kernel taxonomy + idle-gap analysis.

Design grounded in:
  - /root/Profiling_LLM_Inference_with_SGLang_and_Torch_Profiler_EN.md
    (understand model first; read CPU-then-GPU; identify repeating patterns)
  - sglang llm-torch-profiler-analysis skill
    (kernel taxonomy gemm/conv/attn/norm/elem/memory; >=1% render cutoff;
     exclusive_ratio>=0.45 overlap heuristic; same-stream gap dependency)
  - vllm-omni diffusion-perf-opt skill
    (idle_pct = host bubbles; profiler latency is DIAGNOSTIC only, never the
     final latency claim; CUDA-Graph regressions come from re-alloc / backend
     pinning / sync behavior / stage masking)

Outputs per stage + e2e:
  - top kernels by GPU time (categorized)
  - category rollup (how much GEMM vs conv vs attn vs elementwise vs memory)
  - idle_pct (GPU idle fraction = host-bubble opportunity)
  - launch count vs duration (launch-overhead signal)
  - chrome trace for manual Perfetto inspection

Usage:
  /root/index-tts/.venv/bin/python scripts/profile_windex_stages.py [--steps N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root/WIndexTTS")
from windextts.inference import WIndexTTS  # noqa: E402

OUTDIR = Path("/root/windextts_dumps/profiles")
OUTDIR.mkdir(parents=True, exist_ok=True)
REF = "/root/WIndexTTS/test.wav"
TEXT = "人工智能正在改变世界。"

# ---- kernel taxonomy (from sglang triage, adapted for TTS) -------------
# ordered: first match wins. conv added (TTS-specific, big in BigVGAN).
CATEGORY_PATTERNS = [
    ("conv",         [r"convolution", r"convolve", r"cudnn_conv", r"_conv\d?", r"conv_transpose"]),
    ("communication",[r"nccl", r"all_reduce", r"all_gather", r"allgather", r"reduce_scatter"]),
    ("attention",    [r"flash_attn", r"flashattention", r"fmha", r"_efficient_attention",
                      r"scaled_dot_product", r"attention", r"prefill", r"decode_attention"]),
    ("gemm",         [r"cutlass", r"gemm", r"matmul", r"gemv", r"cublas", r"wgmma",
                      r"addmm", r"\bmma\b", r"bmm", r"_gemm"]),
    ("norm",         [r"rmsnorm", r"layernorm", r"layer_norm", r"_norm_", r"group_norm"]),
    ("rope",         [r"rotary", r"rope", r"apply_rotary"]),
    ("activation",   [r"silu", r"gelu", r"relu", r"sigmoid", r"snake", r"tanh", r"act_and_mul"]),
    ("quantize",     [r"quant", r"dequant", r"fp8", r"cvt_"]),
    ("reduce",       [r"\btopk\b", r"argmax", r"reduce", r"multinomial", r"sampling"]),
    ("memory",       [r"memcpy", r"memset", r"_copy_", r"\.copy_", r"to_copy",
                      r"_fill_", r"empty_cache", r"dma", r"prefetch"]),
    ("elementwise",  [r"elementwise", r"vectorized", r"_add_kernel", r"_mul_kernel",
                      r"_sub_kernel", r"_add\b", r"_mul\b", r"_sub\b", r"where", r"clamp"]),
]


def classify_kernel(name: str) -> str:
    low = name.lower()
    for cat, pats in CATEGORY_PATTERNS:
        for p in pats:
            if re.search(p, low):
                return cat
    return "other"


def load_trace_events(path: Path) -> list[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        data = json.load(f)
    evts = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in evts if e.get("ph") == "X" and isinstance(e.get("dur"), (int, float))]


def gpu_kernel_events(events: list[dict]) -> list[dict]:
    # GPU kernels live on a thread named like "stream N" with cat "kernel",
    # or have cat in {kernel, gpu_memcpy, gpu_memset}. Filter by cat + dur>0.
    out = []
    for e in events:
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
            if e.get("dur", 0) > 0:
                out.append(e)
    return out


def merge_intervals(intervals: list[tuple[float, float]]) -> tuple[float, float]:
    """Return (busy_union, span) from sorted (start, end) intervals."""
    if not intervals:
        return 0.0, 0.0
    intervals = sorted(intervals)
    busy = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            busy += ce - cs
            cs, ce = s, e
    busy += ce - cs
    span = intervals[-1][1] - intervals[0][0]
    return busy, span


def analyze_stage(label: str, events: list[dict], wall_ms: float) -> dict:
    kernels = gpu_kernel_events(events)
    if not kernels:
        print(f"\n[{label}] no GPU kernel events"); return {}
    # category rollup
    cat_tot = Counter()
    cat_cnt = Counter()
    name_tot = Counter()
    name_cnt = Counter()
    for k in kernels:
        nm = k.get("name", "?")
        cat = classify_kernel(nm)
        d = k["dur"]  # us
        cat_tot[cat] += d
        cat_cnt[cat] += 1
        name_tot[nm] += d
        name_cnt[nm] += 1
    total_gpu_us = sum(cat_tot.values())
    # idle analysis (host bubbles)
    intervals = [(k["ts"], k["ts"] + k["dur"]) for k in kernels]
    busy_us, span_us = merge_intervals(intervals)
    idle_us = span_us - busy_us
    idle_pct = idle_us / span_us * 100 if span_us else 0
    # launch overhead signal: kernels < 10us
    tiny = sum(1 for k in kernels if k["dur"] < 10)
    tiny_us = sum(k["dur"] for k in kernels if k["dur"] < 10)

    print(f"\n{'='*70}")
    print(f"[{label}]  wall={wall_ms:.0f}ms  GPU-work={total_gpu_us/1000:.0f}ms  "
          f"idle={idle_pct:.1f}% ({idle_us/1000:.0f}ms bubble)  launches={len(kernels)}")
    print(f"  tiny_kernels(<10us)={tiny} ({tiny_us/1000:.1f}ms) — launch-overhead signal")
    print(f"  -- category rollup (>=1%) --")
    for cat, us in cat_tot.most_common():
        pct = us / total_gpu_us * 100
        if pct >= 1.0:
            print(f"    {cat:14s} {us/1000:7.1f}ms  {pct:5.1f}%   ({cat_cnt[cat]} launches)")
    print(f"  -- top kernels (>=1%) --")
    for nm, us in name_tot.most_common(15):
        pct = us / total_gpu_us * 100
        if pct < 1.0:
            break
        cat = classify_kernel(nm)
        print(f"    [{cat:11s}] {us/1000:6.1f}ms {pct:5.1f}% x{name_cnt[nm]:4d}  {nm[:48]}")
    return dict(label=label, wall_ms=wall_ms, gpu_ms=total_gpu_us/1000,
                idle_pct=idle_pct, cat={c: cat_tot[c]/1000 for c in cat_tot})


def capture_stage(tts, label, fn):
    """Run fn under torch.profiler; return (events, wall_ms)."""
    with torch.no_grad():
        try:  # warm the stage
            fn()
            torch.cuda.synchronize()
        except Exception:
            pass
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); wall = (time.perf_counter()-t0)*1000
    path = OUTDIR / f"stage_{label}.json.gz"
    prof.export_chrome_trace(str(path))
    return load_trace_events(path), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15, help="CFM Euler steps")
    args = ap.parse_args()

    print(">> loading model...")
    tts = WIndexTTS(device="cuda", dtype=torch.float16)
    for _ in range(2):
        tts.infer(REF, "warmup.", "ZH", cfm_steps=args.steps)
    torch.cuda.synchronize()

    # ---- capture e2e (authoritative: includes cross-stage host bubbles) ----
    print(">> capturing e2e...")
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            sr, w = tts.infer(REF, TEXT, "ZH", cfm_steps=args.steps)
            torch.cuda.synchronize(); wall = (time.perf_counter()-t0)*1000
    epath = OUTDIR / f"e2e_steps{args.steps}.json.gz"
    prof.export_chrome_trace(str(epath))
    e2e = analyze_stage(f"E2E (steps={args.steps})", load_trace_events(epath), wall)

    # ---- per-stage: capture boundary tensors from one real infer ----
    print("\n>> caching intermediates for per-stage isolation...")
    tts.s2mel.cfm.estimator.enable_teacache(thresh=0.15)
    cache = {}
    orig_gpt_gen = tts.gpt.generate
    orig_s2mel = tts.s2mel.inference
    def grab_gpt(conds, text_inputs, langs, *a, **kw):
        cache["gpt_conds"]=conds; cache["gpt_text"]=text_inputs
        cache["gpt_lang"]=langs; cache["gpt_kw"]=kw
        r = orig_gpt_gen(conds, text_inputs, langs, *a, **kw); cache["gpt_codes"]=r
        return r
    def grab_s2mel(spk, s, refmel, style, **kw):
        cache["spk"]=spk; cache["s"]=s; cache["refmel"]=refmel; cache["style"]=style
        cache["s2mel_kw"]=kw
        return orig_s2mel(spk, s, refmel, style, **kw)
    tts.gpt.generate = grab_gpt
    tts.s2mel.inference = grab_s2mel
    tts.infer(REF, TEXT, "ZH", cfm_steps=args.steps)
    tts.gpt.generate = orig_gpt_gen
    tts.s2mel.inference = orig_s2mel

    # Stage 1: GPT-AR with REAL cached inputs (same conds/text/lang)
    print(">> capturing GPT-AR...")
    g_ev, g_wall = capture_stage(tts, "gpt",
        lambda: tts.gpt.generate(
            cache["gpt_conds"], cache["gpt_text"], cache["gpt_lang"],
            **cache["gpt_kw"]))
    print(">> capturing S2Mel-CFM...")
    s_ev, s_wall = capture_stage(tts, "s2mel",
        lambda: tts.s2mel.inference(cache["spk"], cache["s"], cache["refmel"],
                                     cache["style"], **cache["s2mel_kw"]))
    # recompute mel for bigvgan input
    mel = tts.s2mel.inference(cache["spk"], cache["s"], cache["refmel"],
                              cache["style"], **cache["s2mel_kw"])
    bg_dtype = next(tts.bigvgan.parameters()).dtype
    print(">> capturing BigVGAN...")
    b_ev, b_wall = capture_stage(tts, "bigvgan",
        lambda: tts.bigvgan(mel.to(bg_dtype)))

    g = analyze_stage("GPT-AR", g_ev, g_wall)
    s = analyze_stage(f"S2Mel-CFM (steps={args.steps})", s_ev, s_wall)
    b = analyze_stage("BigVGAN", b_ev, b_wall)

    # summary
    print(f"\n{'='*70}\nSUMMARY (steps={args.steps})")
    print(f"{'stage':16s} {'wall(ms)':>9s} {'gpu(ms)':>8s} {'idle%':>6s}")
    for r in [e2e, g, s, b]:
        if r: print(f"{r['label']:16s} {r['wall_ms']:9.0f} {r['gpu_ms']:8.0f} {r['idle_pct']:6.1f}")
    print(f"\n  -> 'idle%' is host-bubble / launch-overhead opportunity")
    print(f"  -> sum of per-stage GPU should be <= e2e GPU (overlap/stage-host gaps")


if __name__ == "__main__":
    main()
