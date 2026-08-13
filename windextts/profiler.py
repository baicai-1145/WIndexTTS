"""WIndexTTS professional trace analyzer — pure Python, no torch import.

Evidence-driven GPU profiler analysis for the WIndexTTS pipeline. Built from
the methodology in sglang's llm-torch-profiler-analysis skill and vllm-omni's
diffusion-perf-opt skill (see docs/PROFILER_SPEC.md).

Design principles:
  - pure-Python over trace JSON: importable anywhere, unit-testable without
    loading the model
  - profiler latency is DIAGNOSTIC only — never the final latency claim
  - the S2Mel micro-gap histogram is the differentiator: ~75% of its idle is
    10-50us Python-dispatch gaps that vllm-omni's --min-gap-ms 5 would miss

Public API:
  - load_trace(path) -> list[event-dict]
  - split_events(events) -> (gpu_events, cpu_events)
  - classify_kernel(name) -> category
  - analyze(label, events, wall_ms=None, min_gap_us=100, topn=15) -> dict
  - CLI: python -m windextts.profiler <trace.json.gz> [--min-gap-us N] [--topn N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 1. Trace loading
# ---------------------------------------------------------------------------

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
CPU_CATS = {"cpu_op", "user_annotation", "python_function",
            "cuda_runtime", "cuda_driver"}


def load_trace(path: str | Path) -> list[dict]:
    """Load a chrome-trace .json or .json.gz into a list of X-duration events."""
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        data = json.load(f)
    evts = data.get("traceEvents", data) if isinstance(data, dict) else data
    out = []
    for e in evts:
        if not isinstance(e, dict):
            continue
        if e.get("ph") == "X" and isinstance(e.get("dur"), (int, float)):
            out.append(e)
    return out


def split_events(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate GPU (kernel/memcpy/memset) from CPU-side events."""
    gpu, cpu = [], []
    for e in events:
        cat = e.get("cat", "")
        if cat in GPU_CATS and e.get("dur", 0) > 0:
            gpu.append(e)
        elif cat in CPU_CATS and e.get("dur", 0) > 0:
            cpu.append(e)
    return gpu, cpu


# ---------------------------------------------------------------------------
# 2. Kernel taxonomy (sglang ordering, conv added first for TTS)
# ---------------------------------------------------------------------------

COMMUNICATION_STRONG = ("nccl", "allreduce", "all_reduce", "reduce_scatter",
                        "allgather", "all_gather", "alltoall", "all_to_all",
                        "cross_device_reduce", "deepep", "mooncake")
COMMUNICATION_WEAK = ("broadcast", "dispatch", "combine")
MEMORY_STRONG = ("memcpy", "memset", "dma", "prefetch")
MEMORY_WEAK = ("copy", "fill")
COMPUTE_HINT = ("gemm", "gemv", "matmul", "cublas", "cutlass", "wgmma", "mma",
                "bmm", "nvjet", "fmha", "attention", "flash_attn",
                "flashattention", "grouped_mm", "groupgemm", "moe", "expert")

# first-match-wins list; conv FIRST (TTS-specific: BigVGAN convs must not be
# misclassified as gemm)
CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("conv",          ("convolution", "convolve", "cudnn_conv", "_conv", "conv_transpose")),
    ("hybrid_linear", ("gdn", "gated_delta", "mamba", "selective_scan", "ssd",
                       "causal_conv", "ssm")),
    ("attention",     ("flash_attn", "flashattention", "flash_attention", "fmha",
                       "attention", "mla", "paged_attention", "decode_attention")),
    ("moe",           ("fused_moe", "grouped_mm", "groupgemm", "group_gemm", "moe",
                       "expert", "groupproblemshape")),
    ("gemm",          ("gemm", "gemv", "matmul", "cublas", "cutlass", "wgmma",
                       "mma", "bmm", "nvjet")),
    ("norm",          ("rmsnorm", "layernorm", "_norm_", " norm", "normkernel")),
    ("rope",          ("rotary", "rope", "mrope")),
    ("softmax",       ("softmax",)),
    ("activation",    ("silu", "gelu", "relu", "act_and_mul", "sigmoid")),
    ("quantize",      ("quant", "fp8", "mxfp", "nvfp4", "dequant", "cvt")),
    ("reduce",        ("topk", "argmax", "argtopk", "reduce", "sampling", "multinomial")),
    ("elementwise",   ("elementwise", "vectorized", "add_kernel", "sub_kernel",
                       "mul_kernel", "floor_kernel", "log_kernel", "neg_kernel",
                       "where", "clamp")),
]


def _contains_any(low: str, kws: tuple[str, ...]) -> bool:
    return any(k in low for k in kws)


def classify_kernel(name: str) -> str:
    """sglang precedence: comm-strong > mem-strong > [mem-weak unless
    compute-like] > CATEGORY_PATTERNS > comm-weak unless compute-like > other."""
    low = name.lower()
    if _contains_any(low, COMMUNICATION_STRONG):
        return "communication"
    if _contains_any(low, MEMORY_STRONG):
        return "memory"
    looks_compute = _contains_any(low, COMPUTE_HINT)
    if _contains_any(low, MEMORY_WEAK) and not looks_compute:
        return "memory"
    for cat, kws in CATEGORY_PATTERNS:
        if _contains_any(low, kws):
            return cat
    if _contains_any(low, COMMUNICATION_WEAK) and not looks_compute:
        return "communication"
    return "other"


# ---------------------------------------------------------------------------
# 3. Per-stage report primitives
# ---------------------------------------------------------------------------

def merged_intervals(gpu_events: list[dict]) -> list[list[Any]]:
    """vllm-omni sweep-line union of GPU busy intervals.

    Returns list of merged [start, end, [events...]] sorted by start.
    """
    rows = sorted(
        (e["ts"], e["ts"] + e["dur"], e["dur"], e.get("name", "?"),
         e.get("cat", ""), e.get("pid"), e.get("tid"))
        for e in gpu_events
    )
    merged: list[list[Any]] = []
    for start, end, dur, name, cat, pid, tid in rows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end, [(start, end, dur, name, cat, pid, tid)]])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append((start, end, dur, name, cat, pid, tid))
    return merged


def span_busy_idle(gpu_events: list[dict]) -> tuple[float, float, float]:
    """(span_us, busy_union_us, idle_us) over GPU events."""
    if not gpu_events:
        return 0.0, 0.0, 0.0
    merged = merged_intervals(gpu_events)
    span = merged[-1][1] - merged[0][0]
    busy = sum(m[1] - m[0] for m in merged)
    return span, busy, span - busy


def kernel_aggregates(gpu_events: list[dict]) -> dict[str, dict]:
    """Aggregate per canonical kernel name: total_us, count, max_us, category."""
    agg: dict[str, dict] = {}
    for e in gpu_events:
        nm = e.get("name", "?")
        d = e["dur"]
        a = agg.get(nm)
        if a is None:
            agg[nm] = dict(total_us=d, count=1, max_us=d, cat=classify_kernel(nm))
        else:
            a["total_us"] += d
            a["count"] += 1
            a["max_us"] = max(a["max_us"], d)
    return agg


def category_rollup(gpu_events: list[dict]) -> tuple[Counter, Counter]:
    """(cat_total_us, cat_launch_count)."""
    tot: Counter = Counter()
    cnt: Counter = Counter()
    for e in gpu_events:
        cat = classify_kernel(e.get("name", "?"))
        tot[cat] += e["dur"]
        cnt[cat] += 1
    return tot, cnt


# ---------------------------------------------------------------------------
# 4. Gap analysis — the differentiator
# ---------------------------------------------------------------------------

# (lo, hi) in microseconds; last bucket is open-ended
GAP_BUCKETS = [(0, 1), (1, 5), (5, 10), (10, 50), (50, 100),
               (100, 500), (500, 5000), (5000, float("inf"))]

# vllm-omni interesting_cpu filter
INTERESTING_SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize",
                          "cudaLaunch", "cudaMemcpy")


def gap_histogram(gpu_events: list[dict]) -> list[dict]:
    """Distribution of inter-kernel gaps (us). Reveals the micro-gap ocean
    (10-50us Python-dispatch) that a coarse --min-gap-ms filter would miss."""
    if len(gpu_events) < 2:
        return []
    starts = sorted(e["ts"] for e in gpu_events)
    ends = sorted(e["ts"] + e["dur"] for e in gpu_events)
    gaps = [starts[i] - ends[i - 1] for i in range(1, len(starts))]
    total = sum(g for g in gaps if g > 0)
    rows = []
    for lo, hi in GAP_BUCKETS:
        sel = [g for g in gaps if lo <= g < hi]
        rows.append(dict(bucket=f"[{lo},{hi})" if hi != float("inf") else f"[{lo},inf)",
                         count=len(sel), total_us=sum(sel),
                         pct=(sum(sel) / total * 100 if total else 0)))
    return rows


def _interesting_cpu(cpu_events: list[dict]) -> list[tuple]:
    """vllm-omni filter: dur>=1000us AND (python/user_annotation OR name has
    cudaSync/Launch/Memcpy). Returns (start, end, dur, name, cat)."""
    out = []
    for e in cpu_events:
        if e["dur"] < 1000:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        if cat in ("python_function", "user_annotation") or any(
            s in name for s in INTERESTING_SYNC_NAMES
        ):
            out.append((e["ts"], e["ts"] + e["dur"], e["dur"], name, cat))
    return out


def big_gaps(gpu_events: list[dict], cpu_events: list[dict],
             min_gap_us: int = 100, topn: int = 15) -> list[dict]:
    """vllm-omni GAP blocks: gaps between merged busy intervals >= min_gap_us,
    attributed to CPU containers overlapping the gap midpoint.

    Enhanced beyond vllm-omni: also reports the nearest interesting CPU event
    (by time distance) when no container overlaps the midpoint — many launch
    gaps are caused by a CPU event that *ends* right at gap start (CPU
    prepared the kernel, GPU went idle waiting), which midpoint containment
    misses.
    """
    if not gpu_events:
        return []
    merged = merged_intervals(gpu_events)
    interesting = _interesting_cpu(cpu_events)
    gaps = []
    for idx in range(1, len(merged)):
        gap_start = merged[idx - 1][1]
        gap_end = merged[idx][0]
        dur = gap_end - gap_start
        if dur < min_gap_us:
            continue
        prev_event = max(merged[idx - 1][2], key=lambda x: x[1])
        next_event = min(merged[idx][2], key=lambda x: x[0])
        mid = (gap_start + gap_end) / 2
        containers = [r for r in interesting if r[0] <= mid <= r[1]]
        containers = sorted(containers, key=lambda x: x[2])[:8]
        # nearest interesting CPU event by time distance to the gap
        nearest = None
        if interesting and not containers:
            def _dist(r):
                if r[1] <= gap_start:
                    return gap_start - r[1]  # CPU finished just before gap
                if r[0] >= gap_end:
                    return r[0] - gap_end   # CPU starts after gap
                return 0.0                   # overlaps the gap span
            nearest = min(interesting, key=_dist)
        gaps.append(dict(dur_us=dur, start=gap_start, end=gap_end,
                         prev=prev_event, next=next_event,
                         containers=containers, nearest=nearest))
    gaps.sort(key=lambda g: g["dur_us"], reverse=True)
    return gaps[:topn]


# ---------------------------------------------------------------------------
# 5. Overlap / bottleneck heuristics (sglang-style)
# ---------------------------------------------------------------------------

def overlap_analysis(gpu_events: list[dict]) -> dict[str, dict]:
    """Per-kernel-name exclusive/hidden accounting across streams.

    exclusive_us: time NOT overlapped by a co-active kernel on another stream.
    Single-stream traces (GPT decode under CUDA graph) show ~0 overlap — that
    is expected and informative (no overlap headroom exists).
    """
    if not gpu_events:
        return {}
    # per-stream timelines
    by_stream: dict[Any, list[tuple]] = defaultdict(list)
    for e in gpu_events:
        by_stream[e.get("stream", e.get("tid"))].append(
            (e["ts"], e["ts"] + e["dur"], e.get("name", "?"), e["dur"])
        )
    # total per-name duration across all streams
    total_by_name: Counter = Counter()
    for evts in by_stream.values():
        for _, _, name, d in evts:
            total_by_name[name] += d

    # for overlap: count, for each name, how much of its time is concurrent
    # with another kernel on a DIFFERENT stream.
    # Build a global busy map (per stream) then compute exclusive via sweep.
    names = list(total_by_name)
    # exclusive time per name: scan events of that name; a moment is exclusive
    # if no other-stream kernel is active at that moment.
    # To keep it tractable: build per-stream interval lists sorted.
    stream_lists = {s: sorted(evts) for s, evts in by_stream.items()}
    exclusive: Counter = Counter()
    for s, evts in stream_lists.items():
        others = [e for os_, evs in stream_lists.items() if os_ != s for e in evs]
        others = sorted(others)
        oi = 0
        # active other-stream events covering current time
        active_others: list = []
        # simple two-pointer over sorted (start) with heap of end
        import heapq
        active_heap: list = []  # (end, idx)
        for st, en, name, d in evts:
            # push all others starting before st
            while oi < len(others) and others[oi][0] <= st:
                heapq.heappush(active_heap, (others[oi][1], oi))
                oi += 1
            # pop expired
            while active_heap and active_heap[0][0] <= st:
                heapq.heappop(active_heap)
            # exclusive portion = d minus overlap with active others
            if not active_heap:
                exclusive[name] += d
            else:
                # overlap length limited by earliest other end
                overlap_end = active_heap[0][0]
                excl = max(0.0, min(en, overlap_end) - st) if overlap_end > st else 0.0
                exclusive[name] += excl
    out: dict[str, dict] = {}
    for nm in names:
        tot = total_by_name[nm]
        excl = exclusive.get(nm, 0.0)
        out[nm] = dict(total_us=tot, count=0, max_us=0,
                       exclusive_us=excl, hidden_us=tot - excl,
                       exclusive_ratio=(excl / tot if tot else 0.0),
                       hidden_ratio=(1 - excl / tot if tot else 0.0))
    # fill count/max from aggregates
    aggs = kernel_aggregates(gpu_events)
    for nm, a in aggs.items():
        if nm in out:
            out[nm]["count"] = a["count"]
            out[nm]["max_us"] = a["max_us"]
            out[nm]["cat"] = a["cat"]
    return out


# ---------------------------------------------------------------------------
# 6. Kernel -> source attribution (when with_stack=True captured)
# ---------------------------------------------------------------------------

# sglang-ish source_location_priority: own-framework/kernel-lib high,
# torch medium, py frames medium-low, noise negative.
def _frame_priority(fname: str) -> float:
    low = fname.lower()
    if "windextts" in low:
        return 300.0
    if "site-packages" in low:
        if "torch" in low or "cuda" in low:
            return 120.0
        return 200.0
    if "/usr/" in low or "lib/python" in low:
        return 20.0
    return 100.0


def source_attribution(gpu_events: list[dict], topn: int = 8) -> list[dict]:
    """Aggregate python_frames (when present in event args) per kernel name.
    Returns top sites ranked by weighted frame priority.

    NOTE: torch profiler's chrome export (this torch version) does not embed
    python_frames in kernel event args — the *working* source attribution path
    is the gap analysis: python_function events overlapping a gap midpoint are
    printed as ``in [python_function] file.py(line): func`` containers (see
    big_gaps). This function is kept for trace formats that do embed frames.
    """
    sites: dict[str, Counter] = defaultdict(Counter)  # kernel -> Counter(file:line)
    for e in gpu_events:
        frames = (e.get("args") or {}).get("python_frames")
        if not frames:
            continue
        for fr in frames:
            fname = fr.get("filename", "") if isinstance(fr, dict) else str(fr)
            line = fr.get("lineno", "") if isinstance(fr, dict) else ""
            key = f"{Path(fname).name}:{line}"
            sites[e.get("name", "?")][key] += _frame_priority(fname)
    out = []
    for nm, site_cnt in sites.items():
        total = sum(site_cnt.values())
        top_site = site_cnt.most_common(1)[0][0] if site_cnt else ""
        out.append(dict(kernel=nm, total=total, top_site=top_site,
                        sites=[(k, v) for k, v in site_cnt.most_common(5)]))
    out.sort(key=lambda r: -r["total"])
    return out[:topn]


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def analyze(label: str, events: list[dict], wall_ms: Optional[float] = None,
            min_gap_us: int = 100, topn: int = 15) -> dict:
    """Full per-stage report; prints a readable block and returns a summary dict."""
    gpu, cpu = split_events(events)
    if not gpu:
        print(f"\n[{label}] no GPU kernel events")
        return {}

    cat_tot, cat_cnt = category_rollup(gpu)
    total_gpu_us = sum(cat_tot.values())
    span_us, busy_us, idle_us = span_busy_idle(gpu)
    idle_pct = idle_us / span_us * 100 if span_us else 0
    tiny = [e for e in gpu if e["dur"] < 10]
    tiny_us = sum(e["dur"] for e in tiny)
    aggs = kernel_aggregates(gpu)
    ranked = sorted(aggs.items(), key=lambda kv: -kv[1]["total_us"])
    hist = gap_histogram(gpu)
    gaps = big_gaps(gpu, cpu, min_gap_us=min_gap_us, topn=topn)
    overlap = overlap_analysis(gpu)

    print(f"\n{'='*74}")
    if wall_ms is not None:
        print(f"[{label}]  profiler-wall={wall_ms:.0f}ms (DISTORTED — see note)  "
              f"GPU-work={total_gpu_us/1000:.0f}ms  idle={idle_pct:.1f}% "
              f"({idle_us/1000:.0f}ms bubble)  launches={len(gpu)}")
        print("  NOTE: profiler inflates wall time; profiler-free repeats are the "
              "authoritative latency.")
    else:
        print(f"[{label}]  GPU-work={total_gpu_us/1000:.0f}ms  "
              f"idle={idle_pct:.1f}% ({idle_us/1000:.0f}ms bubble)  launches={len(gpu)}")
    print(f"  tiny_kernels(<10us)={len(tiny)} ({tiny_us/1000:.1f}ms) "
          f"— launch-overhead signal")

    print("  -- category rollup (>=1%) --")
    for cat, us in cat_tot.most_common():
        pct = us / total_gpu_us * 100 if total_gpu_us else 0
        if pct >= 1.0:
            print(f"    {cat:14s} {us/1000:7.1f}ms  {pct:5.1f}%   ({cat_cnt[cat]} launches)")

    print("  -- top kernels (>=1%) --")
    for nm, a in ranked:
        pct = a["total_us"] / total_gpu_us * 100 if total_gpu_us else 0
        if pct < 1.0:
            break
        print(f"    [{a['cat']:11s}] {a['total_us']/1000:6.1f}ms {pct:5.1f}% "
              f"x{a['count']:4d}  {nm[:48]}")

    # gap histogram
    if hist:
        print("  -- gap distribution (the micro-gap ocean) --")
        tot_gap = sum(h["total_us"] for h in hist)
        print(f"    total idle {tot_gap/1000:.1f}ms over {sum(h['count'] for h in hist)} gaps")
        for h in hist:
            if h["count"]:
                print(f"    {h['bucket']:10s} {h['count']:6d} gaps  "
                      f"{h['total_us']/1000:8.2f}ms  {h['pct']:5.1f}%")

    # big gaps
    if gaps:
        print(f"  -- top gaps >= {min_gap_us}us (attributed) --")
        for g in gaps:
            print(f"    GAP {g['dur_us']/1000:.3f}ms ts={g['start']:.0f}->{g['end']:.0f}")
            pv = g["prev"]
            print(f"      prev [{pv[4]}] {pv[2]/1000:.3f}ms {pv[3][:70]}")
            nx = g["next"]
            print(f"      next [{nx[4]}] {nx[2]/1000:.3f}ms {nx[3][:70]}")
            for c in g["containers"]:
                print(f"      in   [{c[4]}] {c[2]/1000:.3f}ms {c[3][:70]}")
            if g.get("nearest") and not g["containers"]:
                c = g["nearest"]
                print(f"      near [{c[4]}] dur={c[2]/1000:.3f}ms {c[3][:70]}")
    else:
        print("  -- no gaps >= %dus --" % min_gap_us)

    # overlap summary
    if overlap:
        p1 = []
        p4 = []
        for nm, o in overlap.items():
            if o.get("count", 0) == 0:
                continue
            share = o["total_us"] / total_gpu_us * 100 if total_gpu_us else 0
            if share < 1.0:
                continue
            if o["exclusive_ratio"] >= 0.45 and o.get("cat") not in (
                "compute", "gemm", "attention", "conv"
            ):
                p1.append((nm, o["exclusive_ratio"], share))
            if o.get("cat") in ("elementwise", "memory") and o["hidden_ratio"] >= 0.65:
                p4.append((nm, o["hidden_ratio"], share))
        if p1 or p4:
            print("  -- overlap heuristics --")
            for nm, er, share in sorted(p1, key=lambda x: -x[2])[:5]:
                print(f"    P1 try-fusion/overlap: {nm[:40]} excl_ratio={er:.2f} "
                      f"share={share:.1f}%")
            for nm, hr, share in sorted(p4, key=lambda x: -x[2])[:5]:
                print(f"    P4 low-ROI-hidden:   {nm[:40]} hidden_ratio={hr:.2f} "
                      f"share={share:.1f}%")

    return dict(
        label=label, wall_ms=wall_ms, gpu_ms=total_gpu_us / 1000,
        idle_pct=idle_pct, idle_ms=idle_us / 1000, launches=len(gpu),
        tiny_kernels=len(tiny), tiny_ms=tiny_us / 1000,
        cat={c: cat_tot[c] / 1000 for c in cat_tot},
        top_kernels=[(nm, round(a["total_us"] / 1000, 2), round(
            a["total_us"] / total_gpu_us * 100, 1) if total_gpu_us else 0.0,
            a["count"]) for nm, a in ranked[:15]],
        gap_hist=[dict(bucket=h["bucket"], count=h["count"],
                       ms=round(h["total_us"] / 1000, 2), pct=round(h["pct"], 1))
                  for h in hist if h["count"]],
        n_big_gaps=len(gaps),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="WIndexTTS trace analyzer (pure "
                                             "Python, no torch)")
    ap.add_argument("trace", help="chrome trace .json or .json.gz")
    ap.add_argument("--min-gap-us", type=int, default=100)
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    events = load_trace(args.trace)
    label = args.label or Path(args.trace).stem
    print(f"loaded {len(events)} X-events from {args.trace}")
    analyze(label, events, min_gap_us=args.min_gap_us, topn=args.topn)


if __name__ == "__main__":
    main()
