# WIndexTTS Profiler System — Implementation Spec

## Goal
A professional, evidence-driven profiler for the WIndexTTS pipeline. Every
optimization decision must be backed by profile evidence; "looks faster but is
slower" cases must be explainable from a trace.

## Architecture (one module, importable + CLI)

`windextts/profiler.py` — pure analysis, no model loading. Reusable.
`scripts/profile_windex_stages.py` — the capture harness (loads model, captures
  traces via torch.profiler, writes .json.gz, then calls windextts.profiler).
`scripts/profile_ab.py` — profiler-free A/B harness (≥3 repeats, the authority
  on wall-clock latency; profiler numbers are diagnostic only).

## The analyzer (windextts/profiler.py) — required features

### 1. Trace loading
- Open .json or .json.gz. Handle both raw event list and {traceEvents: [...]}.
- Keep only ph=="X" events with numeric dur>0.
- Separate: GPU events (cat in {kernel, gpu_memcpy, gpu_memset}), CPU events
  (cat in {cpu_op, user_annotation, python_function, cuda_runtime, cuda_driver}).

### 2. Kernel taxonomy (from sglang, adapted for TTS)
Ordered first-match-wins classify_kernel(name):
  communication-strong > memory-strong > [memory-weak unless compute-like] >
  CATEGORY_PATTERNS list > communication-weak unless compute-like > other.
Keyword tuples (verbatim from sglang, see research notes). ADD a `conv` category
 FIRST in CATEGORY_PATTERNS (TTS-specific: cudnn_conv, conv_transpose) since
 conv is huge in BigVGAN and must not be misclassified as gemm.

### 3. Per-stage report
For a given event list (one stage's trace):
  - wall_ms (caller-provided, separate from GPU work — IMPORTANT: profiler
    distorts wall time, so report it but label clearly)
  - gpu_total_ms (sum of GPU kernel dur)
  - idle_pct + idle_ms: merged-interval (span - busy_union)/span, the vllm-omni
    algorithm. Quote: sort GPU by start; merge overlapping (start>cur_end → new
    interval; else extend end=max); span=last_end-first_start; busy=sum(union).
  - tiny_kernels count+dur (<10us) — launch-overhead signal.
  - category rollup (>=1% only): category, ms, pct, launches.
  - top kernels (>=1%): name (truncated 48), category, ms, pct, count.
  - PRINT all of the above as a readable block.

### 4. Gap analysis — THE KEY DIFFERENTIATOR
Two complementary views (the research proved both are needed):

(a) Gap DISTRIBUTION histogram — because S2Mel's idle is 75% "micro-gaps of
    10-50us" (Python dispatch), NOT a few big gaps. vllm-omni's --min-gap-ms 5
    would MISS this entirely. Buckets (us): [0,1),[1,5),[5,10),[10,50),
    [50,100),[100,500),[500,5000),[5000+]. Print count, total_ms, pct per bucket.
    This is what reveals "CUDA-Graph-eliminatable micro-dispatch overhead".

(b) Big-gap attribution (vllm-omni style) — for gaps >= a threshold (default
    100us, NOT 5ms), find the merged-interval gap, print prev/next GPU event,
    and the CPU containers overlapping the gap midpoint (the interesting_cpu
    filter: dur>=1000us AND (cat in python/user_annotation OR name contains
    cudaStreamSynchronize/cudaDeviceSynchronize/cudaLaunch/cudaMemcpy).
    Containers sorted by dur asc, top 8.

### 5. Overlap / bottleneck heuristics (from sglang)
For each kernel aggregate, compute:
  - total_us, count, max_us
  - exclusive_us (time NOT overlapped by any co-active kernel on another stream)
  - hidden_us (time overlapped/hidden under a heavier kernel)
  - overlap_with (Counter of which kernel dominates the overlap)
Then classify priority:
  - share_pct < 1.0 → P5 skip
  - headroom(exclusive_ratio>=0.45) + non-compute → P1 "try fusion/overlap"
  - low-roi-hidden: category in {elementwise,memory} + hidden_ratio>=0.65 → P4 skip
  - else by dependency.
NOTE: this requires per-stream co-activity accounting. Single-stream traces
(GPT decode under CUDA graph) will show little overlap — that's expected and
informative.

### 6. Kernel→source attribution (with_stack)
When the trace was captured with with_stack=True, parse python_frames in args
to attribute each kernel to a source file:line (sglang source_location_priority:
own-framework high, kernel-lib high, torch low, py medium, noise negative).
Rank the python stack frames for each kernel aggregate's top site.

## The capture harness (scripts/profile_windex_stages.py)
- warmup BEFORE capture (prime cudnn/autotune/cgraph).
- Capture e2e + per-stage (via monkey-patch boundary capture, already working).
- Always run BOTH: (1) torch.profiler trace for diagnostics, (2) profiler-free
  wall-clock ≥3 repeats for the authoritative latency. Print both, clearly
  labeled (the research: "profiler latency is diagnostic only").
- Write traces to /root/windextts_dumps/profiles/<label>.json.gz.

## The A/B harness (scripts/profile_ab.py)
- Two configs (e.g. fp32 vs bf16, eager vs graph). Each: warmup, then ≥3
  profiler-free repeats. Report mean/median/min/max + student-t-ish check.
- Pair with a profiler capture of each for the diff explanation.
- This is the FINAL word on whether an optimization helps.

## Verification
- Analyzer is unit-testable: feed it the existing
  /root/windextts_dumps/profiles/stage_s2mel.json.gz and it must reproduce the
  numbers (idle ~48%, gemm 57%, etc.) — regression guard.
- All analysis is pure-Python over the trace JSON; no torch import needed in
  the analyzer itself (testable anywhere).
