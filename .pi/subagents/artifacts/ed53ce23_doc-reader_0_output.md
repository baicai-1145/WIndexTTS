I've read the full SKILL.md, both analysis scripts, `profile_common.py`, both triage helper modules, and the reference docs. Here is the reusable methodology.

## 1. Profiling workflow (warmup → capture → analyze)

The skill has one public workflow (`triage`) with two shapes: **single-trace** and **two-trace (mapping + formal)**.

**Defaults (from `profile_common.py` and `analyze_llm_torch_profile.py`):**
- `DEFAULT_WARMUP_STEPS = 10`, `--num-steps` default `5`, `--probe-delay` default `0.5` (live).
- Stage-separated workload contract: prefill `input 4090 / output 1`, decode `input 1 / output 2048` (`DEFAULT_PREFILL_INPUT_LEN=4090`, `DEFAULT_PREFILL_OUTPUT_LEN=1`, `DEFAULT_DECODE_INPUT_LEN=1`, `DEFAULT_DECODE_OUTPUT_LEN=2048`). Captured into **separate `prefill/` and `decode/` output dirs** so tables stay comparable across frameworks.

**Capture sequence (live, in `profile_common.py`):**
1. **Warmup:** send warmup probe requests *before* arming the profiler (`probe_plan.warmup_requests = warmup_steps`; for decode, warmup = 1 request with `max_new_tokens = warmup_steps`). Rationale: warm the allocator/cuda-graph/weights so capture isn't polluted by cold-start.
2. **Arm:** `POST /start_profile` (payload: `output_dir`, `num_steps`, `activities:["CPU","GPU"]`, `with_stack`, `record_shapes`, `profile_id`; SGLang also `profile_by_stage`, `merge_profiles`).
3. **Delay:** `time.sleep(max(5.0, probe_delay))` before sending active probes — server-side profilers do setup after `/start_profile`, so early probes miss the window.
4. **Capture:** send active probes with `unique_probe_prompt(...)` (uniqueness avoids prefix-cache pollution in prefill) and `sampling_seed` offsets after warmup.
5. **Stop + wait:** `POST /stop_profile`, then `wait_for_profiler_artifact(..., timeout_s=180.0)` — SGLang trace flush can lag well beyond seconds. New trace files moved into `prefill/` / `decode/` stage dirs.

**Analysis sequence (single pass in `analyze_llm_torch_profile.py`):**
- `extract_trace_data(trace)` → kernels / cpu_ops / python_frames / launch_events / chosen_pid / window_us.
- `group_kernels_by_stage` → `aggregate(... key_fn=canonical_name)` → `build_kernel_rows` sorted by `total_us` desc → `detect_fusion_opportunities` → render.
- Read results in order: **kernel table → overlap-opportunity table → fuse-pattern table**. Render cutoff: all three tables only show rows at/above **`MIN_RENDER_SHARE_PCT = 1.0`** cumulative GPU-time share.

**Two-trace shape (for stronger overlap/source attribution):** mapping trace with graph disabled/low-fusion first (recovers `kernel → cpu_op → python scope`), then formal trace with real optimizations on. Mapping sampling: `MAPPING_KERNEL_SAMPLE_LIMIT_PER_NAME = 16` evenly-spaced events per canonical kernel (`sample_kernels_for_mapping`). The mapping pass keeps sampling disabled so it doesn't perturb output; matching is prefix/alias/`common_prefix_len>=64` fuzzy (kernel_helpers relaxed lookup).

## 2. Metrics extracted from the trace

The scripts read raw `ph=="X"` complete-duration events and build aggregates; they **do not** parse `self_cuda_time`/`self_cuda_time_total` from ChromeTrace JSON. The only duration they use is the kernel's `dur` (µs) and `ts`.

**Kernel table columns** (`render_kernel_table_for_stage`):
`| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |` → fields `kernel`, `category`, `total_us` (formatted `ms`), `share_pct`, `launches` (`Aggregate.count`), `location`, `cpu_op`.

**Aggregate (`Aggregate` dataclass):** `total_us` (Σ dur), `count` (launch count), `max_us`, plus `avg_us` property.

**Per-site attribution (`MappingSiteAggregate` / stage payload):** `location`, `display_location`, `launches`, `total_us`, `share_pct_within_kernel`, `top_cpu_op`, `stack`. Source locations ranked by `source_location_priority` (own-framework 300→280, kernel lib 260, torch 20, `py:` 120, noise frames −20, low-signal frames −80).

**Overlap metrics (per kernel aggregate, `AggregateStats`):**
- `hidden_us`, `exclusive_us`, `hidden_by_compute_us`, `overlap_with` (Counter of dominant co-active kernel).
- Derived ratios: `exclusive_ratio = exclusive_us/total_us`, `hidden_ratio = hidden_us/total_us`, plus `mapping_ratio = mapped_count/total_count`.
- Trace-level: `total_busy_us`, `total_overlap_us`, `max_concurrent_streams`.

**CPU-GPU relationship:** There is **no CPU-GPU gap table**. CPU ops are matched to kernels via `External id` / `correlation` (`build_correlation_external_lookup`), then Python frames active at the launch time give the source scope. `gap_us` is used only as a **same-stream dependency signal** between adjacent kernels (see §4).

**Explicitly NOT computed:** `self_cuda_time`, occupancy, arithmetic intensity, launch-overhead fraction. (Confirmed via grep: no `occupancy`, `self_cuda`, `arithmetic`, `intensity` in any script.)

## 3. Kernel classification / taxonomy

Two taxonomies live in the two triage helpers.

**Fine-grained (`triage_kernel_helpers.py:classify_kernel` + `CATEGORY_PATTERNS`), order matters:**
1. `communication` — strong keywords `nccl, allreduce, all_reduce, reduce_scatter, allgather, all_gather, alltoall, all_to_all, cross_device_reduce, deepep, mooncake`; weak `broadcast, dispatch, combine` (weak only counts if not compute-like).
2. `memory` — strong `memcpy, memset, dma, prefetch`; weak `copy, fill` (weak only if not compute-like).
3. `CATEGORY_PATTERNS` list (first match wins):
   - `hybrid_linear`: `gdn, gated_delta, mamba, selective_scan, ssd, causal_conv, ssm`
   - `attention`: `flash_attn, flashattention, flash_attention, fmha, attention, mla, paged_attention, decode_attention`
   - `moe`: `fused_moe, grouped_mm, groupgemm, group_gemm, moe, expert, groupproblemshape`
   - `gemm`: `gemm, gemv, matmul, cublas, cutlass, wgmma, mma, bmm, nvjet`
   - `norm`: `rmsnorm, layernorm, _norm_, " norm", normkernel`
   - `rope`: `rotary, rope, mrope`
   - `softmax`: `softmax`
   - `activation`: `silu, gelu, relu, act_and_mul, sigmoid`
   - `quantize`: `quant, fp8, mxfp, nvfp4, dequant, cvt`
   - `reduce_topk`: `topk, reduce, argmax, argtopk, sampling, multinomial`
   - `sampling_io`: `prepare_inputs, write_req_to, catarraybatched, prepare_next, copy_next`
   - `elementwise`: `elementwise, vectorized_elementwise_kernel, ..., add_kernel, sub_kernel, mul_kernel, floor_kernel, log_kernel, neg_kernel`
4. else `"other"`.

**Broad overlap buckets (`triage_overlap_helpers.py:classify_kernel`):** `compute` (gemm/attention/cutlass/cublas), `communication`, `elementwise` (sigmoid/topk/gate/rmsnorm/layernorm/rope/casts), `memory`, `other`. Used only for prioritization via weights `communication:1.3, memory:1.15, elementwise:1.0, other:0.8, compute:0.35`.

There is **no explicit "launch-overhead" category**. Launch overhead is handled indirectly through launch counts + the fuse catalog's low-share "Fused decode metadata setup" family + dependency gaps.

## 4. "Is this kernel a bottleneck?" heuristics

- **Share bar:** rows below `1.0%` GPU share are hidden by default (`MIN_RENDER_SHARE_PCT`).
- **Overlap headroom candidate** (`top_overlap_opportunities`): `total_us >= 5.0` and `exclusive_ratio >= 0.45`; non-`compute` prioritized first.
- **Low-ROI hidden** (`top_hidden_low_roi`): category in `{elementwise, memory}` and `total_us >= 5.0` and `hidden_ratio >= 0.65` → "do not chase this first; focus on fusion/launch reduction/schedule".
- **Priority + recommendation** (`build_priority_and_recommendation`, with `share_pct < 1.0 → P5/skip`):
  - `headroom` + dep `low` + `communication` → `P1 "try overlap"`; + non-comm → `P1 "try fusion"`.
  - `headroom` + dep high → `P2 "check deps"`.
  - `exclusive_ratio >= 0.85` + dep `low` → `P3 "defer"`.
  - `low-roi-hidden` → `P4 "skip"`.
  - `hidden_ratio >= 0.7` → `P5 "skip"`; dep high → `P4 "check deps"`; dep unclear → `P4 "inspect"`; else `P4 "defer"`.
- **Dependency signal (compute vs serial):** nearest previous/next kernels on the **same stream**; `tight_gap_threshold = max(2.0, min(20.0, current.dur * 0.15))` µs. Gap ≤ threshold ⇒ tight ⇒ `prev-side / next-side / both-side serial risk` (labels: `low`, `high`, `unclear`). This is a heuristic, "not proof of dataflow."
- **Compute-bound vs memory-bound:** NOT measured arithmetically (no arithmetic intensity, no occupancy). It is only *implied* by category (`compute` vs `elementwise`/`memory`) and by `hidden_by_compute_us` (hidden-under-compute is treated as lower ROI). Launch-vs-duration tension is surfaced via the fuse table's low-share metadata patterns and via high `count` on short kernels.
- **Fusion detection thresholds** (`FusionPatternSpec`): per-pattern `min_share` (0.02–0.5) and `likely_share` (0.2–4.0). Confidence `"Confirmed"` if the fused kernel is active **or** related share ≥ `likely_share`, else `"Candidate"`. Match = active keywords present, or every split-group matched, and (if no active match) `pct(related_us,total_us) >= min_share`. Rows deduped by covered kernel identity and `subsumes` blocking.

## 5. Separating prefill vs decode (stage split)

Three mechanisms, all used together:
- **Path/directory labels** (`parse_stage`): markers `-extend, -prefill, _extend, _prefill` or `/prefill/`/`/extend/` → `extend`; `-decode, _decode` or `/decode/` → `decode`; else `all`. Live capture forces separate `prefill/` and `decode/` dirs.
- **Trace annotations** (`build_stage_annotations`): reads events with `cat` in `{user_annotation, gpu_user_annotation}`; `generation_0`/`prefill` → `extend`, `generation_1`/`decode` → `decode`. Kernels resolved via `External id`, else nearest enclosing GPU/CPU stage window with tolerance `nearest_gap <= 20_000.0` µs (`resolve_kernel_stage`).
- **Render grouping** (`STAGE_ORDER = {"extend":0,"prefill":0,"decode":1,"all":2}`): each table is split into `##### extend/prefill` and `##### decode` sections when a clean split exists (`stage_label("extend") → "extend/prefill"`; `stage_aliases` for fallback lookup). TP-rank selection via `select_heaviest_pid(... preferred_substrings=("TP00","TP-0"))` and `parse_tp_rank`.

**Reusable takeaway for WIndexTTS (pure-torch TTS):** the transferable methodology is (a) warmup-then-arm-then-capture with stage-separated dirs; (b) aggregate GPU kernels by canonical name → `total_us`/`count`/`share_pct`; (c) the ordered keyword taxonomies (`gemm/gemv/matmul/cublas/cutlass/wgmma/bmm`, `norm`, `rope`, `softmax`, `activation`, `elementwise`, `memory`) map cleanly to TTS ops (conv, conformer, attention, norms, mel fbank); (d) the `exclusive_ratio >= 0.45` / `hidden_ratio >= 0.65` + same-stream gap `max(2, 0.15·dur)` dependency heuristic is directly reusable; (e) the render cutoff `≥1.0%` share. You'll need to add arithmetic-intensity/occupancy extraction yourself — this skill intentionally omits it.

---
**Residual risks to carry into your own tool:** the skill assumes `cat`/`args` shapes of LLM-serving traces (e.g. `cat=="gpu_kernel"`, `External id`, `correlation`, `stream`/`cuda_stream` args); a TTS trace from `torch.profiler` may use slightly different `args`/`cat` markers, so kernel-event detection (`is_gpu_kernel_event`) and the correlation→external-id mapping are the fragile parts to re-verify first.