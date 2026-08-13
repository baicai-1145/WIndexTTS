I have all the material. Here is the implementable extraction.

---

## 1. `classify_kernel()` + `CATEGORY_PATTERNS` (kernel-attribution file)

**File:** `/root/sglang/.claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py`, `classify_kernel` at line 1371, `CATEGORY_PATTERNS` at line 29.

`classify_kernel` (L1371-1392):
```python
def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if contains_any_keyword(lowered, COMMUNICATION_STRONG_KEYWORDS):
        return "communication"
    if contains_any_keyword(lowered, MEMORY_STRONG_KEYWORDS):
        return "memory"
    looks_compute_like = contains_any_keyword(lowered, COMPUTE_HINT_KEYWORDS)
    if contains_any_keyword(lowered, MEMORY_WEAK_KEYWORDS) and not looks_compute_like:
        return "memory"
    for category, keywords in CATEGORY_PATTERNS:
        if contains_any_keyword(lowered, keywords):
            return category
    if (
        contains_any_keyword(lowered, COMMUNICATION_WEAK_KEYWORDS)
        and not looks_compute_like
    ):
        return "communication"
    return "other"
```
**Order of precedence (first-match-wins):** communication-strong → memory-strong → memory-weak (unless compute-like) → `CATEGORY_PATTERNS` in list order → communication-weak (unless compute-like) → `"other"`.

Keyword tuples (attribution file):
- `COMMUNICATION_STRONG_KEYWORDS` (L130): `nccl, allreduce, all_reduce, reduce_scatter, allgather, all_gather, alltoall, all_to_all, cross_device_reduce, deepep, mooncake`
- `COMMUNICATION_WEAK_KEYWORDS` (L141): `broadcast, dispatch, combine`
- `MEMORY_STRONG_KEYWORDS` (L146): `memcpy, memset, dma, prefetch`
- `MEMORY_WEAK_KEYWORDS` (L152): `copy, fill`
- `COMPUTE_HINT_KEYWORDS` (L157): `gemm, gemv, matmul, cublas, cutlass, wgmma, mma, bmm, nvjet, fmha, attention, flash_attn, flashattention, grouped_mm, groupgemm, moe, expert`

**`CATEGORY_PATTERNS` FULL ordered list (L29-128), preserving first-match order:**
1. `hybrid_linear` → `gdn, gated_delta, mamba, selective_scan, ssd, causal_conv, ssm`
2. `attention` → `flash_attn, flashattention, flash_attention, fmha, attention, mla, paged_attention, decode_attention`
3. `moe` → `fused_moe, grouped_mm, groupgemm, group_gemm, moe, expert, groupproblemshape`
4. `gemm` → `gemm, gemv, matmul, cublas, cutlass, wgmma, mma, bmm, nvjet`
5. `norm` → `rmsnorm, layernorm, _norm_, " norm", normkernel`
6. `rope` → `rotary, rope, mrope`
7. `softmax` → `softmax`
8. `activation` → `silu, gelu, relu, act_and_mul, sigmoid`
9. `quantize` → `quant, fp8, mxfp, nvfp4, dequant, cvt`
10. `reduce_topk` → `topk, reduce, argmax, argtopk, sampling, multinomial`
11. `sampling_io` → `prepare_inputs, write_req_to, catarraybatched, prepare_next, copy_next`
12. `elementwise` → `elementwise, vectorized_elementwise_kernel, unrolled_elementwise_kernel, gpu_kernel_impl, binary_internal, unaryfunctor, add_kernel, sub_kernel, mul_kernel, div_, floor_kernel, log_kernel, neg_kernel`

Note: `" norm"` (with leading space) is a literal substring match, not a regex. Matching uses `contains_any_keyword(lowered, keywords)` (case-insensitive substring).

There is also a **second, different** `classify_kernel` in the overlap file (`triage_overlap_helpers.py:347`) that uses **coarser buckets** (memory/communication/compute/elementwise/other). Its precedence: memory-strong → communication-strong → compute → elementwise → memory-weak (unless compute-like) → communication-weak (unless compute-like) → `"other"`. Keyword sets for it are at L30-133:
- `COMMUNICATION_STRONG`: `allreduce, all_reduce, reduce_scatter, allgather, all_gather, nccl, cross_device_reduce, deepep, a2a, alltoall, allreduce_fusion, mooncake`
- `COMMUNICATION_WEAK`: `broadcast, dispatch, combine`
- `MEMORY_STRONG`: `memcpy, memset, dma, prefetch`
- `MEMORY_WEAK`: `fill, copy`
- `ELEMENTWISE_KEYWORDS`: `sigmoid, silu, gelu, relu, softmax, layernorm, rmsnorm, norm, rotary, rope, topk, gate, bias, _cast, index, gather, scatter, masked, elementwise, activation`
- `COMPUTE_KEYWORDS`: `cublas, cudnn, cutlass, triton, gemm, gemv, matmul, grouped_mm, flash, attention, fmha, marlin, fused_moe, moe_kernel, groupgemm, mma, wgmma, conv, bmm, mm_kernel`
- `CATEGORY_PRIORITY` (L134): `compute:4, communication:3, memory:2, elementwise:1, other:0`

**Note for re-implementation:** the bottleneck math and `build_priority_and_recommendation` live in `triage_overlap_helpers.py` and use the **coarse** categories (`compute`, `communication`, `memory`, `elementwise`, `other`), NOT the 12-way attribution categories.

---

## 2. `build_priority_and_recommendation()` (overlap file, L1620-1650)

```python
def build_priority_and_recommendation(
    verdict: str,
    category: str,
    dependency_signal: str,
    stats: AggregateStats,
    share_pct: float,
) -> Tuple[str, str]:
    dep_label = dependency_risk_label(dependency_signal)
    if share_pct < 1.0:
        return "P5", "skip"

    if verdict == "headroom":
        if dep_label == "low":
            if category == "communication":
                return "P1", "try overlap"
            return "P1", "try fusion"
        return "P2", "check deps"

    if verdict == "low-roi-hidden":
        return "P4", "skip"

    if stats.exclusive_ratio >= 0.85 and dep_label == "low":
        return "P3", "defer"
    if stats.hidden_ratio >= 0.7:
        return "P5", "skip"
    if dep_label == "high":
        return "P4", "check deps"
    if dep_label == "unclear":
        return "P4", "inspect"
    return "P4", "defer"
```
Branch table:
- `share_pct < 1.0` → `("P5", "skip")`
- verdict `"headroom"` + dep low + category `communication` → `("P1", "try overlap")`
- verdict `"headroom"` + dep low + any other category → `("P1", "try fusion")`
- verdict `"headroom"` + dep not-low → `("P2", "check deps")`
- verdict `"low-roi-hidden"` → `("P4", "skip")`
- `exclusive_ratio >= 0.85` and dep low → `("P3", "defer")`
- `hidden_ratio >= 0.7` → `("P5", "skip")`
- dep `"high"` → `("P4", "check deps")`
- dep `"unclear"` → `("P4", "inspect")`
- else → `("P4", "defer")`

Note: the thresholds `exclusive_ratio >= 0.45` and `hidden_ratio >= 0.65` from the task are **not** in `build_priority_and_recommendation`. Inside it, the thresholds are `exclusive_ratio >= 0.85` and `hidden_ratio >= 0.7`. The 0.45/0.65 thresholds are the *candidate filters* in `top_overlap_opportunities` / `top_hidden_low_roi` (section 4).

**Dependency signal** (`classify_dependency_signal`, L1521-1607; label mapping at L1609-1618):
```python
tight_gap_threshold = max(2.0, min(20.0, current.dur * 0.15))
prev_tight = prev_gap is not None and prev_gap <= tight_gap_threshold
next_tight = next_gap is not None and next_gap <= tight_gap_threshold
```
- `prev_gap = max(current.ts - prev_event.end, 0.0)`; `next_gap = max(next_event.ts - current.end, 0.0)`.
- `prev_risk = prev_tight and (same_scope_family(current_scope, prev_scope) or (current_launch != "n/a" and current_launch == prev_launch) or is_neighbor_dependency_like(current, prev_event))`. `next_risk` analogous.
- `prev_unclear = prev_tight and not prev_risk and (current_scope == "unmapped" or prev_scope == "unmapped")`; `next_unclear` analogous.
- Signal: both risks → `"both-side serial risk"`; prev → `"prev-side serial risk"`; next → `"next-side serial risk"`; else if any unclear → `"adjacency unclear"`; else `"serial risk low"`.
- `dependency_risk_label`: `"serial risk low"→"low"`, `"prev-side/next-side/both-side serial risk"→"high"`, `"adjacency unclear"→"unclear"`.

Helpers:
- `same_scope_family` (L1464): `parse_scope_signature` splits `"path(line): func"` (regex `(.+?)\(\d+\):\s*(.+)$`); returns True if both paths equal, else if both funcs equal and non-empty.
- `is_neighbor_dependency_like` (L1474): if current is `communication` → neighbor in `{compute, elementwise, memory, other}`; if current in `{elementwise, memory}` → neighbor in `{compute, communication, elementwise, memory}`; else False.
- Neighbors come from `build_stream_neighbor_index` (L1481): per-stream sort by `(ts, end, idx)`; each event's neighbors are the immediately adjacent events on the *same stream*.

---

## 3. Per-kernel `exclusive_us` / `hidden_us` / `overlap_with` accounting

**Sweep line in `analyze_overlap` (`triage_overlap_helpers.py:566-616`).** Core loop (L582-611):
```python
    for time_point, is_start, event_idx in points:
        if prev_time is not None and time_point > prev_time and active:
            segment = time_point - prev_time
            active_events = list(active.values())
            distinct_streams = {event.stream for event in active_events}
            total_busy += segment
            max_concurrent = max(max_concurrent, len(distinct_streams))
            if len(distinct_streams) >= 2:
                total_overlap += segment
            for event in active_events:
                overlapping_events = [
                    other
                    for other in active_events
                    if other.idx != event.idx and other.stream != event.stream
                ]
                if overlapping_events:
                    event.hidden_us += segment
                    if any(other.category == "compute" for other in overlapping_events):
                        event.hidden_by_compute_us += segment
                    overlap_name = dominant_overlap_name(event, active_events)
                    if overlap_name:
                        event.overlap_with[overlap_name] += segment
                else:
                    event.exclusive_us += segment

        if is_start == 0:
            active.pop(event_idx, None)
        else:
            active[event_idx] = event_map[event_idx]
        prev_time = time_point
```
Semantics:
- Points are `(ts, 1, idx)` start / `(end, 0, idx)` end, sorted by `(time, is_start)` (start before end at ties).
- Between consecutive distinct times with non-empty `active`, a time `segment` is charged. For each active event:
  - If it overlaps any active event **on a different stream** (`overlapping_events` non-empty) → that segment is **hidden_us**; if any of those different-stream events is category `compute`, also added to **hidden_by_compute_us**; the `dominant_overlap_name`'s kernel name gets +segment in `overlap_with`.
  - Else (no different-stream overlap) → segment is **exclusive_us**.
- A kernel is therefore `exclusive` when nothing on another stream runs concurrently with it; it is `hidden` when other streams are concurrently busy.
- `dominant_overlap_name` (L549): among different-stream candidates, sort by `(CATEGORY_PRIORITY[category], dur)` descending and take the top `canonical_name`.

**Aggregation into `AggregateStats` (`aggregate_events`, L621-641):** key = `(canonical_name, category)`; sums `count, total_us, hidden_us, exclusive_us, hidden_by_compute_us`; `overlap_with.update(...)`; keeps `representative_idx` = the event maximizing `hidden_us + exclusive_us`. Ratios are properties (L210-215):
```python
hidden_ratio    = hidden_us / total_us if total_us else 0.0
exclusive_ratio = exclusive_us / total_us if total_us else 0.0
```

---

## 4. `top_overlap_opportunities()` and `top_hidden_low_roi()` (`triage_overlap_helpers.py`)

`top_overlap_opportunities` (L666-692):
```python
def top_overlap_opportunities(aggregates):
    category_weight = {
        "communication": 1.3,
        "memory": 1.15,
        "elementwise": 1.0,
        "compute": 0.35,
        "other": 0.8,
    }
    candidates = [
        stats for stats in aggregates.values()
        if stats.total_us >= 5.0 and stats.exclusive_ratio >= 0.45
    ]
    primary = [stats for stats in candidates if stats.category != "compute"]
    fallback = [stats for stats in candidates if stats.category == "compute"]
    primary.sort(key=lambda stats: stats.exclusive_us * category_weight.get(stats.category, 1.0), reverse=True)
    fallback.sort(key=lambda stats: stats.exclusive_us * category_weight.get(stats.category, 1.0), reverse=True)
    return (primary + fallback)[:5]
```
Filter: `total_us >= 5.0` **and** `exclusive_ratio >= 0.45`. Category is not filtered here, but non-compute (primary) rank above compute (fallback); scoring = `exclusive_us * category_weight`.

`top_hidden_low_roi` (L645-663):
```python
def top_hidden_low_roi(aggregates):
    candidates = [
        stats for stats in aggregates.values()
        if stats.category in {"elementwise", "memory"}
        and stats.total_us >= 5.0
        and stats.hidden_ratio >= 0.65
    ]
    candidates.sort(
        key=lambda stats: (
            stats.hidden_us * (1.0 + stats.hidden_by_compute_us / max(stats.hidden_us, 1.0)),
            stats.hidden_ratio,
        ),
        reverse=True,
    )
    return candidates[:5]
```
Filter: `category ∈ {elementwise, memory}` **and** `total_us >= 5.0` **and** `hidden_ratio >= 0.65`. Sort key = `hidden_us * (1 + hidden_by_compute_us/hidden_us)`, then `hidden_ratio`, descending.

**Verdicts:** rows from `top_overlap_opportunities` get verdict `"headroom"`; rows from `top_hidden_low_roi` get verdict `"low-roi-hidden"` (assignments at L1727 and L1745). In the action table builder, headroom rows with `priority == "P5"` are skipped, and low-roi rows are skipped if already seen by name.

---

## 5. Fusion detection (`triage_kernel_helpers.py`)

`FusionPatternSpec` (L352-366) fields:
```python
class FusionPatternSpec:
    pattern: str
    candidate_path: str
    active_keywords: Tuple[str, ...] = ()
    split_groups: Tuple[Tuple[str, ...], ...] = ()
    rationale_hint: str = ""
    origin: str = "mainline"          # mainline | upstream | (pending)
    model_include: Tuple[str, ...] = ()
    model_exclude: Tuple[str, ...] = ()
    min_tp_size: int = 1
    require_tp: bool = False
    min_share: float = 0.25
    likely_share: float = 3.0
    priority: int = 0
    subsumes: Tuple[str, ...] = ()
```
Defaults: `min_share=0.25`, `likely_share=3.0`. The registry `FUSION_PATTERN_REGISTRY` (L369+) contains ~40 specs, most overriding `min_share`/`likely_share`. Example specs:
- **"Fused residual add + RMSNorm"** (`min_share=0.1`, `likely_share=1.0`): `active_keywords=("fused_add_rmsnorm","gemma_fused_add_rmsnorm","npu_add_rms_norm","add_rmsnorm_bias")`, candidate path `python/sglang/srt/layers/layernorm.py<br>.../modelslim.py`.
- **"FlashInfer unified allreduce_fusion"** (L389): next in registry.
- (a full separate match with `min_share=0.5, likely_share=4.0` at L415 "matmul" family; `min_share=0.02, likely_share=0.2` at L546; `min_share=0.4, likely_share=2.0` at L572/613/696; `min_share=0.3, likely_share=1.5` at L593/633/718/745; `min_share=0.05, likely_share=0.5` at L531/679/792; `min_share=0.02, likely_share=0.2` at L546/805; etc.)

**Detection** (`detect_pattern_match`, L2737-2797):
- Early exits: `total_us <= 0`, framework unsupported (`pattern_supports_framework`), `spec.require_tp and tp_size < spec.min_tp_size`, model excluded (`pattern_model_matches`).
- `active_rows = matching_rows_for_keywords(kernel_rows, spec.active_keywords)`; `split_groups = [matching_rows_for_keywords(kernel_rows, kws) for kws in spec.split_groups]`.
- `has_active_match = bool(active_rows)`; `has_split_match = bool(split_groups) and all(split_groups)`; if neither → return None.
- `related_us = sum(row.total_us for row in merge_kernel_rows(active_rows, *split_groups))`; if `related_us <= 0` → None; if no active match and `pct(related_us, total_us) < spec.min_share` → None.
- **Confidence (`Confirmed` vs `Candidate`):**
```python
confidence=(
    "Confirmed" if has_active_match or pct(related_us, total_us) >= spec.likely_share
    else "Candidate"
)
```
  I.e. Confirmed when a fused kernel name is directly matched (`active_keywords` hit) **or** related GPU-time share ≥ `likely_share`; otherwise Candidate (matched only via split-group share above `min_share`).
- `status = pattern_status(spec, has_active_match)`: mainline direct/split, upstream direct/split, or "pending direct/split" — i.e. whether a matching path already exists in-tree vs. it's the fused-name vs. split kernels.
- `FusionOpportunity` (L333) carries `pattern, status, confidence, related_us, evidence, current_locations, candidate_path, rationale, covered_row_keys, pattern_span, has_active_match, priority, subsumes`.

---

## Key correction to the task's assumptions
The thresholds were described as if in `build_priority_and_recommendation`, but they are distributed across three functions:
- `share_pct < 1.0 → ("P5","skip")` — **in** `build_priority_and_recommendation` (confirmed, L1628-1629).
- `total_us >= 5.0` + `exclusive_ratio >= 0.45` — **in** `top_overlap_opportunities` filter, not the priority builder.
- `hidden_ratio >= 0.65` — **in** `top_hidden_low_roi` filter.
- Inside `build_priority_and_recommendation` the actual thresholds are `exclusive_ratio >= 0.85` and `hidden_ratio >= 0.7`.
- `tight_gap_threshold = max(2.0, min(20.0, current.dur * 0.15))` — confirmed in `classify_dependency_signal` (L1567).

Also, the bottleneck code uses the **coarse** 5-bucket categories (`compute/communication/memory/elementwise/other`) from `triage_overlap_helpers.py:classify_kernel`, not the 12-way `CATEGORY_PATTERNS` from `triage_kernel_helpers.py`.