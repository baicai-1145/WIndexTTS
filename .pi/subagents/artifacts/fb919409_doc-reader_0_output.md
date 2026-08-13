I've read the complete 144-line source. Here's the exhaustive report.

---

# `trace_analyzer.py` — Complete Implementation Report

**Path:** `/root/vllm-omni/.claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py`
**Length:** 144 lines. Docstring: *"Summarize torch profiler Chrome traces for vLLM Omni diffusion optimization."*

## 1. Event filtering logic — GPU vs CPU `cat` values

The category sets are module-level constants (lines 14–15), quoted verbatim:

```python
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"python_function", "user_annotation", "cpu_op", "cuda_runtime", "cuda_driver"}
```

**Event-level filtering** (lines 41–43, inside the `for event in events` loop). A row is kept for **any** bucket only if ALL of:
- `dur` is not `None`
- `ts` is not `None`
- `dur > 0` (strictly greater than 0; `dur <= 0` is skipped)

```python
        dur = event.get("dur")
        ts = event.get("ts")
        if dur is None or ts is None or dur <= 0:
            continue
```

It then computes `cat = str(event.get("cat", ""))` and `name = event_name(event)` (where `event_name` returns `str(event.get("name", ""))`, so missing names become `""`).

**Bucketing** — every surviving event updates `by_name` first (regardless of category), then:
- if `cat in GPU_CATS` → appended to `gpu` list, and `by_gpu_name` updated
- elif `cat in CPU_CATS` → appended to `cpu` list only
- Additionally (independent of the above), if `"nccl" in name.lower()` → `nccl` dict updated with key `(cat, name)`. This check runs for every event regardless of GPU/CPU category.

A "real device event" is thus defined strictly by membership in `GPU_CATS`. Note: a single `row` tuple shape is used for both GPU and CPU lists:
`(float(ts), float(ts+dur), float(dur), name, cat, event.get("pid"), event.get("tid"))` — 7 elements: `start, end, dur, name, cat, pid, tid` (lines 48–49).

The three stats dicts (`by_name`, `by_gpu_name`, `nccl`) are all `collections.defaultdict(lambda: [0, 0.0, 0.0])` — i.e. `[count, total_dur, max_dur]`. Updated per line 50–53: `count += 1`, `total += dur`, `max = max(max, dur)`.

## 2. Merged-interval span / busy / idle computation

Sorting: GPU rows are sorted by the 7-tuple's natural order (`gpu.sort()`, line 63). Since the first two tuple elements are `(start, end)`, the sort is primarily by `start` time, with `end` as a tiebreaker.

**Interval merge** (lines 64–70). Uses a plain Python list `merged` holding `[start, end, events_list]` entries:

```python
    gpu.sort()
    merged: list[list[Any]] = []
    for start, end, dur, name, cat, pid, tid in gpu:
        if not merged or start > merged[-1][1]:
            merged.append([start, end, [(start, end, dur, name, cat, pid, tid)]])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append((start, end, dur, name, cat, pid, tid))
```

Algorithm: iterate the sorted events; if the list is empty **or** the event's `start` is strictly greater than the current interval's `end` (`start > merged[-1][1]`, so touching boundaries where `start == end` are merged), open a new interval. Otherwise extend the current interval's `end` to `max(cur_end, end)` and accumulate the event tuple into that interval's `[2]` list. This is classic sweep-line union of intervals; the "events list" per merged interval is retained for later gap analysis.

**Span/busy/idle** (lines 72–75):

```python
    span = max(end for _, end, *_ in gpu) - min(start for start, *_ in gpu)
    busy = sum(end - start for start, end, _ in merged)
    idle = span - busy
```

- `span` = (latest GPU event end) − (earliest GPU event start).
- `busy` = sum of the *union* interval widths (overlap-free due to merging).
- `idle` = `span − busy`.

Printed (lines 76–80):
```
gpu_span_s={span/1e6:.3f} busy_union_s={busy/1e6:.3f} idle_union_s={idle/1e6:.3f} idle_pct={idle/span*100:.2f}
```
(`idle_pct` is `idle / span * 100`; uses the literal 100, not `100.0`.)

## 3. GAP block analysis (the important part)

### 3a. `interesting_cpu` filter (lines 82–91)

This selects from the `cpu` list rows to be considered as "CPU containers" that may explain a GPU gap. A row qualifies if it satisfies BOTH:

**Duration threshold** — `row[2] >= 1000` (duration in microseconds ≥ 1000 µs = 1 ms).

**Name/category match** (OR of the following), where `row[4]` is `cat` and `row[3]` is `name`:
- `row[4] in {"python_function", "user_annotation"}` — i.e. cat is python_function or user_annotation (any duration ≥ 1ms)
- OR substring `"cudaStreamSynchronize"` in `row[3]`
- OR substring `"cudaDeviceSynchronize"` in `row[3]`
- OR substring `"cudaLaunch"` in `row[3]`
- OR substring `"cudaMemcpy"` in `row[3]`

Quoted verbatim:

```python
    interesting_cpu = [
        row
        for row in cpu
        if row[2] >= 1000
        and (
            row[4] in {"python_function", "user_annotation"}
            or "cudaStreamSynchronize" in row[3]
            or "cudaDeviceSynchronize" in row[3]
            or "cudaLaunch" in row[3]
            or "cudaMemcpy" in row[3]
        )
    ]
```

Note: the substring checks are NOT case-normalized — they match literal substrings against the raw name.

### 3b. Gap detection & CPU-container lookup (lines 93–104)

Gaps are computed **between consecutive merged intervals** (not raw events):

```python
    gaps = []
    for idx in range(1, len(merged)):
        gap_start = merged[idx - 1][1]
        gap_end = merged[idx][0]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap_us:
            prev_event = max(merged[idx - 1][2], key=lambda x: x[1])
            next_event = min(merged[idx][2], key=lambda x: x[0])
            mid = (gap_start + gap_end) / 2
            containers = [row for row in interesting_cpu if row[0] <= mid <= row[1]]
            containers = sorted(containers, key=lambda x: x[2])[:8]
            gaps.append((gap_dur, gap_start, gap_end, prev_event, next_event, containers))
```

Details:
- `gap_start` = end of previous merged interval; `gap_end` = start of next merged interval.
- **Gap threshold:** `gap_dur >= min_gap_us` (the `--min-gap-ms` flag converted to µs).
- **`prev_event`** = the event inside the previous merged interval with the **maximum `end`** (`x[1]`).
- **`next_event`** = the event inside the next merged interval with the **minimum `start`** (`x[0]`).
- **`mid`** = `(gap_start + gap_end) / 2` (the arithmetic midpoint of the gap).
- **`containers`** = all `interesting_cpu` rows whose `[start, end]` interval contains the midpoint: `row[0] <= mid <= row[1]`. That's the "CPU containers overlapping the gap midpoint" lookup.
- **Containers are sorted by duration ascending** (`key=lambda x: x[2]`) and truncated to the **first 8** (the *shortest* 8). Note: this means it keeps the 8 shortest-duration overlapping CPU events, not the longest.

### 3c. Gap output (lines 106–115)

A summary line prints count and total gap sum:
```python
    print(f"gaps_ge_{min_gap_us / 1000:.3f}ms count={len(gaps)} sum_s={sum(g[0] for g in gaps) / 1e6:.3f}")
```
(Note: label uses `min_gap_us/1000` → `ms`.)

Then gaps are sorted by `gap_dur` **descending** (`sorted(gaps, reverse=True)`) and the top `topn` printed:

```python
    for gap_dur, gap_start, gap_end, prev_event, next_event, containers in sorted(gaps, reverse=True)[:topn]:
        print(f"\nGAP {gap_dur / 1000:.3f} ms ts={gap_start:.0f}->{gap_end:.0f}")
        print(f"  prev {prev_event[4]} {prev_event[2] / 1000:.3f} ms {prev_event[3][:160]}")
        print(f"  next {next_event[4]} {next_event[2] / 1000:.3f} ms {next_event[3][:160]}")
        for row in containers:
            print(f"  in   {row[4]} {row[2] / 1000:.3f} ms {row[3][:180]}")
```

For each gap, output format:
- Header: `GAP {dur_ms:.3f} ms ts={gap_start:.0f}->{gap_end:.0f}` (durations in ms, ts in µs, no decimals).
- `prev` line: `cat`, `dur_ms`, and `name[:160]` (truncated to 160 chars).
- `next` line: `cat`, `dur_ms`, and `name[:160]`.
- `in` line per container: `cat`, `dur_ms`, and `name[:180]` (truncated to 180 chars).

`prev_event[4]` = cat, `prev_event[2]` = dur (µs), `prev_event[3]` = name. Note the names are formatted as `{prev_event[3][:160]}` — so 160-char truncation of the raw name, with the units labeled `ms` (dur is in µs, divided by 1000).

## 4. Top-events ranking

Two rankings, both sorted by **total duration descending** and truncated to `topn`:

**Top GPU/operator events** (lines 117–119):
```python
    for name, (count, total, max_dur) in sorted(by_gpu_name.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
        print(f"  {int(count):8d} total={total / 1e6:9.3f}s max={max_dur / 1000:9.3f}ms {name[:180]}")
```
- Sort key: `kv[1][1]` = `total` (sum of durations of that GPU event name).
- **Dedup by name**: keys are the event `name` strings — all events sharing the same name aggregate into one row. Stats shown: `count`, `total` (seconds, `total/1e6`), `max` (ms, `max_dur/1000`), `name[:180]`.

**Top NCCL-like events** (lines 121–123):
```python
    for (cat, name), (count, total, max_dur) in sorted(nccl.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
        print(f"  {int(count):8d} total={total / 1e6:9.3f}s max={max_dur / 1000:9.3f}ms cat={cat} {name[:160]}")
```
- Keys are `(cat, name)` tuples — dedup by (category, name) pair. Same sort-by-total logic. Prints `cat=` then `name[:160]`.

**`topn` default = 20** (both rankings and gap output). Note: unlike `by_gpu_name`, `by_name` (the all-events-by-name dict) is computed but **never printed** — it's a dead/unused aggregate.

## 5. Argument parsing (lines 126–135)

```python
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="trace.json or trace.json.gz files")
    parser.add_argument("--min-gap-ms", type=float, default=5.0, help="minimum GPU idle gap to print")
    parser.add_argument("--topn", type=int, default=20, help="number of gaps/hotspots to print")
    args = parser.parse_args()

    for trace in args.traces:
        summarize_trace(trace, args.min_gap_ms * 1000.0, args.topn)
```

| Flag | Type | Default | Conversion |
|------|------|---------|-----------|
| positional `traces` | `Path`, `nargs="+"` (1+ required) | — | — |
| `--min-gap-ms` | `float` | `5.0` | multiplied by `1000.0` → µs before calling `summarize_trace` |
| `--topn` | `int` | `20` | passed as-is |

`description=__doc__` uses the module docstring as the help description. Multiple traces are processed sequentially in order given.

## 6. Edge cases / early returns / error handling

- **File opening** (lines 17–20): `open_trace` checks `path.suffix == ".gz"`; gz files opened with `gzip.open(path, "rt")` (text mode), others with `path.open("rt")`. No other compression handled.
- **Trace JSON shape** (line 31): `data` is used directly as the list if the root is a list, otherwise `data.get("traceEvents", [])` (so both bare-event-array traces and `{"traceEvents": [...]}` Chrome traces work). No handling for `"traceEvents"` under other nested keys.
- **Skipped events** (lines 41–44): any event missing `dur`, missing `ts`, or with `dur <= 0` is silently dropped from all buckets (still counted in `len(events)` output, which uses the pre-filtered `events` list).
- **No GPU events** (lines 58–59): if `gpu` is empty after filtering, prints `"No GPU events found."` and **returns early** — the span/busy/gap/top-rank sections are all skipped. CPU-only analysis is not performed in that case.
- **No explicit try/except** anywhere — no custom error handling around `json.load`, file I/O, or gzip errors; failures propagate as exceptions. `json.load` on a non-JSON file will raise `json.JSONDecodeError`.
- **Division by zero risk**: if `gpu` is non-empty, `span > 0` guarantees `idle/span` is safe. `min_gap_ms` of 0 is allowed (would report all gaps).
- **Empty trace**: if `data` yields an empty `events` list, `gpu` is empty → early return after printing `events=0 gpu_events=0 cpu_events=0`.
- **`by_name`** is populated but never consumed for output (dead aggregate).
- **Unicode/type notes**: `event_name` coerces name to `str` via `str(event.get("name", ""))`. `ts`/`dur` are coerced to `float` when building the row. Name substring searches for NCCL use `.lower()` but the `interesting_cpu` filters are case-sensitive.

---

## Acceptance Report