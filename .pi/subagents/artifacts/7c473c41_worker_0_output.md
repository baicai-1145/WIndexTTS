The GPT analysis reproduces the expected signature: GPU-work=256ms, idle 20.2%, gemm 58.4%, attention 21.5% — matching the earlier analysis. The `reduce` category now correctly catches `gatherTopK` (the GPT sampling) — an improvement over the old classifier.

The complete profiler system is implemented and verified. Let me finalize the acceptance report.

## Summary

**Implemented the professional profiler system** per `/root/WIndexTTS/docs/PROFILER_SPEC.md`:

### Deliverables
1. **`windextts/profiler.py`** (new, ~700 lines) — torch-free trace analyzer:
   - Trace loading (.json/.json.gz, ph=="X", GPU/CPU cat separation)
   - sglang-faithful kernel taxonomy (comm-strong > mem-strong > mem-weak-unless-compute > patterns > comm-weak > other), with `conv` first for TTS
   - Per-stage report: GPU-work, idle% (merged-interval), tiny-kernels, category rollup, top kernels
   - **Gap analysis**: (a) micro-gap distribution histogram (the differentiator — catches the 10-50us dispatch ocean that vllm-omni's --min-gap-ms 5 misses), (b) big-gap attribution with prev/next GPU + CPU containers + enhanced nearest-event
   - Overlap/bottleneck heuristics (exclusive_ratio, hidden_ratio, P1/P4 priority)
   - Source attribution (gap-container path works; python_frames path documented as torch-version-limited)
   - CLI: `python -m windextts.profiler <trace>`

2. **`scripts/profile_windex_stages.py`** (rewritten) — capture harness: warmup → profiled traces (record_shapes + with_stack) + **profiler-free ≥3 repeats** for authoritative latency

3. **`scripts/profile_ab.py`** (new) — profiler-free A/B harness with config flags, ≥5 repeats, delta verdict, optional paired trace

### Verification
- Analyzer reproduces the old script's numbers **exactly** on the same trace (idle 72.7%, GPU 103ms, launches 11151, gemm 37.9ms/36.8%/1136)
- New harness runs end-to-end producing authoritative table (E2E 594ms mean, S2Mel 261ms wall / 103ms GPU = 158ms host bubble)
- Gap attribution shows **source-level cause**: `bigvgan.py(265): forward` → Conv1d → 3.3ms GPU idle bubble
- Classification unit tests pass; analyzer is torch-free (runs on miniconda python)