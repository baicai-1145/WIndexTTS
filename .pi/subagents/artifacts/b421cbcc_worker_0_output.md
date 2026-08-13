All work complete. Final summary:

## Implemented: CUDA-Graph beam search (R14)

**Changed files:**
- `windextts/models/gpt.py` — new `_generate_beam_graph()`: beam search over a captured CUDA Graph with **static batch K=num_beams**, fixed KV buffers, no beam removal/reordering. EOS beams freeze their score+codes and keep feeding stop_token so graph shape never changes; logits ignored after. New `_beam_graph_cache` keyed `(num_beams, max_seq, dtype)`. `generate()` routes `use_cuda_graph + num_beams>1` to it.
- `windextts/inference.py` — `use_cg` no longer disabled for `num_beams>1`; warmup() now pre-captures the beam3 graph; docstring updated.
- `docs/PERFORMANCE.md` — R14 row, stage breakdown, rejected-approach table updated.

**Validation:**
- **GPT stage: eager beam3 1464ms → graph beam3 ~490ms steady (3.0x)** — the acceptance target (~500ms) met.
- **E2E (beam3, 4 texts): 1.4s eager → 0.65s min / 0.74s mean** vs official fp32 1.60s min (RTF 5.66x vs 2.09x).
- **9/10 seeds bit-identical** to eager beam3 (same codes, lengths 73–83); seed 4 diverges only at final token — mid-loop EOS shrinks the eager batch (different multinomial RNG draw count). Statistically equivalent, documented.
- **12/12 quality cases** (3 texts × 4 seeds): eager/graph durations identical, no brick/NaN/silence.
- **15/15 alignment tests pass** (`gpt_beam_align`, `gpt_ar_graph_align`, etc.).
- First-request beam capture moved into warmup (verified cache reuse, no recapture).

**Open risks/questions:** None blocking. The seed-4 divergence is inherent to static-K graphs (a true KV-gather beam graph would restore bit-exactness at ~2x complexity — documented in PERFORMANCE.md as rejected).

**Recommended next step:** Re-run the profiler-driven optimization loop on the new beam3 path if further latency reduction is desired (beam3 KV buffers are 3× greedy; bucket sizing for beam is the same 64-step as greedy).