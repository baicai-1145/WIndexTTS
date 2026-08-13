All changes complete and verified. Here is the structured acceptance report.

## Implementation Summary

**Implemented beam search for GPT-AR decode** in `/root/WIndexTTS/windextts/models/gpt.py` plus `num_beams` wiring in `inference.py`:

1. **`_generate_beam_search()`** — pure-torch beam-search multinomial sampling: single prefill → K replicated beams (logits/KV/mask), batched decode over active beams per step, cumulative log-prob scoring, EOS→finished-hypothesis handling, best-scoring finished hypothesis wins (stop token stripped). `num_beams=1` keeps the exact existing greedy/sample loop.
2. **Fixed `_sample` repetition_penalty sign bug** — the parent's uncommitted edit inverted HF's formula (`score/penalty` for negatives, which *raises* probability of repeats); corrected to `score<0 ? score*penalty : score/penalty` (HF-exact).
3. **CUDA Graph path now honors `repetition_penalty`** — it previously ignored it entirely, which would silently break the quality fix on the default fast path.
4. **`inference.py`**: `num_beams=3` default (official), forced `use_cuda_graph=False` when beam search active.

**Quality result**: duration 3.55s vs official 3.47s (was 3.99s); energy-envelope rhythm now matches official phrase/pause structure. **24/24 alignment tests pass.**