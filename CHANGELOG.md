# Changelog

## 0.3.0 (2026-08-16)

Density release: core inference package cut from 5702 → **5201 lines**
(-9%) via dead-code removal and shared-path consolidation. Pure refactor —
**zero behavior change**; every lane re-verified against the same align
suite and reference-audio fingerprints.

### Changed
- **config.py 205→72**: deleted 8 dataclasses mirroring config.yaml — only
  the 6 runtime-consumed fields remain (defaults == official 2.5).
- **gpt.py 637→582**: `_prefill`/`_eager_step`/`_get_graph` shared by the 4
  decode paths (eager / beam / beam-graph / cuda-graph); dispatch collapsed.
- **inference.py 555→507**: `_MEL_RATIO` 101-language table packed to one
  string; `_lazy` property factory; `_mat_lookup` dedup.
- **models sweep**: codec/campplus/length_regulator/s2mel_dit training-dead
  paths removed (params kept — strict ckpt load); `__main__` smoke blocks
  dropped from 11 files; `__all__` re-exports removed from 8 files.
- **scripts -500 lines**: dropped bench_harness / profile_windex (superseded
  by profile_windex_stages), measure_lang_ratios (output baked into source).
- README: one-line vs-official claim (16k fewer core code lines, 4.6x
  fastest lane, CUDA-Graph/W4A16/low-vram features official lacks).

### Verified
- 21/24 align tests green (same 3 pre-existing env failures as 0.2.0:
  campplus CUDA tolerance, tokenizer/w2v2 official-package compare).
- GPT: prefill max_diff 1.96e-05; greedy codes exact; graph 2.10x;
  beam3 eager-vs-graph bit-identical.
- E2E reference audio: duration bit-identical, RMS within run-to-run noise.

## 0.2.0 (2026-08-16)

Quality-focused release: fixes the "harsh/brittle audio" report, tightens
alignment with official IndexTTS-2.5, and speeds up every lane via a
CUDA-Graph KV-pool sizing fix.

### Fixed
- **TeaCache HF-energy artifacts (harsh audio)** — root cause: the monitor
  signal excluded the timestep and no model-specific rescaling coefficients
  were used, skipping up to 80% of DiT steps. Reimplemented with vLLM-Omni
  semantics (adaLN-modulated monitor + calibrated polyfit coefficients) but
  listening verification showed any effective skip rate is audible for this
  audio DiT → **disabled by default** (`teacache_thresh=0.0`); the enable
  switch is now explicit per call (was sticky).
- **Four alignment bugs vs official** (spk_cond diff 11.1 → 0.002):
  w2v-bert stats used variance as std; featurizer dropped odd tail frames
  (official pads with 1.0); w2v-bert encoder missed HF masked-position
  zeroing; emotion default path now matches official conformer semantics.
- **CUDA-Graph KV pool oversizing** — `max_mel_tokens` was derived from the
  segment token *limit* (120 × 13 = 1560 for ZH), so per-step attention/mask
  kernels scanned a ~1664-position pool while typical requests need ~130.
  Now derived per segment from actual token count (+2x headroom).

### Changed
- `cfm_steps` default 12 → **15** (listening-verified lossless floor; 12
  degrades syllable-final breaths).
- fp32 lane also uses the S2Mel CUDA Graph path (bit-identical to eager
  full steps, verified same-seed to 6 decimals).
- README perf table now leads with **RTF** + per-lane GPU memory.

### Performance (A10G, unified protocol, RTF = synth/audio)
- default lane (fp16 beam3 15 steps): 0.28 → **0.24** (2.6x official fp32)
- W4A16 greedy: 0.18 → **0.14**; W4A16 low-vram greedy: **0.12** @ 2.9G
- all lanes now beat vLLM-Omni's default deploy config (0.20 @ ~18G)

### Added
- `WIndexTTS.infer_from_codes()` — diagnostic seam (external GPT codes →
  win downstream) used for the A/B isolation that pinned the audio issues.

## 0.1.1
- deps fixes: correct text-normalizer package, torchcodec audio backend.

## 0.1.0
- initial release: pure-torch IndexTTS-2.5, CUDA-Graph GPT/S2Mel, W4A16,
  webui/server/CLI entry points, Windows portable package.
