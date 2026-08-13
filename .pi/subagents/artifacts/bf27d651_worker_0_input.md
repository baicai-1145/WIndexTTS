# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
## Task: Implement beam search for GPT-AR decode (quality fix)

WIndexTTS GPT-AR generation produces low-quality audio because it lacks beam search. Official IndexTTS uses HF `generate(num_beams=3, repetition_penalty=10.0, do_sample=True, top_p=0.8, top_k=30, temperature=0.8, length_penalty=0.0)`. We need a pure-torch reimplementation.

## Context
- File: `/root/WIndexTTS/windextts/models/gpt.py`
- Method: `UnifiedVoice.generate()` (line ~675) currently does single-sequence greedy/sampling decode loop (lines ~760-800).
- Helper: `_sample()` (static, line ~640) already supports `repetition_penalty` via `generated_ids` param.
- Prefill already implemented (lines ~748-757): returns `kvs` (per-layer KV cache list), `logits` [B,S+1,V].
- Decode step (lines ~778-795): embed next token with mel_pos_embedding at position `step+2`, call `self.gpt(inputs_embeds=emb_dec, past_key_values=kvs, attention_mask=...)` to get new logits + updated kvs.
- Model: GPT-2 architecture, 24 layers, 20 heads, dim=1280, head_dim=64, vocab=8194. stop_mel_token=8193.

## Requirements
1. Add `num_beams: int = 1` param to `generate()` (default 1 = current behavior).
2. When `num_beams > 1`: implement **beam-search multinomial sampling** matching HF semantics:
   - Maintain K beams (K=num_beams). Start: prefill once for batch=1, then replicate to K beams.
   - Each step: for each active beam, get logits [V], apply repetition_penalty (using beam's generated ids), then if do_sample: apply temperature/top_k/top_p warpers and sample 1 token per beam (multinomial). Compute log-prob of sampled token = log_softmax(logits)[token].
   - Candidate score = beam_score + token_logprob.
   - Expand: K beams × 1 sample = K candidates. Keep top-K by candidate_score. Handle EOS (stop_token=8193): when a beam emits stop_token, move it to finished list (it's a complete hypothesis), continue with remaining beams.
   - length_penalty=0.0 (HF): final score = beam_score (no length normalization).
   - Stop when all beams finished OR max_new_tokens reached.
   - Return the highest-scoring finished hypothesis's codes [1, T_gen] (strip stop_token). If none finished, return best active beam.
3. KV cache: each beam needs its own KV cache. After prefill, replicate kvs to [K, ...]. On beam reordering, gather KV caches by selected beam indices.
4. Keep `num_beams=1` path EXACTLY as current (greedy/sample loop) — no regression.
5. CUDA Graph path (`_generate_cuda_graph`) stays greedy-only (beam search has dynamic branching, incompatible with graph). When num_beams>1, force use_cuda_graph=False.

## Constraints (AGENTS.md)
- Pure torch only: no cpp_extension/triton/flash_attn.
- No transformers/HF dependency.
- Keep existing greedy path unchanged.

## Verification
1. `num_beams=1`: must produce IDENTICAL output to current greedy/sample (run before/after, compare codes).
2. `num_beams=3, do_sample=False`: codes should be DETERMINISTIC (beam search without sampling = argmax at each beam expansion, but you still do top-K candidate selection by score).
3. Test end-to-end: `tts.infer(REF, '测试一下语音合成的质量和效果', 'ZH', do_sample=True, repetition_penalty=10.0, num_beams=3)` — codes length should be closer to official (~88 tokens, not 101).
4. Run `python -m pytest tests/align/ -q` — all 21 alignment tests must still pass.

## Return
- Modified `windextts/models/gpt.py` with beam search.
- Add `num_beams` param to `inference.py` `infer()` and `_infer_single()` (default=3 to match official).
- Report: codes length comparison (greedy vs beam3 vs official 88), any test failures.

## Out of scope
- Do NOT touch S2Mel, BigVGAN, codec, or frontend code.
- Do NOT optimize for speed yet (correctness first; beam search 3x cost is expected).

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```