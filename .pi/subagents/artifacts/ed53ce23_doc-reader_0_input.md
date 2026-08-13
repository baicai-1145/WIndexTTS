# Task for doc-reader

Read and summarize the SGLang torch-profiler analysis skill at /root/sglang/.claude/skills/llm-torch-profiler-analysis/. Read SKILL.md fully, then skim the scripts in scripts/ (analyze_llm_torch_profile.py, analyze_sglang_torch_profile.py) and any reference docs in references/.

Goal: Extract a REUSABLE methodology for profiling PyTorch inference workloads that I can apply to a pure-torch TTS pipeline (WIndexTTS).

Return specifically:
1. The profiling workflow steps (warmup -> capture -> analyze) they recommend
2. What metrics they extract from a torch.profiler trace (self_cuda_time, CPU-GPU gap, kernel categories, occupancy, etc.) — list the exact keys/tables they look at
3. How they classify/categorize kernels (e.g. GEMM vs elementwise vs memcpy vs launch-overhead) — any taxonomy or grouping logic in the scripts
4. Any heuristics for "is this kernel a bottleneck worth optimizing" (e.g. arithmetic intensity, memory-bound vs compute-bound checks, launch-count vs duration)
5. How they separate prefill vs decode or stage-vs-stage in a single trace

Be concrete: quote the actual column names, table sort keys, and any threshold numbers. This is for building my own profiler tool, not for running sglang. Mode: read-only, do not edit anything.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

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