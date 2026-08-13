# Task for doc-reader

Read and summarize the vLLM-Omni diffusion performance optimization skill at /root/vllm-omni/.claude/skills/diffusion-perf-opt/. Read SKILL.md fully, then skim scripts/trace_analyzer.py and any files in references/ and agents/.

Goal: Extract methodology specifically for profiling DIFFUSION model inference (this is directly relevant — my S2Mel-CFM stage is a 12-25 step DiT flow-matching solver, similar workload to image diffusion).

Return specifically:
1. How they profile a diffusion/flow-matching solver (per-step profiling? whole-loop? CUDA graph considerations?)
2. What they identify as typical diffusion bottlenecks (the DiT forward, the Euler loop overhead, the CFG double-forward, etc.)
3. Any metrics or thresholds they use to decide an optimization helps vs hurts
4. How trace_analyzer.py parses the trace — what it looks for, what report it generates
5. Any "pitfalls" or "why an optimization that looks fast is actually slow" guidance (this is important to me — I've seen CUDA Graph make my vocoder SLOWER and need to understand why)

Quote concrete details (function names, metric keys, threshold values). Mode: read-only, do not edit.

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