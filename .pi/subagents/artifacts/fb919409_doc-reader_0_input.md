# Task for doc-reader

Read the FULL source code of /root/vllm-omni/.claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py (144 lines) and report its COMPLETE implementation in enough detail that I can re-implement it.

This is a Chrome-trace (.json/.json.gz) analyzer for GPU profiling. I need the EXACT algorithm, not a summary.

Report concretely:
1. The full event filtering logic — which 'cat' values are GPU vs CPU, how it decides what's a real device event vs host. Quote the GPU_CATS / CPU_CATS sets verbatim.
2. How it computes the merged-interval span/busy/idle (the idle_pct calc) — quote the interval-merge algorithm. What data structure does it use to sort/merge?
3. The GAP block analysis — this is the most important part. For each gap >= min_gap_us, what exactly does it print (prev GPU event, next GPU event, the 'in' CPU containers)? How does it find the CPU containers overlapping the gap midpoint? Quote the filter that picks 'interesting CPU' rows (the >=1000us threshold, the name substring filters like cudaStreamSynchronize/cudaDeviceSynchronize/cudaLaunch/cudaMemcpy).
4. The top-events ranking — sort key, topn default, how it dedups by name.
5. Argument parsing (argparse) — all flags and defaults (--min-gap-ms, --topn, etc).
6. Any edge cases / early returns / error handling.

Be exhaustive and quote actual code lines with line numbers where the logic is non-obvious. Mode: read-only, do not edit.

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