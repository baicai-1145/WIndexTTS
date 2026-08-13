# Task for doc-reader

Read /root/sglang/.claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (2840 lines). I need the kernel CLASSIFICATION and BOTTLENECK-DETECTION logic in implementable detail — not a high-level summary.

Focus on these specific functions (find them, read them, report their real logic):
1. classify_kernel() and the CATEGORY_PATTERNS list — quote the FULL ordered list of (category -> regex patterns). First-match-wins ordering matters, so preserve order exactly. Include both 'strong' and 'weak' keyword logic if present.
2. The bottleneck heuristics in build_priority_and_recommendation() (or similarly named) — quote the actual thresholds: exclusive_ratio >= 0.45, hidden_ratio >= 0.65, share_pct < 1.0 -> P5, the dependency signal (tight_gap_threshold = max(2.0, min(20.0, current.dur * 0.15))), and what priority/recommendation string each branch produces.
3. How it computes 'exclusive_us', 'hidden_us', 'overlap_with' for a kernel aggregate — the per-kernel overlap accounting. Quote the core loop.
4. top_overlap_opportunities() and top_hidden_low_roi() — the exact filter conditions (total_us >= 5.0, exclusive_ratio >= 0.45, category in {elementwise,memory}, hidden_ratio >= 0.65).
5. The fusion detection (FusionPatternSpec min_share/likely_share, Confirmed vs Candidate) — just the structure and a couple example patterns.

This is for re-implementing the analyzer myself, so precision matters more than breadth. Quote real code. Mode: read-only, do not edit.

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