# Task for researcher

Goal: Find downloadable multilingual parallel text corpora covering 99+ languages, where each language has at least 50 short sentences.\n\n**Requirement**:\n- Must cover these 99 Whisper languages: en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi, fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la, mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy, ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be, tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl, mg, as, tt, haw, ln, ha, ba, jw, su\n- Plus extensions: yue, minnan, wuyu, dialect, zh/en, en/zh\n- At least 50 sentences per language (100+ preferred)\n- Each sentence should be natural standalone (not too long, not too short)\n\n**Best candidates to investigate**:\n1. FLORES-200/ FLORES-101 dataset (842 sentences in 200 languages, on HuggingFace `facebook/flores`)\n2. OPUS-100 / OPUS datasets\n3. Tatoeba sentences\n4. UDHR (Universal Declaration of Human Rights) translations\n\n**Deliverable**: For each source found, give me:\n- Exact download URL or HuggingFace dataset name\n- How to load it (python code snippet: `datasets.load_dataset(...)` or `urllib` URL)\n- How many of the 99 languages it covers\n- Format (parallel pairs? monolingual?)\n\n**Focus**: FLORES-200 is the best bet (200 languages, 842 sentences each). Find its exact HF dataset path and loading code. Also check if there's a way to get yue/minnan/wuyu from FLORES or other sources.\n\nRead-only research. Return concrete loading code and coverage info.

---
**Output:**
Write your findings to exactly this path: /root/WIndexTTS/.pi/subagents/artifacts/outputs/2d45ea4d/research.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

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