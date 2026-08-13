# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Build a professional GPU profiler analyzer module for the WIndexTTS project, per the spec in /root/WIndexTTS/docs/PROFILER_SPEC.md. Read that spec FIRST and follow it.\n\nDeliverables (3 files):\n\n## FILE 1: /root/WIndexTTS/windextts/profiler.py (the analyzer — pure Python, NO torch import)\nA reusable trace analyzer. Must implement ALL 6 sections of the spec:\n1. Trace loading (.json/.json.gz, ph=="X" dur>0, GPU vs CPU cat separation)\n2. Kernel taxonomy classify_kernel() with the EXACT sglang ordering:\n   communication-strong > memory-strong > [memory-weak unless compute-like] > CATEGORY_PATTERNS (first match wins) > communication-weak unless compute-like > other.\n   Keyword tuples (verbatim):\n   - COMMUNICATION_STRONG: nccl, allreduce, all_reduce, reduce_scatter, allgather, all_gather, alltoall, all_to_all, cross_device_reduce, deepep, mooncake\n   - COMMUNICATION_WEAK: broadcast, dispatch, combine\n   - MEMORY_STRONG: memcpy, memset, dma, prefetch\n   - MEMORY_WEAK: copy, fill\n   - COMPUTE_HINT: gemm, gemv, matmul, cublas, cutlass, wgmma, mma, bmm, nvjet, fmha, attention, flash_attn, flashattention, grouped_mm, groupgemm, moe, expert\n   - CATEGORY_PATTERNS (first match wins, ADD conv FIRST for TTS):\n     ("conv", ["convolution","convolve","cudnn_conv","_conv","conv_transpose"]),\n     ("hybrid_linear", ["gdn","gated_delta","mamba","selective_scan","ssd","causal_conv","ssm"]),\n     ("attention", ["flash_attn","flashattention","flash_attention","fmha","attention","mla","paged_attention","decode_attention"]),\n     ("moe", ["fused_moe","grouped_mm","groupgemm","group_gemm","moe","expert","groupproblemshape"]),\n     ("gemm", ["gemm","gemv","matmul","cublas","cutlass","wgmma","mma","bmm","nvjet"]),\n     ("norm", ["rmsnorm","layernorm","_norm_"," norm","normkernel"]),\n     ("rope", ["rotary","rope","mrope"]),\n     ("softmax", ["softmax"]),\n     ("activation", ["silu","gelu","relu","act_and_mul","sigmoid"]),\n     ("quantize", ["quant","fp8","mxfp","nvfp4","dequant","cvt"]),\n     ("reduce", ["topk","argmax","argtopk","reduce","sampling","multinomial"]),\n     ("elementwise", ["elementwise","vectorized","add_kernel","sub_kernel","mul_kernel","floor_kernel","log_kernel","neg_kernel","where","clamp"]).\n3. Per-stage report function analyze(label, events, wall_ms) -> dict + prints: gpu_total_ms, idle_pct+idle_ms (merged-interval: sort GPU by start; merge overlapping where start>cur_end opens new interval else end=max(end); span=last_start...last_end; busy=sum(union widths)), tiny_kernels(<10us) count+dur, category rollup >=1%, top kernels >=1% (name[:48]).\n4. Gap analysis with BOTH views:\n   (a) DISTRIBUTION histogram buckets(us) [0,1),[1,5),[5,10),[10,50),[50,100),[100,500),[500,5000),[5000+): count, total_ms, pct. THIS IS THE DIFFERENTIATOR — S2Mel has 75% of idle as 10-50us micro-gaps that vllm-omni --min-gap-ms 5 would miss entirely.\n   (b) Big-gap attribution for gaps>=100us (default): merged-interval gaps, print prev/next GPU event + CPU containers overlapping midpoint. interesting_cpu filter: dur>=1000us AND (cat in {python_function,user_annotation} OR name contains cudaStreamSynchronize|cudaDeviceSynchronize|cudaLaunch|cudaMemcpy). Containers sorted by dur asc, top 8.\n5. Overlap/bottleneck heuristics: for each kernel name aggregate compute total_us,count,max_us,exclusive_us (time NOT overlapped by co-active kernel on another stream — single-stream traces will show ~0 overlap, that is expected and informative),hidden_us. Priority: share_pct<1.0→P5 skip; exclusive_ratio>=0.45 + non-compute→P1 fusion/overlap; low-roi-hidden {elementwise,memory}+hidden_ratio>=0.65→P4 skip.\n6. (Stretch, only if time) kernel→source via python_frames when present.\nThe analyzer must be importable as a module: `from windextts.profiler import load_trace, analyze, classify_kernel`. Add a `if __name__=="__main__"` CLI: `python -m windextts.profiler <trace.json.gz> [--min-gap-us 100] [--topn 15]` that runs the full report.\n\n## FILE 2: /root/WIndexTTS/scripts/profile_windex_stages.py (REWRITE the existing one)\nKeep the working capture logic (warmup, e2e + per-stage monkey-patch boundary capture). ADD: after capturing each stage, call windextts.profiler.analyze() and ALSO run profiler-free ≥3 wall-clock repeats for the authoritative latency (label clearly: "profiler wall_ms is distorted; the repeats below are authoritative"). Write traces to /root/windextts_dumps/profiles/. Capture with record_shapes=True AND with_stack=True.\n\n## FILE 3: /root/WIndexTTS/scripts/profile_ab.py (NEW)\nProfiler-free A/B harness. Two modes callable via argparse: --a and --b select config flags (e.g. --bf16, --graph, --steps N). Each config: warmup, then ≥5 profiler-free repeats (torch.cuda.synchronize around each infer). Report mean/median/min/max for both + the delta. Pair with optional profiler capture (--trace) for diff explanation. This is the FINAL word on whether an optimization helps.\n\n## VERIFICATION (do this, do not skip)\nAfter implementing, validate the analyzer against the EXISTING trace at /root/windextts_dumps/profiles/stage_s2mel.json.gz. It MUST reproduce approximately: idle ~48%, gemm ~57%, ~11000 launches. Run: `/root/index-tts/.venv/bin/python -m windextts.profiler /root/windextts_dumps/profiles/stage_s2mel.json.gz`. Show me the output. If numbers diverge from the existing scripts/profile_windex_stages.py output (stages_out3.txt), debug.\nDo NOT modify any model code (windextts/models/*, windextts/inference.py) — only profiler.py + the 2 scripts + docs/PROFILER_SPEC.md. Use the /root/index-tts/.venv/bin/python interpreter for all runs. Keep the analyzer torch-free so it is testable without the model.

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