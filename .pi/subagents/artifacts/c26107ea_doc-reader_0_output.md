I've read all four files fully. Here's the extraction tailored to your S2Mel-CFM DiT flow-matching solver.

## 1. How they profile a diffusion/flow-matching solver

The skill treats diffusion generation as a three-stage pipeline and profiles it **both whole-loop (low-overhead stage timing) and per-operator (diagnostic trace)**, keeping the two strictly separate:

- **Whole-loop / stage-level (authoritative latency):** Use server `--log-stats` and `--enable-diffusion-pipeline-profiler` (low overhead). The named stages are `vae.encode`, `diffuse`, `vae.decode`. The `diffuse` stage is the DiT + CFG solver loop. Baseline collection must be profiler-free: disable torch profiler and stack collection, fix `torch.compile` state for fair A/B, avoid `--enforce-eager` unless eager is the target.
- **Per-operator (diagnostic only):** After narrowing hypotheses, run torch profiler on 1–3 requests, excluding warmup via `/start_profile`…`/stop_profile` endpoints. Two separate trace modes:
  - **Operator/shape trace**: `torch_profiler_record_shapes=True`, stack off — ranks CUDA kernels, NCCL collectives, attention/MLP/norm/RoPE, shape-specific hot operators.
  - **Host-stack trace**: `torch_profiler_with_stack=True`, shapes normally off — maps CPU/Python host gaps, sync points, scheduler paths.

  Key discipline: **torch profiler latency is diagnostic only and must never be the final latency claim** ("traces are diagnostic artifacts and may distort latency"). CUPTI `Command Buffer Full` is profiler overhead, not a model bottleneck.

For your 12–25 step Euler loop specifically: the skill does not profile per-step; it profiles the whole `diffuse` stage and then ranks **per-operator kernel totals**. But the scheduler section is directly relevant — it calls out "Scheduler work can create small host/device gaps; cache tiny solve coefficients when timesteps/order are known" and lists scheduler caching as P0. The Euler-loop overhead in your CFM would map to this: repeated tiny per-step coefficient computation (e.g., `torch.linalg.solve`, timestep math) creating host bubbles between GPU kernels. The playbook lists "Scheduler fusion/cache: cache coefficient solves and fuse CFG combine with scheduler arithmetic."

## 2. Typical diffusion bottlenecks identified

- **The DiT transformer forward** — self-attention/FFN dominate when token count is high; FA/SDPA kernel dominance. For Wan2.2 I2V, self-attention dominates cross-attention.
- **CFG double-forward** — positive + negative prompts double transformer forwards. Bottleneck framing: `CFG=2 x USP=world/2` vs `CFG=1 x USP=world`. For your 12-25-step solver with a `cfg_rate=0.7` (from your index-tts config), CFG guidance is directly relevant.
- **Euler/solver loop overhead** — host/device gaps from small scheduler linear algebra; cache coefficient solves and fuse CFG combine with scheduler arithmetic.
- **VAE encode/decode** — relevant to your S2Mel only insofar as the mel-spec feature path (Kaldi fbank) is the analog of `vae.encode`; the skill treats this as a separate stage.
- **Host/runtime stalls** — `torch.cuda.empty_cache()`, `cudaStreamSynchronize`/`cudaDeviceSynchronize`, Python preprocessing, small allocation paths leaving GPU lanes idle.
- **NCCL/parallel comm** — only if you go multi-GPU (likely N/A for a single-GPU CFM; the skill is vLLM-multi-GPU-heavy).
- **Cross-attention on sequence-parallel** — "skip SP/USP when condition tokens are much smaller than latent tokens; extra all-to-all can exceed compute saved."

## 3. Metrics and thresholds used to decide help-vs-hurt

- **Stage timings in ms**: always report `vae.encode`, `diffuse`, `vae.decode`, server `inference_time`, client end-to-end, and peak memory. "A configuration can improve `diffuse` while hurting `vae.decode`; record both effects."
- **trace_analyzer keys**: `gpu_span_s`, `busy_union_s`, `idle_union_s`, `idle_pct`, `GAP` blocks, "Top GPU/operator events by total duration", "Top NCCL-like events by category". Analyzer defaults: `--min-gap-ms 5` (host gaps: lower to `1`), `--topn 20`.
- **idle_pct** — high value signals host/runtime stalls; maps to gap analysis.
- **A/B protocol thresholds**: 1 warmup request per scenario, then **≥3 measured repeats** for shortlisted configs; exploratory one-shot pruning only when "the margin is large," explicitly labeled one-shot. One variable per test, identical model/input/seed/shape/steps.
- **Quality gate** for every optimization: exact/near-exact agreement for math-preserving changes; SSIM/PSNR/LPIPS/cosine-similarity/MAE/MSE + temporal flicker + seed stability for precision/quantization/approx. **Failed or inconclusive quality validation blocks a ready-to-merge claim.** (This maps to your project's `torch.allclose(atol=1e-4, rtol=1e-3)` numerics gate.)
- **Priority**: P0 = low-risk measurement/fixes; P1 = contained code changes with trace evidence; P2 = high-risk (custom kernels, quantization, approximate attention).

## 4. How trace_analyzer.py parses the trace

- Opens `.json` or `.json.gz`; handles both a raw event list and the `{traceEvents: [...]}` wrapper.
- Iterates events, keeping those with `dur>0`. Classifies by `cat`:
  - `GPU_CATS = {"kernel","gpu_memcpy","gpu_memset"}` — real device work.
  - `CPU_CATS = {"python_function","user_annotation","cpu_op","cuda_runtime","cuda_driver"}` — host work.
  - Events with `"nccl"` in the lowercased name are bucketed by `(cat, name)`.
- Per-name stats: `[count, total_dur, max_dur]`.
- **GPU span/busy/idle**: sorts GPU events by start, merges overlapping intervals into a union, computes `span`, `busy_union_s`, `idle_union_s`, `idle_pct`. **This is the GPU-idle analysis** — gaps are intervals between merged GPU busy regions.
- **Interesting CPU** filter: rows with `dur>=1000us` that are `python_function`/`user_annotation`, or contain `cudaStreamSynchronize`/`cudaDeviceSynchronize`/`cudaLaunch`/`cudaMemcpy` in the name — used to attribute gaps to host code.
- **GAP blocks**: for each gap ≥ `min_gap_us`, it prints `prev` and `next` GPU events (with category, duration, name truncated to 160 chars) and up to 8 `in` CPU containers overlapping the gap midpoint (the enclosing host functions).
- **Reports**: top GPU/operator events by total duration (sorted desc, `--topn`), and top NCCL-like events by category.
- **Explicit limitations** (stated in SKILL.md): the analyzer "summarizes timing only. It does not parse tensor shapes, attribute overlap to individual streams, prove quality, or provide final latency claims." Shape analysis is deferred to `ops_rankN.xlsx` / PyTorch key averages.
- **Critical interpretation note**: "Treat `cat=user_annotation` NCCL ranges as enclosing annotations; prefer `cat=kernel` or `cat=gpu_user_annotation` for real device work" — annotations can overcount nested intervals.

## 5. Pitfalls / "why an optimization that looks fast is actually slow" guidance

This is the most important part for your CUDA Graph vocoder regression:

- **Profiler distortion**: torch profiler traces inflate latency and add host overhead; `Command Buffer Full` (CUPTI) is profiler overhead. Never judge an optimization from profiler numbers — re-run the non-profiler baseline. "Torch profiler latency is diagnostic only and must not be used as the final latency claim."
- **Host-device sync traps**: `torch.cuda.empty_cache()` prevents OOM but forces synchronization → GPU idle bubbles; guard it by memory headroom or make it optional. Explicit `cudaStreamSynchronize`/`cudaDeviceSynchronize` are idle sources.
- **Looking fast on `diffuse` but slow on another stage**: evaluate by stage and memory, not end-to-end — an optimization can improve one stage and hurt another (`vae.decode` example).
- **Warmup/lazy-init**: first-request FSDP/HSDP lazy init creates idle; exclude first-request lazy init, always warmup before measuring.
- **Rank/stream caveats**: the analyzer does not attribute overlap to individual streams; expanding to more ranks is only done if rank 0 suggests imbalance — relevant to multi-GPU CUDA Graph capture considerations.
- **Ann notation overcount**: `user_annotation` NCCL ranges overcount nested intervals — use `cat=kernel` for real device work.
- **Attention backends**: verify the kernel is actually on the intended fast path before declaring a win; compare supported backends only with identical workload/quality settings; reduce attention input size only via safe model/config choices.
- **Scheduler overhead is real**: small per-step coefficient solves create tiny host/device gaps; cache coefficients when timesteps/order are fixed (your 25-step fixed Euler loop is a prime candidate).
- **Don't rewrite operators without proof**: "Do not implement high-risk operator rewrites before proving the operator is a bottleneck for the target shapes." For custom CUDA/Triton/quantization, require strict A/B + quality + seed-stable regression; a P2 change that fails the quality gate is not "ready to merge."

**Direct relevance to your vocoder CUDA Graph regression**: the skill's host/runtime + measurement-reliability layers point at the classic causes — (a) graph capture forcing re-allocations or layout conversion, (b) graph replay pinning the kernel to a suboptimal attention/backend path for your exact shapes, (c) capture-time vs replay-time host sync behavior that profiler numbers (not measured) would hide, and (d) stage-level (not E2E) regression masked by another stage. The prescribed fix is a **profiler-free A/B on the `diffuse`-analog stage (your S2Mel-CFM solver) with ≥3 repeats, identical seed/shape/steps**, plus a `by_shape`-style shape audit (`ops_rankN.xlsx` analog) to confirm the graph-captured kernel is on the intended fast path for the captured shapes.

---

Concrete sources: `SKILL.md` (stages `diffuse`/`vae.encode`/`vae.decode`; P0/P1/P2 rules; quality gate metrics; heuristics incl. `empty_cache`, `Command Buffer Full`, scheduler coefficient caching; analyzer keys), `scripts/trace_analyzer.py` (GPU_CATS/CPU_CATS sets, merged-interval span/busy/idle, `interesting_cpu` filter at `>=1000us`, GAP prev/next/in with 160-char truncation, `--min-gap-ms`/`--topn`, "No GPU events found" early-return), `references/optimization-playbook.md` (scheduler fusion/cache, FA-to-LA necessity check, cross-attn SP skip, output table format), `agents/openai.yaml` (interface metadata only, no methodology).

**Severity notes** (applying your own risk framing): measurement-protocol discipline (baseline vs diagnostic separation, warmup, ≥3 repeats) is the highest-value transferable item — the SKILL treats it as a correctness gate, not style. The per-operator/numerics-alignment (your `atol=1e-4, rtol=1e-3` rule) and per-stage evaluation are the second. Multi-GPU parallelism (USP/CFG/HSDP/VAE-PP) is largely **not applicable** to your single-GPU S2Mel-CFM; only its "CFG doubles forward work" framing and scheduler-caching guidance transfer.