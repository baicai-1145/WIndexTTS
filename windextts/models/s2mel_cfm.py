"""S2Mel-CFM: Conditional Flow Matching for mel generation.

Re-implements ``indextts/s2mel/modules/flow_matching.py`` (BASECFM + the Euler
solver) using our already-aligned ``DiT`` estimator and ``InterpolateRegulator``.
Zero indextts/transformers dependency.

The CFM solves an ODE from noise (t=0) to data (t=1) using a fixed-step Euler
solver with classifier-free guidance (CFG). The velocity field is the DiT
estimator. Verified contract (flow_matching.py:31-110):

  inference(mu, x_lens, prompt, style, f0, n_timesteps=25, cfg_rate=0.7):
    z = randn([B, 80, T]) * temperature         # initial noise
    t_span = linspace(0, 1, n_timesteps+1)       # [0, ..., 1]
    # prompt region: prompt_x holds ref mel in [:, :, :prompt_len], x zeros it
    for step in 1..n_timesteps:
        dt = t_span[step] - t_span[step-1]
        if cfg_rate > 0:                          # classifier-free guidance
            stack cond/uncond batches, single estimator forward, then
            dphi = (1+cfg)*dphi_cond - cfg*dphi_uncond
        x = x + dt * dphi
        x[:, :, :prompt_len] = 0                  # keep prompt region zeroed

Output: mel [B, 80, T] (the caller strips the prompt region: x[:, :, prompt_len:]).

This module is the third and final S2Mel sub-task. With length_regulator + DiT
both bit-aligned, the CFM loop is deterministic given a fixed RNG seed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

import torch
import torch.nn as nn

from windextts.models.length_regulator import InterpolateRegulator
from windextts.models.s2mel_dit import DiT

__all__ = ["S2MelCFM", "S2Mel"]


class S2MelCFM(nn.Module):
    """Conditional Flow Matching wrapper around DiT estimator.

    Holds the estimator (DiT) and implements the Euler ODE solver + CFG.
    The length_regulator is owned by the parent ``S2Mel`` (it is shared /
    called separately for prompt vs target), not by CFM.
    """

    def __init__(self, estimator: DiT, in_channels: int = 80, sigma_min: float = 1e-6):
        super().__init__()
        self.estimator = estimator
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self._graph_cache: dict = {}  # keyed by (B,C,T_bucket,prompt_len,dtype,cfg,n_steps)
        # mel-frame bucketing for the graph path: round T up to this multiple
        # so different text lengths reuse one captured graph (max a few graphs).
        self._GRAPH_BUCKET = 64

    @torch.no_grad()
    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor | None,
        n_timesteps: int = 25,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.7,
        use_graph: bool = False,
    ) -> torch.Tensor:
        """Solve the CFM ODE from noise to mel.

        Args:
            mu: condition [B, T, 512] (cat of prompt_condition + target cond).
            x_lens: [B] lengths of mu's time dim.
            prompt: reference mel [B, 80, T_prompt].
            style: [B, 192] CAMPPlus embedding.
            f0: unused (None).
            n_timesteps: Euler steps (default 25).
            temperature: noise scale.
            inference_cfg_rate: classifier-free guidance rate (default 0.7).
            use_graph: if True, use CUDA-Graph-captured Euler loop (faster,
                but incompatible with TeaCache — disables it for this call).
        Returns:
            mel [B, 80, T] including the prompt region (caller strips it).
        """
        B, T = mu.size(0), mu.size(1)
        device = mu.device
        # DiT transformer caches (RoPE freqs_cis, optional KV) must be sized to T.
        # IMPORTANT (graph mode): freqs_cis is precomputed from block_size (16384),
        # so it NEVER needs rebuilding for different T. But setup_caches rebuilds
        # it whenever max_seq_length grows. If a previously-captured graph bound
        # the old freqs_cis tensor address, rebuilding invalidates it → garbage
        # RoPE → brick audio. Fix: in graph mode, setup_caches ONCE to a large
        # fixed max_seq_length (covers all requests); subsequent calls return early.
        cache_T = T
        if use_graph:
            bucket = getattr(self, "_GRAPH_BUCKET", 64)
            cache_T = ((T + bucket - 1) // bucket) * bucket
            # Pin estimator caches to a large fixed size so setup_caches never
            # rebuilds freqs_cis after the first call (graph-safe).
            cache_T = max(cache_T, getattr(self, "_GRAPH_PIN_SEQ", 256))
        self.estimator.setup_caches(max_batch_size=2 * B, max_seq_length=cache_T)
        z = torch.randn([B, self.in_channels, T], device=device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mu.dtype)
        if use_graph:
            # graph path requires the estimator forward to be capture-safe;
            # TeaCache's data-dependent branching is incompatible, so bypass it.
            est = self.estimator
            tc_was = getattr(est, "teacache_enabled", False)
            est.teacache_enabled = False
            try:
                return self.solve_euler_graph(
                    z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate
                )
            finally:
                est.teacache_enabled = tc_was
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor | None,
        t_span: torch.Tensor,
        inference_cfg_rate: float = 0.7,
    ) -> torch.Tensor:
        """Fixed-step Euler ODE solver with classifier-free guidance.

        Replicates flow_matching.py:57-111 exactly. When the estimator has
        TeaCache enabled, redundant DiT forwards are skipped (vLLM-Omni style).
        """
        # reset TeaCache state for this solve (cond + uncond branches)
        est = self.estimator
        if getattr(est, "teacache_enabled", False):
            est._tc_state = {"cnt": 0, "prev_core": None, "prev_residual": None, "accum": 0.0}
        # Reduced-precision DiT forward strategies. Two modes:
        #   - estimator_autocast_dtype: wraps each op in torch.autocast (per-op
        #     dtype-policy dispatch). Numerically safe for fp16 (10 mantissa
        #     bits) but the per-op dispatch cost EXCEEDS the kernel speedup at
        #     batch=1, so it is net-SLOWER than fp32 eager (profiler-free A/B).
        #   - estimator_fp16_weights: cast DiT params to fp16 ONCE, then cast
        #     only the estimator input->fp16 and output->fp32 per call. Pure
        #     fp16 cuBLAS kernels with just 2 cast ops (no per-op policy query),
        #     so it captures the real 2-4x GEMM speedup.
        ac_dtype = getattr(self, "estimator_autocast_dtype", None)
        fp16_w = getattr(self, "estimator_fp16_weights", False)
        def _run_estimator(*a, **kw):
            if fp16_w:
                a16 = tuple(ai.to(torch.float16) if torch.is_tensor(ai) and ai.is_floating_point() else ai for ai in a)
                kw16 = {k: (v.to(torch.float16) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
                return est(*a16, **kw16).float()
            if ac_dtype is None:
                return est(*a, **kw)
            with torch.autocast(a[0].device.type, dtype=ac_dtype):
                return est(*a, **kw)
        prompt_len = prompt.size(-1)
        # prompt_x: holds the reference mel in the prompt region; x's prompt region is zeroed.
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0

        # ---- hoist all loop-invariant tensors out of the step loop ----
        # The CFG stacked buffers (prompt_x/style/mu with a zeroed uncond half)
        # never change shape or (mostly) content across steps; allocating them
        # once outside the loop removes per-step Python dispatch + kernel
        # launches that show up as host bubbles (profiler: 50.9% idle here).
        # Only stacked_x changes per step (new x), so we update just its halves
        # in-place instead of torch.cat each iteration.
        if inference_cfg_rate > 0:
            zero_prompt_x = torch.zeros_like(prompt_x)
            zero_style = torch.zeros_like(style)
            zero_mu = torch.zeros_like(mu)
            stacked_prompt_x = torch.cat([prompt_x, zero_prompt_x], dim=0)
            stacked_style = torch.cat([style, zero_style], dim=0)
            stacked_mu = torch.cat([mu, zero_mu], dim=0)
            x_lens_s = torch.cat([x_lens, x_lens], dim=0) if x_lens is not None else None
            # stacked_x lives across steps; refresh its two halves per step.
            stacked_x = torch.empty(2 * x.size(0), x.size(1), x.size(2),
                                    dtype=x.dtype, device=x.device)
            # precompute the two repeated t entries per step is cheap; keep as-is.
            cfg = float(inference_cfg_rate)
            one_plus_cfg = 1.0 + cfg

        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # refresh only the per-step-changing half of stacked_x (in-place)
                stacked_x[:x.size(0)].copy_(x)
                stacked_x[x.size(0):].copy_(x)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                stacked_dphi_dt = _run_estimator(
                    stacked_x, stacked_prompt_x, x_lens_s, stacked_t, stacked_style, stacked_mu,
                ).float()
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                # CFG formula (flow_matching.py:103): (1+cfg)*cond - cfg*uncond
                dphi_dt = one_plus_cfg * dphi_dt - cfg * cfg_dphi_dt
            else:
                dphi_dt = _run_estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu).float()

            x = x + dt * dphi_dt
            t = t + dt
            # keep the prompt region zeroed across steps
            x[:, :, :prompt_len] = 0

        return x

    # ------------------------------------------------------------------
    # CUDA Graph accelerated path (stage 4)
    # ------------------------------------------------------------------

    def solve_euler_graph(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor | None,
        t_span: torch.Tensor,
        inference_cfg_rate: float = 0.7,
    ) -> torch.Tensor:
        """CUDA-Graph-captured Euler solver (same math as solve_euler).

        The step loop body has static shapes per capture, so we capture one
        step into a CUDAGraph and replay it ``len(t_span)-1`` times. CFM is
        deterministic (no sampling RNG), so the whole loop is graph-capturable.

        Length bucketing: x is padded up to the nearest bucket (multiple of
        ``_GRAPH_BUCKET`` frames) so different text lengths reuse the same
        captured graph (avoids per-request re-capture, the prior failure mode).
        Output is sliced back to the true length afterwards.
        """
        prompt_len = prompt.size(-1)
        B, C, T_true = x.shape
        device = x.device
        dtype = x.dtype
        n_steps = len(t_span) - 1
        dt_val = (t_span[1] - t_span[0]).item()  # constant (linspace)
        cfg = float(inference_cfg_rate)

        # round T up to a bucket so similar lengths reuse one graph
        bucket = self._GRAPH_BUCKET
        T = ((T_true + bucket - 1) // bucket) * bucket
        if T != T_true:
            # pad x, prompt, mu to bucket T (tail content irrelevant — sliced off)
            pad = T - T_true
            x = torch.nn.functional.pad(x, (0, pad))
            prompt_padded = torch.nn.functional.pad(prompt, (0, pad))
            mu = torch.nn.functional.pad(mu.transpose(1, 2), (0, pad)).transpose(1, 2)
        else:
            prompt_padded = prompt

        # cache lookup keyed by BUCKETED shape (not exact T)
        key = (B, C, T, prompt_len, dtype, cfg, n_steps, getattr(self, "estimator_fp16_weights", False))
        cache = self._graph_cache.get(key)

        # set up prompt_x + zero x's prompt region (on bucketed tensors)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt_padded[..., :prompt_len]
        x[..., :prompt_len] = 0
        # x_lens for the attention mask: STATIC buffer (size 2, fixed address)
        # so the captured graph reads the same tensor every replay. We copy the
        # TRUE request length into it before replay — this keeps the mask
        # correct (only attends real frames) while letting the graph reuse
        # across requests of different lengths within the same bucket.
        x_lens_s = torch.zeros(2, dtype=torch.long, device=device)
        x_lens_s.fill_(T_true)  # init to true len so capture/warmup mask is correct

        if cache is None:
            zero_prompt_x = torch.zeros_like(prompt_x)
            zero_style = torch.zeros_like(style)
            zero_mu = torch.zeros_like(mu)
            s_prompt_x = torch.cat([prompt_x, zero_prompt_x], dim=0)
            s_style = torch.cat([style, zero_style], dim=0)
            s_mu = torch.cat([mu, zero_mu], dim=0)
            s_x = torch.cat([x, x], dim=0)
            t_buf = t_span[0].clone()
            dt_buf = torch.tensor(dt_val, device=device, dtype=dtype)
            s_t = torch.cat([t_buf.unsqueeze(0), t_buf.unsqueeze(0)], dim=0)
            # fp16-weights mode: estimator params are fp16, so all estimator
            # inputs must be fp16 (graph captures fixed dtype). Euler state
            # (x_buf) stays fp32 for accuracy; only the estimator call is fp16.
            fp16w = getattr(self, "estimator_fp16_weights", False)
            if fp16w:
                s_prompt_x = s_prompt_x.half()
                s_style = s_style.half()
                s_mu = s_mu.half()
                s_t = s_t.half()
            torch.cuda.synchronize()
            # warmup runs the estimator with the INITIAL x (pre-loop) so the
            # capture below sees warmed-up cuda allocator & cudnn plans.
            for _ in range(3):
                sd = self.estimator(s_x.half() if fp16w else s_x, s_prompt_x, x_lens_s, s_t, s_style, s_mu)
                dphi, cfg_dphi = sd.chunk(2, dim=0)
                dphi = (1.0 + cfg) * dphi - cfg * cfg_dphi
            torch.cuda.synchronize()
            s_x_buf = s_x.clone()
            s_t_buf = s_t.clone()
            x_buf = x.clone()
            # static keep-mask: True on valid frames [prompt_len:T_true],
            # False on prompt region and padding tail. Refreshed per-request
            # (content changes, address fixed) so the captured mul reads it.
            keep = torch.ones(1, C, T, dtype=torch.bool, device=device)
            keep[:, :, :prompt_len] = False
            keep[:, :, T_true:] = False
            self._graph_keep_mask = keep
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # fp16 mode: cast s_x_buf to fp16 for the estimator (it lives
                # fp32 for the replay loop's copy_(cat([x,x])) simplicity).
                if fp16w:
                    sd = self.estimator(s_x_buf.half(), s_prompt_x, x_lens_s, s_t_buf, s_style, s_mu)
                    dphi = sd.float()
                else:
                    sd = self.estimator(s_x_buf, s_prompt_x, x_lens_s, s_t_buf, s_style, s_mu)
                dphi, cfg_dphi = sd.chunk(2, dim=0)
                dphi = (1.0 + cfg) * dphi - cfg * cfg_dphi
                x_buf.copy_(x_buf + dt_buf * dphi)
                # Bug-1 fix: mask out both the prompt region AND the bucket
                # padding tail each Euler step, so WN reflect-pad conv cannot
                # leak padding/prompt values into the valid region. Uses a
                # static bool mask buffer (same address every replay); its
                # CONTENT is refreshed per-request (see cache hit branch), so
                # the captured mul still reads the correct mask each replay.
                x_buf.mul_(self._graph_keep_mask)
            cache = dict(
                graph=graph, s_x_buf=s_x_buf, s_t_buf=s_t_buf, x_buf=x_buf,
                s_prompt_x=s_prompt_x, s_style=s_style, s_mu=s_mu, x_lens_s=x_lens_s,
                fp16w=fp16w, keep_mask=self._graph_keep_mask, dt_buf=dt_buf,
            )
            self._graph_cache[key] = cache
        else:
            # refresh per-request values into static buffers (shapes match)
            fp16w = cache.get("fp16w", False)
            tgt = torch.float16 if fp16w else cache["s_prompt_x"].dtype
            cache["s_prompt_x"][: x.size(0)].copy_(prompt_x.to(tgt))
            cache["s_style"][: style.size(0)].copy_(style.to(tgt))
            cache["s_mu"][: mu.size(0)].copy_(mu.to(tgt))
            # x_lens_s is the cached static buffer; set it to true len
            cache["x_lens_s"].fill_(T_true)
            # refresh the keep-mask content for this request's true length
            # (prompt region [0:prompt_len] and padding tail [T_true:T] masked)
            km = cache["keep_mask"]
            km.fill_(True)
            km[:, :, :prompt_len] = False
            km[:, :, T_true:] = False

        # ---- replay loop (capture-free after first call) ----
        g = cache["graph"]
        s_x_buf = cache["s_x_buf"]
        s_t_buf = cache["s_t_buf"]
        x_buf = cache["x_buf"]
        x_buf.copy_(x)  # init accumulator from current x
        for step in range(1, n_steps + 1):
            s_x_buf.copy_(torch.cat([x, x], dim=0))
            t_val = t_span[step - 1]
            s_t_buf.copy_(torch.cat([t_val.unsqueeze(0), t_val.unsqueeze(0)], dim=0))
            g.replay()
            x.copy_(x_buf)
        # slice bucketed result back to the true (un-padded) length
        return x[:, :, :T_true]


class S2Mel(nn.Module):
    """Full S2Mel module: length_regulator + CFM(DiT).

    This is the top-level S2Mel used by the inference pipeline. ``gpt_layer``
    (use_gpt_latent path) is NOT loaded by default (config default off).
    """

    def __init__(self, length_regulator: InterpolateRegulator, cfm: S2MelCFM):
        super().__init__()
        self.length_regulator = length_regulator
        self.cfm = cfm

    def length_regulate(
        self,
        spk_cond: torch.Tensor,
        s_infer: torch.Tensor,
        ref_mel: torch.Tensor,
        duration_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the CFM condition from ref + target.

        Returns (prompt_condition, cond, cat_condition). Mirrors infer_v2_5.py:651-862.
        """
        device = spk_cond.device
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)
        prompt_condition = self.length_regulator(
            spk_cond, ylens=ref_target_lengths, n_quantizers=3, f0=None
        )[0]

        target_lengths = torch.LongTensor(
            [int(s_infer.shape[1] * 1.72 * duration_factor)]
        ).to(device)
        cond = self.length_regulator(
            s_infer, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]

        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        return prompt_condition, cond, cat_condition

    @torch.no_grad()
    def inference(
        self,
        spk_cond: torch.Tensor,
        s_infer: torch.Tensor,
        ref_mel: torch.Tensor,
        style: torch.Tensor,
        duration_factor: float = 1.0,
        n_timesteps: int = 25,
        inference_cfg_rate: float = 0.7,
        use_graph: bool = False,
    ) -> torch.Tensor:
        """End-to-end S2Mel: condition → CFM → mel (prompt stripped).

        Args:
            spk_cond: normalized w2v-bert feat [B, T_w2v, 1024] (for prompt_condition).
            s_infer: EnhancedCodec.decode output [B, T_codec, 1024].
            ref_mel: reference mel [B, 80, T_ref].
            style: [B, 192].
            duration_factor: scales target length (1.72 * factor).
        Returns:
            mel [B, 80, T_target] (prompt region stripped).
        """
        _, _, cat_condition = self.length_regulate(
            spk_cond, s_infer, ref_mel, duration_factor
        )
        x_lens = torch.LongTensor([cat_condition.size(1)]).to(cat_condition.device)
        vc = self.cfm.inference(
            cat_condition, x_lens, ref_mel, style, None,
            n_timesteps=n_timesteps, inference_cfg_rate=inference_cfg_rate,
            use_graph=use_graph,
        )
        # strip prompt region
        return vc[:, :, ref_mel.size(-1):]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")
    from windextts.weights import WeightLoader

    DUMPS = "/root/windextts_dumps"
    dev = "cuda"

    w = WeightLoader()
    net = w.load_s2mel()

    from windextts.config import load_default_config

    cfg = load_default_config()

    # Build components.
    lr = InterpolateRegulator(
        channels=cfg.s2mel.length_reg.channels,
        sampling_ratios=cfg.s2mel.length_reg.sampling_ratios,
        is_discrete=cfg.s2mel.length_reg.is_discrete,
        in_channels=cfg.s2mel.length_reg.in_channels,
        codebook_size=cfg.s2mel.length_reg.content_codebook_size,
    )
    lr.load_official(net["length_regulator"])
    lr = lr.to(dev).eval()

    dit = DiT()  # uses its own DiTConfig defaults (match s2mel.DiT config)
    dit.load_official(net["cfm"])  # flat keys with 'estimator.' prefix, stripped inside
    dit = dit.to(dev).eval()

    cfm = S2MelCFM(dit, in_channels=cfg.s2mel.dit.in_channels).to(dev).eval()
    s2mel = S2Mel(lr, cfm).to(dev).eval()

    # End-to-end alignment vs the fixed-seed dump (seed=123).
    spk_cond = torch.load(f"{DUMPS}/gpt.spk_cond_w2v.pt", weights_only=False).to(dev)
    s_infer = torch.load(f"{DUMPS}/s2mel.S_infer.pt", weights_only=False).to(dev)
    ref_mel = torch.load(f"{DUMPS}/s2mel.ref_mel.pt", weights_only=False).to(dev)
    style = torch.load(f"{DUMPS}/s2mel.style.pt", weights_only=False).to(dev)

    torch.cuda.manual_seed(123)
    out = s2mel.inference(spk_cond, s_infer, ref_mel, style, duration_factor=1.0)
    ref = torch.load(f"{DUMPS}/s2mel.cfm_output_mel_seed123.pt", weights_only=False).to(dev)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"end-to-end S2Mel mel: {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    print(f"max_abs_diff = {diff:.3e}")
    print(f"allclose(atol=1e-3, rtol=1e-3) = {torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3)}")
    print("SMOKE", "OK" if diff < 1e-2 else "FAIL")
