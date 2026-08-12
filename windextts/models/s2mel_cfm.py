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
        self._graph_cache: dict = {}  # keyed by (x.shape, prompt_len, dtype, cfg, n_steps)

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
        Returns:
            mel [B, 80, T] including the prompt region (caller strips it).
        """
        B, T = mu.size(0), mu.size(1)
        device = mu.device
        # DiT transformer caches (RoPE freqs_cis, optional KV) must be sized to T.
        self.estimator.setup_caches(max_batch_size=2 * B, max_seq_length=T)
        z = torch.randn([B, self.in_channels, T], device=device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mu.dtype)
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

        Replicates flow_matching.py:57-111 exactly.
        """
        prompt_len = prompt.size(-1)
        # prompt_x: holds the reference mel in the prompt region; x's prompt region is zeroed.
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0

        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Batched CFG: stack [cond, uncond(null)] for a single estimator forward.
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
                )
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                # CFG formula (flow_matching.py:103): (1+cfg)*cond - cfg*uncond
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

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

        The 25-step loop body has fully static shapes (x/prompt/mu/style don't
        change shape, only values), so we capture one step into a CUDAGraph and
        replay it ``len(t_span)-1`` times. Sampling RNG (multinomial) is absent
        here (CFM is deterministic given x0 noise), so the whole loop is
        graph-capturable.

        Per-step buffers (stacked cond/uncond, t) are pre-allocated once and
        updated in-place via copy_; the graph reads/writes only static tensors.
        """
        prompt_len = prompt.size(-1)
        device = x.device
        dtype = x.dtype
        n_steps = len(t_span) - 1
        dt_val = (t_span[1] - t_span[0]).item()  # constant (linspace)
        cfg = float(inference_cfg_rate)

        # cache lookup (capture cost paid only on first call per shape)
        key = (tuple(x.shape), prompt_len, dtype, cfg, n_steps)
        cache = self._graph_cache.get(key)

        # set up prompt_x + zero x's prompt region (same as solve_euler)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        x_lens_s = torch.cat([x_lens, x_lens], dim=0) if x_lens is not None else None

        if cache is None:
            # ---- first call: pre-allocate buffers + warmup + capture ----
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
            torch.cuda.synchronize()
            for _ in range(3):  # warmup (primes cudnn autotune, allocs)
                sd = self.estimator(s_x, s_prompt_x, x_lens_s, s_t, s_style, s_mu)
                dphi, cfg_dphi = sd.chunk(2, dim=0)
                dphi = (1.0 + cfg) * dphi - cfg * cfg_dphi
            torch.cuda.synchronize()
            # static buffers the graph reads/writes
            s_x_buf = s_x.clone()
            s_t_buf = s_t.clone()
            x_buf = x.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sd = self.estimator(s_x_buf, s_prompt_x, x_lens_s, s_t_buf, s_style, s_mu)
                dphi, cfg_dphi = sd.chunk(2, dim=0)
                dphi = (1.0 + cfg) * dphi - cfg * cfg_dphi
                x_buf.copy_(x_buf + dt_buf * dphi)
                x_buf[..., :prompt_len] = 0
            cache = dict(
                graph=graph, s_x_buf=s_x_buf, s_t_buf=s_t_buf, x_buf=x_buf,
                s_prompt_x=s_prompt_x, s_style=s_style, s_mu=s_mu, x_lens_s=x_lens_s,
            )
            self._graph_cache[key] = cache
        else:
            # refresh per-request values into static buffers (shapes match)
            cache["s_prompt_x"][: x.size(0)].copy_(prompt_x)
            cache["s_style"][: style.size(0)].copy_(style)
            cache["s_mu"][: mu.size(0)].copy_(mu)

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
        return x


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
