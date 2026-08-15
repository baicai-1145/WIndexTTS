"""S2Mel-CFM: Euler solver + CFG around the DiT estimator (pure torch).
Re-implements indextts/s2mel/modules/flow_matching.py (BASECFM): z=randn*T,
t_span=linspace(0,1,n+1), prompt region zeroed in x/prompt_x, CFG stacks
cond+uncond batches through one estimator call. Deterministic given seed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from windextts.models.length_regulator import InterpolateRegulator
from windextts.models.s2mel_dit import DiT

class S2MelCFM(nn.Module):
    # CFM wrapper: DiT estimator + Euler ODE solver with CFG.

    def __init__(self, estimator: DiT, in_channels: int = 80, sigma_min: float = 1e-6):
        super().__init__()
        self.estimator = estimator
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self._graph_cache: dict = {}  # key: (B,C,T_bucket,prompt_len,dtype,cfg,n_steps)
        self._GRAPH_BUCKET = 64  # mel-frame bucket: different lengths reuse one captured graph

    @torch.no_grad()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps=25,
                  temperature=1.0, inference_cfg_rate=0.7, use_graph=False):
        B, T = mu.size(0), mu.size(1)
        device = mu.device
        # GRAPH MODE: freqs_cis is precomputed from block_size (16384) so it
        # never needs rebuilding for different T, but setup_caches rebuilds it
        # whenever max_seq_length grows — a new tensor address invalidates
        # captured graphs (garbage RoPE → brick audio). Pin a large fixed size
        # in graph mode so setup_caches builds once and returns early after.
        cache_T = T
        if use_graph:
            bucket = getattr(self, "_GRAPH_BUCKET", 64)
            cache_T = ((T + bucket - 1) // bucket) * bucket
            cache_T = max(cache_T, getattr(self, "_GRAPH_PIN_SEQ", 256))
        self.estimator.setup_caches(max_batch_size=2 * B, max_seq_length=cache_T)
        z = torch.randn([B, self.in_channels, T], device=device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mu.dtype)
        if use_graph:
            return self.solve_euler_graph(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.7):
        # Reduced-precision DiT strategies:
        #  - estimator_autocast_dtype: per-op torch.autocast — fp16-safe but the
        #    per-op dispatch cost exceeds the kernel speedup at batch=1.
        #  - estimator_fp16_weights: params cast once, estimator input->fp16 +
        #    output->fp32 per call — pure fp16 cuBLAS, real 2-4x GEMM speedup.
        ac_dtype = getattr(self, "estimator_autocast_dtype", None)
        fp16_w = getattr(self, "estimator_fp16_weights", False)
        est = self.estimator
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
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]  # prompt_x holds ref mel
        x[..., :prompt_len] = 0
        # Hoist loop-invariant CFG buffers (only stacked_x changes per step;
        # profiler: 50.9% idle was per-step Python dispatch + kernel launches).
        if inference_cfg_rate > 0:
            zero_prompt_x = torch.zeros_like(prompt_x)
            zero_style = torch.zeros_like(style)
            zero_mu = torch.zeros_like(mu)
            stacked_prompt_x = torch.cat([prompt_x, zero_prompt_x], dim=0)
            stacked_style = torch.cat([style, zero_style], dim=0)
            stacked_mu = torch.cat([mu, zero_mu], dim=0)
            x_lens_s = torch.cat([x_lens, x_lens], dim=0) if x_lens is not None else None
            stacked_x = torch.empty(2 * x.size(0), x.size(1), x.size(2), dtype=x.dtype, device=x.device)
            cfg = float(inference_cfg_rate)
            one_plus_cfg = 1.0 + cfg

        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                stacked_x[:x.size(0)].copy_(x)  # refresh only the per-step half (in-place)
                stacked_x[x.size(0):].copy_(x)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                stacked_dphi_dt = _run_estimator(
                    stacked_x, stacked_prompt_x, x_lens_s, stacked_t, stacked_style, stacked_mu,
                ).float()
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = one_plus_cfg * dphi_dt - cfg * cfg_dphi_dt  # (1+cfg)*cond - cfg*uncond
            else:
                dphi_dt = _run_estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu).float()
            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0  # keep prompt region zeroed across steps
        return x

    # CUDA Graph path: one static-shape step captured, replayed n_steps times
    def solve_euler_graph(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.7):
        prompt_len = prompt.size(-1)
        B, C, T_true = x.shape
        device = x.device
        dtype = x.dtype
        n_steps = len(t_span) - 1
        dt_val = (t_span[1] - t_span[0]).item()  # constant (linspace)
        cfg = float(inference_cfg_rate)

        bucket = self._GRAPH_BUCKET  # round T up so similar lengths reuse one graph
        T = ((T_true + bucket - 1) // bucket) * bucket
        if T != T_true:
            pad = T - T_true  # tail content irrelevant — sliced off
            x = torch.nn.functional.pad(x, (0, pad))
            prompt_padded = torch.nn.functional.pad(prompt, (0, pad))
            mu = torch.nn.functional.pad(mu.transpose(1, 2), (0, pad)).transpose(1, 2)
        else:
            prompt_padded = prompt

        key = (B, C, T, prompt_len, dtype, cfg, n_steps, getattr(self, "estimator_fp16_weights", False))
        cache = self._graph_cache.get(key)

        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt_padded[..., :prompt_len]
        x[..., :prompt_len] = 0
        # x_lens_s: STATIC buffer (fixed address) so the graph reads the same
        # tensor every replay; copy the TRUE length in before replay.
        x_lens_s = torch.zeros(2, dtype=torch.long, device=device)
        x_lens_s.fill_(T_true)

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
            # inputs must be fp16 (graph captures fixed dtype); Euler state
            # (x_buf) stays fp32 — only the estimator call is fp16.
            fp16w = getattr(self, "estimator_fp16_weights", False)
            if fp16w:
                s_prompt_x = s_prompt_x.half()
                s_style = s_style.half()
                s_mu = s_mu.half()
                s_t = s_t.half()
            torch.cuda.synchronize()
            for _ in range(3):  # warmup so capture sees warmed allocator/cudnn
                sd = self.estimator(s_x.half() if fp16w else s_x, s_prompt_x, x_lens_s, s_t, s_style, s_mu)
                dphi, cfg_dphi = sd.chunk(2, dim=0)
                dphi = (1.0 + cfg) * dphi - cfg * cfg_dphi
            torch.cuda.synchronize()
            s_x_buf = s_x.clone()
            s_t_buf = s_t.clone()
            x_buf = x.clone()
            # static keep-mask: valid frames [prompt_len:T_true]; prompt region
            # and padding tail masked. Content refreshed per-request, address
            # fixed, so the captured mul reads the right mask every replay.
            keep = torch.ones(1, C, T, dtype=torch.bool, device=device)
            keep[:, :, :prompt_len] = False
            keep[:, :, T_true:] = False
            self._graph_keep_mask = keep
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # fp16 mode: s_x_buf cast to fp16 here (lives fp32 so the replay
                # loop's copy_(cat([x,x])) stays simple).
                if fp16w:
                    sd = self.estimator(s_x_buf.half(), s_prompt_x, x_lens_s, s_t_buf, s_style, s_mu)
                    dphi = sd.float()
                else:
                    sd = self.estimator(s_x_buf, s_prompt_x, x_lens_s, s_t_buf, s_style, s_mu)
                dphi, cfg_dphi = sd.chunk(2, dim=0)
                dphi = (1.0 + cfg) * dphi - cfg * cfg_dphi
                x_buf.copy_(x_buf + dt_buf * dphi)
                # Bug-1 fix: mask prompt region AND bucket padding tail each step
                # so WN reflect-pad conv cannot leak into the valid region.
                x_buf.mul_(self._graph_keep_mask)
            cache = dict(
                graph=graph, s_x_buf=s_x_buf, s_t_buf=s_t_buf, x_buf=x_buf,
                s_prompt_x=s_prompt_x, s_style=s_style, s_mu=s_mu, x_lens_s=x_lens_s,
                fp16w=fp16w, keep_mask=self._graph_keep_mask, dt_buf=dt_buf,
            )
            self._graph_cache[key] = cache
        else:
            fp16w = cache.get("fp16w", False)  # refresh per-request values into static buffers
            tgt = torch.float16 if fp16w else cache["s_prompt_x"].dtype
            cache["s_prompt_x"][: x.size(0)].copy_(prompt_x.to(tgt))
            cache["s_style"][: style.size(0)].copy_(style.to(tgt))
            cache["s_mu"][: mu.size(0)].copy_(mu.to(tgt))
            cache["x_lens_s"].fill_(T_true)
            km = cache["keep_mask"]  # refresh keep-mask for this request's true length
            km.fill_(True)
            km[:, :, :prompt_len] = False
            km[:, :, T_true:] = False

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
        return x[:, :, :T_true]  # slice bucketed result back to true length


class S2Mel(nn.Module):
    # Full S2Mel: length_regulator + CFM(DiT). gpt_layer not loaded (config off).

    def __init__(self, length_regulator: InterpolateRegulator, cfm: S2MelCFM):
        super().__init__()
        self.length_regulator = length_regulator
        self.cfm = cfm

    def length_regulate(self, spk_cond, s_infer, ref_mel, duration_factor=1.0):
        # (prompt_condition, cond, cat_condition). Mirrors infer_v2_5.py:651-862.
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
    def inference(self, spk_cond, s_infer, ref_mel, style, duration_factor=1.0,
                  n_timesteps=25, inference_cfg_rate=0.7, use_graph=False):
        _, _, cat_condition = self.length_regulate(spk_cond, s_infer, ref_mel, duration_factor)
        x_lens = torch.LongTensor([cat_condition.size(1)]).to(cat_condition.device)
        vc = self.cfm.inference(
            cat_condition, x_lens, ref_mel, style, None,
            n_timesteps=n_timesteps, inference_cfg_rate=inference_cfg_rate,
            use_graph=use_graph,
        )
        return vc[:, :, ref_mel.size(-1):]  # strip prompt region
