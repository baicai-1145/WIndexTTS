#!/usr/bin/env python
"""Benchmark vLLM-Omni IndexTTS-2.5 steady-state (warmup + multiple texts).

Usage:
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /root/vllm-omni/.venv/bin/python scripts/bench_vllmomni.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
import soundfile as sf  # noqa: E402

from vllm_omni import Omni  # noqa: E402
from vllm_omni.model_executor.models.indextts2.prompt_utils import (  # noqa: E402
    build_indextts2_prefill_prompt_ids,
)

MODEL = "/root/IndexTTS-2.5"
REF_AUDIO = "/root/WIndexTTS/test.wav"
TEXTS = [
    "周末的清晨，阳光透过窗帘洒进房间，带来一丝温暖的气息。",
    "这款新产品采用了全新的设计理念，性能提升了百分之三十。",
    "随着科技的不断进步，人们的生活方式发生了翻天覆地的变化，处处都能感受到智能化带来的便利。",
    "她站在山顶，望着远处的云海，心里涌起一阵难以言喻的感动。",
]


def build_request(text: str) -> dict:
    additional = {"text": [text], "voice": [REF_AUDIO], "lang": ["zh"],
                  "duration_factor": [1.0]}
    prompt_kwargs = {"model_type": "indextts2_5", "lang": "zh",
                     "text_normalization": True}
    return {
        "prompt_token_ids": build_indextts2_prefill_prompt_ids(MODEL, text, **prompt_kwargs),
        "additional_information": additional,
    }


def extract_audio(mm):
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [c.reshape(-1) for c in audio if isinstance(c, torch.Tensor) and c.numel() > 0]
        audio = torch.cat(chunks, dim=0) if chunks else None
    sr_val = mm.get("sr")
    if isinstance(sr_val, list):
        sr_val = sr_val[-1] if sr_val else None
    if sr_val is None:
        sample_rate = 22050
    elif hasattr(sr_val, "item"):
        sample_rate = int(sr_val.item())
    elif isinstance(sr_val, torch.Tensor):
        sample_rate = int(sr_val.flatten()[0].item())
    else:
        sample_rate = int(sr_val)
    return (audio if isinstance(audio, torch.Tensor) else None), sample_rate


def run_one(omni, sampling_params, text):
    req = build_request(text)
    last_mm = None
    for omni_out in omni.generate(req, sampling_params_list=sampling_params):
        last_mm = omni_out.multimodal_output
    return extract_audio(last_mm or {})


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "indextts2_5"
    deploy_config = str(Path(f"/root/vllm-omni/vllm_omni/deploy/{cfg_name}.yaml"))
    print(f">> initializing Omni engine ({cfg_name}, 2-4 min for compile+capture)...")
    omni = Omni(model=MODEL, deploy_config=deploy_config, stage_init_timeout=600)

    gpt = SamplingParams(temperature=0.8, top_p=0.8, top_k=30, max_tokens=1500,
                         repetition_penalty=10.0, stop_token_ids=[8193], seed=42, detokenize=False)
    s2mel = SamplingParams(temperature=0.0, max_tokens=65536, detokenize=True)
    sp = [gpt, s2mel]

    # warmup (covers Triton JIT compilation spikes)
    print(">> warmup (1 text, covers JIT compile)...")
    t0 = time.perf_counter()
    audio, sr = run_one(omni, sp, "warmup 测试。")
    torch.cuda.synchronize()
    print(f"   warmup took {time.perf_counter()-t0:.1f}s")

    times = []
    for txt in TEXTS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio, sr = run_one(omni, sp, txt)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        dur = (audio.numel() / sr) if audio is not None else 0
        print(f"   {dt:.3f}s -> {dur:.2f}s audio")
        if audio is not None and len(times) == 1:
            sf.write("/root/windextts_dumps/vllmomni_sample.wav", audio.float().cpu().numpy(), sr)

    print(f"RESULT_vllmomni_{cfg_name}: mean={statistics.mean(times):.3f}s "
          f"min={min(times):.3f}s max={max(times):.3f}s")


if __name__ == "__main__":
    main()
