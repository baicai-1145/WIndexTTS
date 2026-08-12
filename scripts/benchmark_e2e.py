#!/usr/bin/env python
"""End-to-end benchmark: WIndexTTS vs official IndexTTS (same A10G, fp32).

Runs steady-state inference (warmup, then N texts) and reports per-request
wall-clock latency + RTF. Both engines run fp32 with their default settings
(official: use_bf16=False; WIndexTTS: GPT-AR CUDA Graph enabled).

Usage:
    /root/index-tts/.venv/bin/python scripts/benchmark_e2e.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/WIndexTTS")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TEST_TEXTS = [
    "大家好，这是一个测试。",
    "今天天气真不错。",
    "人工智能正在改变世界。",
    "语音合成技术近年来发展迅速。",
]
REF_AUDIO = "/root/WIndexTTS/test.wav"


def bench_windextts() -> None:
    from windextts.inference import WIndexTTS

    tts = WIndexTTS(device="cuda")
    # warmup (graph capture + cudnn autotune)
    tts.infer(REF_AUDIO, "warmup", "ZH")
    times = []
    for txt in TEST_TEXTS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sr, wav = tts.infer(REF_AUDIO, txt, "ZH")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    dur = wav.numel() / sr
    print(f"[WIndexTTS] texts={len(times)}  audio~{dur:.2f}s")
    print(f"  mean={statistics.mean(times):.2f}s  min={min(times):.2f}s  "
          f"max={max(times):.2f}s  RTF(best)={dur/min(times):.2f}x")


def bench_official() -> None:
    os.environ["HF_HUB_CACHE"] = "/root/IndexTTS-2.5/checkpoints/hf_cache"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path="/root/IndexTTS-2.5/config.yaml", model_dir="/root/IndexTTS-2.5",
        use_bf16=False, use_cuda_kernel=False, use_torch_compile=False, use_qwen_emo=False,
    )
    out = "/root/windextts_dumps/_bench_off.wav"
    tts.infer(REF_AUDIO, "warmup", out, lang="ZH", verbose=False)
    times = []
    for txt in TEST_TEXTS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tts.infer(REF_AUDIO, txt, out, lang="ZH", verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    import soundfile as sf
    info = sf.info(out)
    dur = info.frames / info.samplerate
    print(f"[official]   texts={len(times)}  audio~{dur:.2f}s")
    print(f"  mean={statistics.mean(times):.2f}s  min={min(times):.2f}s  "
          f"max={max(times):.2f}s  RTF(best)={dur/min(times):.2f}x")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "windex"):
        bench_windextts()
    if which in ("all", "official"):
        bench_official()
