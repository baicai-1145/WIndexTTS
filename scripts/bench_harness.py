"""Benchmark harness for WIndexTTS optimization rounds.

Reusable profiler: measures per-stage latency + end-to-end RTF, with warmup
and steady-state averaging. Used to quantify each optimization round.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/WIndexTTS")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REF_AUDIO = "/root/WIndexTTS/test.wav"
TEXTS = [
    "大家好，这是一个测试。",
    "今天天气真不错。",
    "人工智能正在改变世界。",
    "语音合成技术发展迅速。",
]


def profile_stages(tts) -> dict:
    """Per-stage latency (ms), warm, using cached ref + precomputed conds."""
    # ensure ref cache populated
    if not tts._ref_cache:
        from windextts.inference import REF_SR_W2V, REF_SR_MEL, REF_MAX_SECONDS
        import torchaudio
        audio, sr = tts._load_audio(REF_AUDIO, REF_MAX_SECONDS)
        a16 = torchaudio.transforms.Resample(sr, REF_SR_W2V)(audio)
        a22 = torchaudio.transforms.Resample(sr, REF_SR_MEL)(audio).to(tts.device).float()
        spk0 = tts.extract_spk_cond(a16); style0 = tts.extract_style(a16); refmel0 = tts.mel_fn(a22)
        import os as _os
        tts._ref_cache[(REF_AUDIO, _os.path.getmtime(REF_AUDIO))] = (spk0, style0, refmel0)
    spk, style, refmel = tts._ref_cache[list(tts._ref_cache)[0]]
    dev = tts.device
    toks = tts.tokenizer.encode("<|zh|> " + TEXTS[0], allowed_special="all")
    tt = torch.nn.functional.pad(torch.IntTensor(toks).unsqueeze(0).to(dev), (0, 1), value=1)
    from windextts.frontend.tokenizer import lang_to_token
    lang = torch.LongTensor([lang_to_token("ZH")]).to(dev)
    emo = tts.build_emo_vec(style)
    conds = tts.gpt.build_conds_latent(style, emo)
    # warm graph
    _ = tts.gpt.generate(conds, tt, lang, max_new_tokens=200, do_sample=False,
                         stop_token=8193, use_cuda_graph=True)

    def bench(fn, n=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / n

    e = {}
    e["gpt_ar"] = bench(lambda: tts.gpt.generate(
        conds, tt, lang, max_new_tokens=200, do_sample=False,
        stop_token=8193, use_cuda_graph=True))
    codes = tts.gpt.generate(conds, tt, lang, max_new_tokens=200,
                             do_sample=False, stop_token=8193, use_cuda_graph=True)
    codes = codes[:, :-1] if codes[0, -1] == 8193 else codes
    s = tts.codec.decode(codes)
    # enable teacache (matches infer() default) for fair S2Mel profiling
    est = tts.s2mel.cfm.estimator
    if not getattr(est, "teacache_enabled", False):
        est.enable_teacache(thresh=0.15)
    e["s2mel"] = bench(lambda: tts.s2mel.inference(spk, s, refmel, style))
    mel = tts.s2mel.inference(spk, s, refmel, style)
    e["bigvgan"] = bench(lambda: tts.bigvgan(mel))
    return e


def bench_e2e(tts) -> dict:
    """End-to-end steady-state (cached ref)."""
    _ = tts.infer(REF_AUDIO, "warmup", "ZH")
    times = []
    for txt in TEXTS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sr, wav = tts.infer(REF_AUDIO, txt, "ZH")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "rtf_best": (wav.numel() / sr) / min(times),
    }


def main():
    from windextts.inference import WIndexTTS

    tts = WIndexTTS(device="cuda")
    stages = profile_stages(tts)
    total = sum(stages.values())
    print("=== STAGE PROFILE (ms) ===")
    for k, v in stages.items():
        print(f"  {k:10s} {v:6.0f}ms  {100*v/total:4.0f}%")
    print(f"  {'TOTAL':10s} {total:6.0f}ms")
    e2e = bench_e2e(tts)
    print("=== END-TO-END (cached ref) ===")
    print(f"  mean={e2e['mean']:.2f}s min={e2e['min']:.2f}s "
          f"RTF(best)={e2e['rtf_best']:.2f}x")


if __name__ == "__main__":
    main()
