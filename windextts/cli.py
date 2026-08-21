"""WIndexTTS CLI — synthesize speech from the command line.

Usage:
    # one command, auto backend: NVIDIA GPU -> CUDA torch, Apple Silicon -> MLX
    # (override with WINDEXTTS_BACKEND=mlx|torch|cuda)

    # install model weights first (~7.5GB, one time, resumable)
    windextts --install-model --model-dir /path/to/IndexTTS-2.5

    # basic (fp16, GPU)
    windextts --ref ref.wav --text "你好世界" -o out.wav

    # W4A16 quantized GPT (fastest) — --quantize is the new alias of --w4a16
    windextts --ref ref.wav --text "你好世界" -o out.wav --quantize

    # low VRAM (3-4GB GPUs; keeps beam3, ~2.9GB steady; CUDA only)
    windextts --ref ref.wav --text "你好世界" -o out.wav --quantize --low-vram

    # from a text file, with emotion vector + duration control
    windextts --ref ref.wav --text-file story.txt -o out.wav \
        --emo-vector 0.8,0,0,0,0,0,0.2,0 --duration 1.1

    # fp32 reference precision (--fp32 = alias of --dtype fp32), greedy (--beams 1)
    windextts --ref ref.wav --text "hello" -o out.wav --fp32 --beams 1 --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="windextts",
        description="WIndexTTS: pure-torch accelerated IndexTTS-2.5 inference "
                    "(zero JIT compile, Windows friendly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1] if __doc__ else None,
    )
    p.add_argument("--ref", required=True, help="reference audio path (5-15s clean speech)")
    p.add_argument("--text", help="text to synthesize (or use --text-file)")
    p.add_argument("--text-file", help="read text from a file (UTF-8)")
    p.add_argument("-o", "--output", default="output.wav", help="output wav path")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fp32", action="store_true", help="fp32 weights (highest precision, ~7.6GB); alias of --dtype fp32")
    mode.add_argument("--w4a16", action="store_true",
                      help="W4A16 INT4 GPT quantization (fastest; needs torchao)")
    mode.add_argument("--quantize", action="store_true",
                      help="alias of --w4a16 (W4A16 INT4 GPT quantization)")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],
                   help="compute precision (default fp16)")
    p.add_argument("--low-vram", action="store_true",
                   help="low-VRAM mode for 3-4GB GPUs (w2v streamed to CPU, "
                        "beam3 kept, ~2.9GB steady)")
    p.add_argument("--model-dir", default=None, help="weights directory (gpt.pth, config.yaml, ...)")

    p.add_argument("--lang", default="ZH",
                   help="language token (ZH/EN/JA/KO/YUE/..., default ZH)")
    p.add_argument("--duration", type=float, default=1.0,
                   help="duration factor (official 1.72 scale × factor, default 1.0)")

    emo = p.add_mutually_exclusive_group()
    emo.add_argument("--emo-vector",
                     help="8-dim emotion weights happy,angry,sad,afraid,disgusted,"
                          "melancholic,surprised,calm — e.g. 0.8,0,0,0,0,0,0.2,0")
    emo.add_argument("--emo-text", help="free-text emotion description (QwenEmotion)")
    emo.add_argument("--emo-ref", help="emotion reference audio path (conformer path)")

    p.add_argument("--greedy", action="store_true", help="greedy decode (beam=1, deterministic; overrides --beams)")
    p.add_argument("--beams", type=int, default=3,
                   help="GPT-AR beam count (default 3; 1 = greedy)")
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)

    p.add_argument("--cfm-steps", type=int, default=12,
                   help="S2Mel CFM Euler steps (official 25, default 12)")
    p.add_argument("--no-normalize", action="store_true",
                   help="skip text normalization (G2P digits/punctuation)")
    p.add_argument("--segment-tokens", type=int, default=120,
                   help="max text tokens per segment (long texts auto-split)")
    p.add_argument("--verbose", action="store_true", help="print per-run timing + VRAM")

    # ModelScope direct links, no SDK dependency, resumable
    p.add_argument("--install-model", action="store_true",
                   help="download model weights to --model-dir (or WINDEXTTS_WEIGHTS_DIR) "
                        "and exit; ~7.5GB from ModelScope direct links. Resumable. "
                        "Combine with --skip-qwen.")
    p.add_argument("--skip-qwen", action="store_true",
                   help="with --install-model: skip qwen0.6bemo4-merge (1.2GB, only "
                        "needed for emo_text)")
    return p


def _pick_backend() -> str:
    """Choose 'mlx' or 'torch' without importing either runtime (cheap + safe)."""
    env = os.environ.get("WINDEXTTS_BACKEND", "").strip().lower()
    if env:
        if env == "mlx":
            return "mlx"
        if env in ("torch", "cuda"):
            return "torch"
        print(f"warning: unknown WINDEXTTS_BACKEND={env!r} (expected mlx|torch|cuda); "
              "auto-detecting", file=sys.stderr)
    if platform.machine().lower() in ("arm64", "aarch64") and importlib.util.find_spec("mlx"):
        return "mlx"
    return "torch"


def dispatch_main(argv: list[str] | None = None) -> int:
    """Unified entry: route to the MLX or torch CLI by backend autodetect."""
    if _pick_backend() == "mlx":
        from windextts_mlx.cli import main as mlx_main
        return mlx_main(argv)
    return main(argv)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.install_model:
        from windextts.download import download_model
        target = args.model_dir or os.environ.get("WINDEXTTS_WEIGHTS_DIR")
        if not target:
            print("error: --install-model needs --model-dir or WINDEXTTS_WEIGHTS_DIR",
                  file=sys.stderr)
            return 2
        download_model(target, include_qwen=not args.skip_qwen)
        return 0

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("error: either --text or --text-file is required", file=sys.stderr)
        return 2
    text = text.strip()
    if not text:
        print("error: empty text", file=sys.stderr)
        return 2

    emo_vector = None
    if args.emo_vector:
        try:
            emo_vector = [float(x) for x in args.emo_vector.split(",")]
        except ValueError:
            print("error: --emo-vector must be 8 comma-separated floats", file=sys.stderr)
            return 2
        if len(emo_vector) != 8:
            print("error: --emo-vector needs exactly 8 values "
                  "(happy,angry,sad,afraid,disgusted,melancholic,surprised,calm)", file=sys.stderr)
            return 2

    import torch

    if not torch.cuda.is_available():
        print("error: CUDA is not available; the torch backend requires an NVIDIA GPU.\n"
              "       On Apple Silicon the `windextts` entry auto-selects the MLX backend\n"
              "       (or force it with WINDEXTTS_BACKEND=mlx).", file=sys.stderr)
        return 1

    from windextts.inference import WIndexTTS

    use_w4a16 = args.w4a16 or args.quantize
    use_fp32 = args.fp32 or args.dtype == "fp32"
    if use_fp32 and use_w4a16:
        print("error: --fp32/--dtype fp32 conflicts with --w4a16/--quantize", file=sys.stderr)
        return 2
    num_beams = 1 if args.greedy else max(1, args.beams)

    dtype = torch.float32 if use_fp32 else torch.float16
    t0 = time.perf_counter()
    tts = WIndexTTS(
        weights_dir=args.model_dir, device="cuda", dtype=dtype,
        enable_w4a16=use_w4a16, low_vram=args.low_vram,
    )
    tts.warmup()
    if args.verbose:
        print(f"[load+warmup] {time.perf_counter() - t0:.1f}s  "
              f"VRAM alloc {torch.cuda.memory_allocated() / 1e9:.2f}GB", file=sys.stderr)

    t0 = time.perf_counter()
    sr, audio = tts.infer(
        spk_audio_prompt=args.ref,
        text=text,
        lang=args.lang,
        emo_vector=emo_vector,
        emo_text=args.emo_text,
        emo_ref_path=args.emo_ref,
        duration_factor=args.duration,
        do_sample=not args.greedy,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        cfm_steps=args.cfm_steps,
        text_normalization=not args.no_normalize,
        max_text_tokens_per_segment=args.segment_tokens,
        num_beams=num_beams,
    )
    dt = time.perf_counter() - t0
    dur = audio.numel() / sr

    import soundfile as sf
    sf.write(args.output, audio.float().cpu().numpy().squeeze(), sr)
    print(f"{args.output}  ({dur:.2f}s audio in {dt * 1000:.0f}ms, RTF={dt / dur:.3f}, {sr}Hz)")
    if args.verbose:
        print(f"[peak VRAM] {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(dispatch_main())
