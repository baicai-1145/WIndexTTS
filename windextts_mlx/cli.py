"""WIndexTTS-MLX CLI — synthesize speech on Apple Silicon (pure MLX).

Parameter-compatible with the torch `windextts` CLI: the unified `windextts`
entry auto-forwards here on Apple Silicon. Torch-only options are accepted
but rejected with a clear message instead of an argparse error.
"""
from __future__ import annotations

import argparse
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="windextts",
        description="WIndexTTS: IndexTTS-2.5 inference on Apple Silicon (pure MLX/Metal backend).")
    p.add_argument("--ref", required=True, help="reference audio path (5-15s clean speech)")
    p.add_argument("--text", help="text to synthesize (or use --text-file)")
    p.add_argument("--text-file", help="read text from a file (UTF-8)")
    p.add_argument("-o", "--output", "--out", dest="output", default="output.wav",
                   help="output wav path")

    p.add_argument("--model-dir", "--weights", dest="model_dir", default=None,
                   help="converted MLX weights directory")
    p.add_argument("--lang", default="ZH",
                   help="language token (ZH/EN/JA/KO/YUE/..., default ZH)")
    p.add_argument("--duration", type=float, default=1.0,
                   help="duration factor (official 1.72 scale × factor, default 1.0)")

    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],
                   help="compute dtype (default fp16)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fp32", action="store_const", const="fp32", dest="dtype",
                      help="fp32 weights; alias of --dtype fp32")
    mode.add_argument("--quantize", "--w4a16", dest="quantize", action="store_true",
                      help="W4A16 INT4 quantization on GPT body + mel_head "
                           "(fastest); --w4a16 kept as alias")

    emo = p.add_mutually_exclusive_group()
    emo.add_argument("--emo-vector",
                     help="8-dim emotion weights happy,angry,sad,afraid,disgusted,"
                          "melancholic,surprised,calm — e.g. 0.8,0,0,0,0,0,0.2,0")
    emo.add_argument("--emo-text", help="free-text emotion description (loads QwenEmotion)")
    emo.add_argument("--emo-ref", help="emotion reference audio path (conformer path)")

    p.add_argument("--greedy", action="store_true",
                   help="greedy decode (beam=1, deterministic; overrides --beams)")
    p.add_argument("--beams", type=int, default=3,
                   help="GPT-AR beam count (default 3; 1 = greedy)")
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)

    p.add_argument("--cfm-steps", type=int, default=15,
                   help="S2Mel CFM Euler steps (default 15)")
    p.add_argument("--no-normalize", action="store_true",
                   help="skip text normalization (G2P digits/punctuation)")
    p.add_argument("--segment-tokens", type=int, default=120,
                   help="max text tokens per segment (long texts auto-split)")
    p.add_argument("--verbose", action="store_true", help="print timing details")

    # torch-only options: parsed for CLI parity, rejected at runtime
    p.add_argument("--low-vram", action="store_true",
                   help="(CUDA only) ignored on the MLX backend with a warning")
    p.add_argument("--install-model", action="store_true",
                   help="(CUDA only) download torch model weights; not supported here")
    p.add_argument("--skip-qwen", action="store_true",
                   help="(CUDA only) with --install-model; not supported here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.install_model or args.skip_qwen:
        print("error: --install-model/--skip-qwen are CUDA(torch)-backend options; "
              "MLX uses converted weights (see README 'MLX 适配' / tests/align/mlx).",
              file=sys.stderr)
        return 2
    if args.low_vram:
        print("warning: --low-vram has no effect on the MLX backend; ignoring.",
              file=sys.stderr)

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

    num_beams = 1 if args.greedy else max(1, args.beams)
    if args.dtype == "fp32" and args.quantize:
        print("error: --fp32/--dtype fp32 conflicts with --quantize/--w4a16", file=sys.stderr)
        return 2

    from windextts_mlx import WIndexTTSMLX
    from windextts_mlx.weights import DEFAULT_MLX_DIR

    t0 = time.perf_counter()
    tts = WIndexTTSMLX(weights_dir=args.model_dir or DEFAULT_MLX_DIR,
                       dtype=args.dtype, quantize=args.quantize)
    load_s = time.perf_counter() - t0

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
    dur = audio.shape[0] / sr

    import soundfile as sf
    sf.write(args.output, audio, sr)
    print(f"{args.output}  ({dur:.2f}s audio in {dt * 1000:.0f}ms, RTF={dt / dur:.3f}, {sr}Hz)")
    if args.verbose:
        print(f"[load] {load_s:.1f}s  [infer] {dt:.1f}s  beams={num_beams} dtype={args.dtype}"
              f"{' quantized' if args.quantize else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
