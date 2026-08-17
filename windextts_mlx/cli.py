# CLI: windextts-mlx — synthesize speech on Apple Silicon (pure MLX).
import argparse
import time


def main():
    p = argparse.ArgumentParser(description="WIndexTTS-MLX: IndexTTS-2.5 inference on Apple Silicon")
    p.add_argument("--text", required=True, help="text to synthesize")
    p.add_argument("--ref", required=True, help="reference audio (voice cloning prompt)")
    p.add_argument("--out", default="out.wav", help="output wav path")
    p.add_argument("--lang", default="ZH")
    p.add_argument("--weights", default="/Volumes/2T/IndexTTS-2.5-mlx", help="converted mlx weights dir")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"], help="compute dtype")
    p.add_argument("--quantize", action="store_true", help="W4A16 on GPT body + mel_head")
    p.add_argument("--beams", type=int, default=3, help="GPT-AR beam count (1 = greedy)")
    p.add_argument("--cfm-steps", type=int, default=15)
    p.add_argument("--emo-text", default=None, help="text-based emotion (loads QwenEmotion)")
    args = p.parse_args()

    t0 = time.time()
    tts = WIndexTTSMLX(weights_dir=args.weights, dtype=args.dtype, quantize=args.quantize)
    print(f">> load {time.time() - t0:.1f}s")
    t0 = time.time()
    sr, audio = tts.infer(args.ref, args.text, lang=args.lang, num_beams=args.beams,
                          cfm_steps=args.cfm_steps, emo_text=args.emo_text)
    print(f">> infer {time.time() - t0:.1f}s ({audio.shape[0] / sr:.2f}s audio, RTF {(time.time() - t0) / (audio.shape[0] / sr):.2f})")
    import soundfile as sf

    sf.write(args.out, audio, sr)
    print(f">> wrote {args.out}")


if __name__ == "__main__":
    from windextts_mlx import WIndexTTSMLX

    main()
