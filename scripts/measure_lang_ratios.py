#!/usr/bin/env python3
"""
Measure text_token → mel_code ratio for ALL 99 Whisper languages (50 sentences each).

Uses Tatoeba sentences (open, per-language TSV). Downloads ~50 standalone sentences
per language, tokenizes with the project tokenizer, runs GPT-AR beam3 to get mel codes,
and records the ratio. Outputs per-language statistics (mean/median/p95/max).

Usage:
    python scripts/measure_lang_ratios.py --out /root/windextts_dumps/lang_ratios.json
"""
from __future__ import annotations

import argparse
import bz2
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from windextts.inference import WIndexTTS, REF_MAX_SECONDS, REF_SR_W2V  # noqa: E402
from windextts.frontend.tokenizer import build_tokenizer, lang_to_token  # noqa: E402
import torchaudio  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Whisper 99 lang codes → Tatoeba 3-letter language code (ISO 639-3 / Tatoeba convention)
# Tatoeba uses ISO 639-3 for most; some special mappings.
WHISPER_TO_TATOEBA = {
    "en": "eng", "zh": "cmn", "de": "deu", "es": "spa", "ru": "rus",
    "ko": "kor", "fr": "fra", "ja": "jpn", "pt": "por", "tr": "tur",
    "pl": "pol", "ca": "cat", "nl": "nld", "ar": "ara", "sv": "swe",
    "it": "ita", "id": "ind", "hi": "hin", "fi": "fin", "vi": "vie",
    "he": "heb", "uk": "ukr", "el": "ell", "ms": "zsm", "cs": "ces",
    "ro": "ron", "da": "dan", "hu": "hun", "ta": "tam", "no": "nob",
    "th": "tha", "ur": "urd", "hr": "hrv", "bg": "bul", "lt": "lit",
    "la": "lat", "mi": "mri", "ml": "mal", "cy": "cym", "sk": "slk",
    "te": "tel", "fa": "pes", "lv": "lvs", "bn": "ben", "sr": "srp",
    "az": "aze", "sl": "slv", "kn": "kan", "et": "est", "mk": "mkd",
    "br": "bre", "eu": "eus", "is": "isl", "hy": "hye", "ne": "npi",
    "mn": "mon", "bs": "bos", "kk": "kaz", "sq": "sqi", "sw": "swh",
    "gl": "glg", "mr": "mar", "pa": "pan", "si": "sin", "km": "khm",
    "sn": "sna", "yo": "yor", "so": "som", "af": "afr", "oc": "oci",
    "ka": "kat", "be": "bel", "tg": "tgk", "sd": "snd", "gu": "guj",
    "am": "amh", "yi": "yid", "lo": "lao", "uz": "uzb", "fo": "fao",
    "ht": "hat", "ps": "pus", "tk": "tuk", "nn": "nno", "mt": "mlt",
    "sa": "san", "lb": "ltz", "my": "mya", "bo": "bod", "tl": "tgl",
    "mg": "mlg", "as": "asm", "tt": "tat", "haw": "haw", "ln": "lin",
    "ha": "hau", "ba": "bak", "jw": "jav", "su": "sun",
}

SENTENCES_PER_LANG = 50
MIN_LEN_CHARS = 8      # skip trivially short
MAX_LEN_CHARS = 200    # skip very long (avoid pathological)


def download_tatoeba(tat_code: str, retries: int = 6) -> list[str]:
    """Download Tatoeba sentences for one language. Returns list of sentence strings."""
    url = f"https://downloads.tatoeba.org/exports/per_language/{tat_code}/{tat_code}_sentences.tsv.bz2"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            # decompress
            text = bz2.decompress(raw).decode("utf-8", errors="replace")
            # Tatoeba TSV can have very long fields; raise csv limit (default 131072)
            csv.field_size_limit(10 * 1024 * 1024)  # 10MB
            reader = csv.reader(io.StringIO(text), delimiter="\t")
            sentences = []
            for row in reader:
                # row: [id, lang, text]
                if len(row) >= 3:
                    s = row[2].strip()
                    if MIN_LEN_CHARS <= len(s) <= MAX_LEN_CHARS:
                        sentences.append(s)
            return sentences
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))  # longer backoff for DNS flakiness
    print(f"    !! {tat_code} download failed after {retries} retries: {last_err}")
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/windextts_dumps/lang_ratios.json")
    ap.add_argument("--weights", default="/root/IndexTTS-2.5")
    ap.add_argument("--ref", default="/root/WIndexTTS/test.wav")
    ap.add_argument("--per-lang", type=int, default=SENTENCES_PER_LANG)
    ap.add_argument("--only", default="", help="comma-separated whisper codes to run (for resume/debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    only = set(args.only.split(",")) if args.only else None

    print(">> Loading WIndexTTS model...")
    tts = WIndexTTS(weights_dir=args.weights, device="cuda", dtype=torch.float16)
    tts.warmup()
    enc = build_tokenizer()

    # ref audio features (fixed across all languages)
    audio, sr = tts._load_audio(args.ref, REF_MAX_SECONDS)
    a16 = torchaudio.transforms.Resample(sr, REF_SR_W2V)(audio)
    spk_cond = tts.extract_spk_cond(a16)
    style = tts.extract_style(a16)

    results: dict[str, dict] = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # incremental save: load existing
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f).get("languages", {})

    n_langs = len(WHISPER_TO_TATOEBA)
    for idx, (whisp, tat) in enumerate(sorted(WHISPER_TO_TATOEBA.items()), 1):
        if only and whisp not in only:
            continue
        # skip only if we already have enough *successful* measurements
        prev_n = results.get(whisp, {}).get("n_measured", 0)
        if whisp in results and prev_n >= args.per_lang:
            print(f"[{idx}/{n_langs}] {whisp} ({tat}): already done ({prev_n} measured), skip")
            continue

        print(f"[{idx}/{n_langs}] {whisp} ({tat}): downloading...", flush=True)
        sentences = download_tatoeba(tat)
        if len(sentences) < args.per_lang:
            print(f"    only {len(sentences)} sentences available (need {args.per_lang}), using all")
        if not sentences:
            results[whisp] = {"tatoeba": tat, "n_available": 0, "samples": [], "error": "no sentences"}
            with open(out_path, "w") as f:
                json.dump({"languages": results}, f, ensure_ascii=False, indent=2)
            continue
        random.shuffle(sentences)
        picked = sentences[: args.per_lang]
        samples = []
        lang_t = torch.LongTensor([lang_to_token(whisp.upper())]).cuda()

        for si, text in enumerate(picked):
            lp = f"<|{whisp.lower()}|> "
            try:
                toks = enc.encode(lp + text, allowed_special="all")
                # defensive: skip if any token id exceeds vocab (avoid CUDA assert)
                # n_vocab=60509 (number_text_tokens); text_embedding covers all of it
                max_tok = max(toks) if toks else 0
                if max_tok >= 60510:  # tokenizer n_vocab safeguard (60509)
                    samples.append({"text": text[:80], "error": f"token id {max_tok} out of vocab"})
                    continue
                tt = F.pad(torch.IntTensor(toks).unsqueeze(0).cuda(), (0, 1), value=1)
                conds = tts.gpt.build_conds_latent(style, tts.build_emo_vec(style, spk_cond))
                torch.manual_seed(args.seed + si)
                codes = tts.gpt.generate(
                    conds, tt, lang_t, max_new_tokens=512, do_sample=False,
                    stop_token=tts.cfg.gpt.stop_mel_token, use_cuda_graph=False,
                )  # greedy: ratio (text→mel length) is beam/sample-invariant;
                  # use greedy+no-beam to minimize KV memory and avoid OOM across 4950 samples
                n_text = len(toks)
                n_mel = int(codes.shape[1])
                ratio = n_mel / n_text if n_text > 0 else 0
                samples.append({"text": text[:80], "n_text": n_text, "n_mel": n_mel, "ratio": round(ratio, 3)})
                del codes, tt, conds
                torch.cuda.empty_cache()  # defrag between samples
            except Exception as e:
                samples.append({"text": text[:80], "error": str(e)[:100]})
                # clear CUDA error state if possible
                torch.cuda.synchronize()

        ratios = [s["ratio"] for s in samples if "ratio" in s]
        if ratios:
            arr = np.array(ratios)
            stats = {
                "tatoeba": tat,
                "n_available": len(sentences),
                "n_measured": len(ratios),
                "mean": round(float(arr.mean()), 2),
                "median": round(float(np.median(arr)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "max": round(float(arr.max()), 2),
                "min": round(float(arr.min()), 2),
                "std": round(float(arr.std()), 2),
                "samples": samples,
            }
        else:
            stats = {"tatoeba": tat, "n_available": len(sentences), "samples": samples, "error": "all failed"}
        results[whisp] = stats
        print(f"    mean={stats.get('mean','?')} median={stats.get('median','?')} p95={stats.get('p95','?')} max={stats.get('max','?')} (n={stats.get('n_measured',0)})")

        # incremental save (resume-friendly)
        with open(out_path, "w") as f:
            json.dump({"languages": results}, f, ensure_ascii=False, indent=2)

    print(f"\n>> Saved to {out_path}")
    print(">> Summary:")
    for whisp, s in sorted(results.items()):
        if "mean" in s:
            print(f"  {whisp:>4} ({s['tatoeba']:>4}): mean={s['mean']:>5} p95={s['p95']:>5} max={s['max']:>5} n={s['n_measured']}")


if __name__ == "__main__":
    main()
