
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "IndexTeam/IndexTTS-2.5"
BASE_URL = f"https://modelscope.cn/models/{REPO}/resolve/master"

# (repo-relative path, required). Order: small config files first so failures
# happen fast, then the big weights.
FILES: list[tuple[str, bool]] = [
    ("config.yaml", True),
    ("multilingual_zh_ja_yue_char_del.tiktoken", True),
    ("wav2vec2bert_stats.pt", True),
    ("feat1.pt", True),
    ("feat2.pt", True),
    ("hf_cache/w2v-bert-2.0/config.json", True),
    ("hf_cache/w2v-bert-2.0/preprocessor_config.json", True),
    ("hf_cache/bigvgan/config.json", True),
    ("hf_cache/campplus_cn_common.bin", True),
    ("hf_cache/bigvgan/bigvgan_generator.pt", True),
    ("hf_cache/w2v-bert-2.0/model.safetensors", True),
    ("codec.pth", True),
    ("s2mel.pth", True),
    ("gpt.pth", True),
    # optional: emo_text path only (1.2GB)
    ("qwen0.6bemo4-merge/config.json", False),
    ("qwen0.6bemo4-merge/generation_config.json", False),
    ("qwen0.6bemo4-merge/model.safetensors", False),
    ("qwen0.6bemo4-merge/tokenizer.json", False),
    ("qwen0.6bemo4-merge/tokenizer_config.json", False),
    ("qwen0.6bemo4-merge/vocab.json", False),
    ("qwen0.6bemo4-merge/merges.txt", False),
    ("qwen0.6bemo4-merge/special_tokens_map.json", False),
    ("qwen0.6bemo4-merge/added_tokens.json", False),
    ("qwen0.6bemo4-merge/chat_template.jinja", False),
]

CORE_FILES = [p for p, req in FILES if req]
QWEN_MARKER = "qwen0.6bemo4-merge/model.safetensors"

# progress callback registry (webui hooks in here)
progress_hooks: list = []


def _notify(msg: str) -> None:
    print(msg, flush=True)
    for h in progress_hooks:
        try:
            h(msg)
        except Exception:
            pass


def check_install(model_dir: str | Path) -> tuple[bool, list[str], list[str]]:
    d = Path(model_dir)
    missing = [f for f in CORE_FILES if not (d / f).exists()]
    missing_opt = [] if (d / QWEN_MARKER).exists() else ["qwen0.6bemo4-merge/"]
    return (not missing, missing, missing_opt)


def _download_one(rel: str, out_dir: Path, retries: int = 3) -> None:
    url = f"{BASE_URL}/{rel}"
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            pos = dest.stat().st_size if dest.exists() else 0
            req = urllib.request.Request(url)
            if pos:
                req.add_header("Range", f"bytes={pos}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0)) + pos
                if r.status == 200 and pos:
                    # server ignored Range — restart
                    pos = 0
                    total = int(r.headers.get("Content-Length", 0))
                mode = "ab" if pos else "wb"
                done = pos
                t0 = time.time()
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)  # 1MB
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total and (done % (64 << 20) < (1 << 20)):  # every ~64MB
                            speed = (done - pos) / max(time.time() - t0, 1e-6) / 1e6
                            _notify(f"  {rel}: {done / 1e9:.2f}/{total / 1e9:.2f}GB "
                                    f"({speed:.1f}MB/s)")
            if total and done != total:
                raise IOError(f"size mismatch: {done} != {total}")
            gb = done / 1e9
            _notify(f"  ✓ {rel}" + (f" ({gb:.2f}GB)" if gb > 0.1 else ""))
            return
        except Exception as e:  # noqa: BLE001 — retry any network error
            if attempt == retries:
                raise
            _notify(f"  retry {attempt}/{retries} {rel}: {e}")
            time.sleep(2 * attempt)


def download_model(
    out_dir: str | Path,
    include_qwen: bool = True,
    only_missing: bool = True,
) -> Path:
    out_dir = Path(out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for rel, req in FILES:
        if not req and not include_qwen:
            continue
        if only_missing and (out_dir / rel).exists() and (out_dir / rel).stat().st_size > 0:
            continue
        todo.append(rel)
    total_n = len(todo)
    if not todo:
        _notify(f">> All files present: {out_dir}")
        return out_dir
    _notify(f">> Downloading {total_n} file(s) -> {out_dir}")
    for i, rel in enumerate(todo, 1):
        _notify(f"[{i}/{total_n}]")
        _download_one(rel, out_dir)

    ok, missing, _ = check_install(out_dir)
    if missing:
        _notify(f"!! core files still missing: {missing}")
    else:
        _notify(f">> Model ready: {out_dir}")
        _notify(f"   export WINDEXTTS_WEIGHTS_DIR={out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="windextts-download",
        description="Download IndexTTS-2.5 weights (~7.5GB) from ModelScope "
                    "direct links (no SDK needed; resumable).",
    )
    p.add_argument("--out", default=None,
                   help="target directory (default: $WINDEXTTS_WEIGHTS_DIR)")
    p.add_argument("--skip-qwen", action="store_true",
                   help="skip qwen0.6bemo4-merge (1.2GB; only needed for emo_text)")
    p.add_argument("--force", action="store_true",
                   help="re-download even if files exist")
    p.add_argument("--check", action="store_true",
                   help="only verify an existing directory, no download")
    args = p.parse_args(argv)

    out = args.out or os.environ.get("WINDEXTTS_WEIGHTS_DIR")
    if not out:
        print("error: --out or WINDEXTTS_WEIGHTS_DIR is required", file=sys.stderr)
        return 2

    if args.check:
        ok, missing, missing_opt = check_install(out)
        print(f"model dir: {out}")
        print(f"  core files:    {'OK' if not missing else 'MISSING ' + str(missing)}")
        print(f"  optional:      {'OK' if not missing_opt else 'missing (emo_text disabled): ' + str(missing_opt)}")
        return 0 if ok else 1

    download_model(out, include_qwen=not args.skip_qwen, only_missing=not args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
