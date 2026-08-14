"""Download IndexTTS-2.5 model weights from the official release.

Sources (both are the official IndexTeam release):
  - HuggingFace:  IndexTeam/IndexTTS-2.5   (via huggingface_hub, default)
  - ModelScope:   IndexTeam/IndexTTS-2.5   (via modelscope, China-friendly)

Total download is ~10GB (gpt.pth 3.1G + hf_cache 5.2G + qwen0.6bemo4 1.2G +
codec 580M + s2mel 396M + small files).

Usage (CLI):
    windextts-download --out /path/to/IndexTTS-2.5          # HuggingFace
    windextts-download --out /path/to/IndexTTS-2.5 --source modelscope
    WINDEXTTS_WEIGHTS_DIR=/path windextts-download          # env-var target

API:
    from windextts.download import download_model
    download_model("/path/to/IndexTTS-2.5", source="huggingface")
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "IndexTeam/IndexTTS-2.5"

# Files that must exist for WIndexTTS to run. qwen0.6bemo4-merge (1.2GB) is
# only needed for the emo_text path; everything else is required.
CORE_FILES = [
    "config.yaml",
    "gpt.pth",
    "codec.pth",
    "s2mel.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "hf_cache/campplus_cn_common.bin",
    "hf_cache/w2v-bert-2.0/model.safetensors",
    "hf_cache/w2v-bert-2.0/config.json",
    "hf_cache/bigvgan/config.json",
    "hf_cache/bigvgan/bigvgan_generator.pt",
]


def check_install(model_dir: str | Path) -> tuple[bool, list[str], list[str]]:
    """Return (complete, missing_core, missing_optional)."""
    d = Path(model_dir)
    missing = [f for f in CORE_FILES if not (d / f).exists()]
    missing_opt = [] if (d / "qwen0.6bemo4-merge").exists() else ["qwen0.6bemo4-merge/"]
    return (not missing, missing, missing_opt)


def download_model(
    out_dir: str | Path,
    source: str = "huggingface",
    include_qwen: bool = True,
) -> Path:
    """Download the full IndexTTS-2.5 release into out_dir.

    source: "huggingface" (huggingface_hub) or "modelscope".
    include_qwen: skip the 1.2GB qwen0.6bemo4-merge dir (only needed for
        emo_text; pass False to save bandwidth).
    Resume: both backends continue partial downloads by default.
    """
    out_dir = Path(out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise SystemExit(
                "huggingface_hub not installed: pip install -U 'huggingface_hub[cli]'"
            ) from e
        print(f">> Downloading {REPO_ID} from HuggingFace -> {out_dir}")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(out_dir),
            # skip qwen if not wanted: HF patterns are regex on repo-relative paths
            ignore_patterns=["qwen0.6bemo4-merge/*"] if not include_qwen else None,
            max_workers=4,
        )
    elif source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as e:
            raise SystemExit("modelscope not installed: pip install modelscope") from e
        print(f">> Downloading {REPO_ID} from ModelScope -> {out_dir}")
        snapshot_download(f"{REPO_ID}", local_dir=str(out_dir))
        # modelscope has no ignore patterns; remove qwen if not wanted
        if not include_qwen and (out_dir / "qwen0.6bemo4-merge").exists():
            import shutil
            shutil.rmtree(out_dir / "qwen0.6bemo4-merge")
    else:
        raise SystemExit(f"unknown source '{source}' (huggingface | modelscope)")

    ok, missing, _ = check_install(out_dir)
    if missing:
        print(f"!! download finished but core files still missing: {missing}", file=sys.stderr)
    else:
        print(f">> Model ready: {out_dir}")
        print(f"   export WINDEXTTS_WEIGHTS_DIR={out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="windextts-download",
        description="Download IndexTTS-2.5 weights (~10GB) for WIndexTTS.",
    )
    p.add_argument("--out", default=None,
                   help="target directory (default: $WINDEXTTS_WEIGHTS_DIR)")
    p.add_argument("--source", default="huggingface", choices=["huggingface", "modelscope"],
                   help="download backend (default huggingface; modelscope is faster in China)")
    p.add_argument("--skip-qwen", action="store_true",
                   help="skip qwen0.6bemo4-merge (1.2GB; only needed for emo_text)")
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

    download_model(out, source=args.source, include_qwen=not args.skip_qwen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
