# finish conversion: tables + runtime text files (idempotent)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from windextts.weights import WeightLoader

src = Path("/Volumes/2T/IndexTTS-2.5")
out = Path("/Volumes/2T/IndexTTS-2.5-mlx")
w = WeightLoader(src)

np.savez(out / "feat.npz", spk=w.load_spk_matrix().detach().numpy(), emo=w.load_emo_matrix().detach().numpy())
mean, var = w.load_w2v_stats()
np.savez(out / "stats.npz", mean=mean.detach().numpy(), var=var.detach().numpy())
print("feat/stats ok")

import shutil

shutil.copy(src / "multilingual_zh_ja_yue_char_del.tiktoken", out / "multilingual_zh_ja_yue_char_del.tiktoken")
shutil.copy(src / "qwen0.6bemo4-merge" / "tokenizer.json", out / "qwen_tokenizer.json")
shutil.copy(src / "hf_cache" / "bigvgan" / "config.json", out / "bigvgan_config.json")
print("copies ok:", sorted(p.name for p in out.iterdir()))
