# Generate only the e2e reference (torch CPU) — separate process, unbuffered.
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tests/align/mlx")
from _ref import RefTTS, SRC, to_np

CACHE = Path("tests/align/mlx/cache")


def main():
    r = RefTTS(SRC)
    print("[e2e] RefTTS loaded", flush=True)
    import torchaudio

    audio, sr = torchaudio.load("/Volumes/2T/WIndexTTS/test.wav")
    a16 = torchaudio.transforms.Resample(sr, 16000)(audio)[:, : 16000 * 6]
    a22 = torchaudio.transforms.Resample(sr, 22050)(audio)[:, : 22050 * 6]
    spk_cond = r.extract_spk_cond(a16)
    style = r.extract_style(a16)
    refmel = r.mel_fn(a22)
    emo_vec = r.build_emo_vec(style, spk_cond)
    conds = r.gpt.build_conds_latent(style, emo_vec)
    print("[e2e] features ok", spk_cond.shape, flush=True)

    from windextts.frontend.normalizer import TextNormalizer
    from windextts.frontend.tokenizer import build_tokenizer, lang_to_token

    text = TextNormalizer().normalize("你好，这是一个端到端对齐测试。")
    tok = build_tokenizer(model_dir=SRC)
    tt = torch.tensor(tok.encode(f"<|zh|> {text}", allowed_special="all") + [1], dtype=torch.long)[None]
    lang_id = torch.tensor([lang_to_token("ZH")], dtype=torch.long)
    codes = r.gpt.generate(conds, tt, lang_id, max_new_tokens=96, do_sample=False,
                           stop_token=r.cfg.gpt.stop_mel_token, use_cuda_graph=False)
    print("[e2e] codes ok", codes.shape, flush=True)
    s_in = r.codec.decode(codes)
    mel = r.s2mel.inference(spk_cond, s_in, refmel, style, n_timesteps=8, inference_cfg_rate=0.7)
    aud = r.bigvgan(mel).squeeze(0).squeeze(0).clamp(-1, 1)
    print("[e2e] audio ok", aud.shape, flush=True)
    np.savez(CACHE / "e2e.npz", spk_cond=to_np(spk_cond), style=to_np(style), refmel=to_np(refmel),
             emo_vec=to_np(emo_vec), conds=to_np(conds), tt=to_np(tt), lang=to_np(lang_id),
             codes=to_np(codes), s=to_np(s_in), mel=to_np(mel), audio=to_np(aud))
    print("[e2e] saved e2e.npz", flush=True)


if __name__ == "__main__":
    main()
