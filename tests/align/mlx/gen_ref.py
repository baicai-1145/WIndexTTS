# Generate the torch-CPU fp32 reference cache for MLX alignment tests.
# Runs in its own process (torch ref + mlx weights do not fit 16GB RAM together).
#   .venv/bin/python tests/align/mlx/gen_ref.py
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root (windextts)
sys.path.insert(0, str(Path(__file__).parent))
from _ref import RefTTS, SRC, to_np

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)


def save(name, **kw):
    np.savez(CACHE / f"{name}.npz", **{k: to_np(v) for k, v in kw.items()})
    print(f"  {name}.npz: {', '.join(kw)}")


def main():
    r = RefTTS(SRC)
    print("== RefTTS loaded (torch CPU fp32)")

    # --- DSP frontends ---
    t = torch.randn(1, 22050, generator=torch.Generator().manual_seed(1))
    save("mel", audio=t, out=r.mel_fn(t))

    t = torch.randn(1, 16000 * 2, generator=torch.Generator().manual_seed(2))
    save("featurizer", audio=t, out=r.featurizer(t, return_mask=True)[0], mask=r.featurizer(t, return_mask=True)[1])

    t = torch.randn(1, 16000 * 3, generator=torch.Generator().manual_seed(3))
    import torchaudio

    f = torchaudio.compliance.kaldi.fbank(t, num_mel_bins=80, dither=0, sample_frequency=16000)
    save("fbank", audio=t, out=f)

    t = torch.randn(1, 48000 * 2, generator=torch.Generator().manual_seed(4))
    r16 = torchaudio.transforms.Resample(48000, 16000)(t)
    r22 = torchaudio.transforms.Resample(48000, 22050)(t)
    save("resample", audio=t, out16=r16, out22=r22)

    # --- neural modules ---
    t = torch.randn(1, 16000 * 3, generator=torch.Generator().manual_seed(5))
    f = torchaudio.compliance.kaldi.fbank(t, num_mel_bins=80, dither=0, sample_frequency=16000)
    save("campplus", fbank=f, out=r.campplus((f - f.mean(0, keepdim=True)).unsqueeze(0)))

    c = torch.randint(0, 8192, (1, 152), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        emb = r.codec.quantizer.vq2emb(c.unsqueeze(0))  # [1,152,1,8]
        dec = r.codec.decoder(emb)  # [B,D,T]
        if hasattr(r.codec, "up"):
            up = r.codec.up(torch.nn.functional.interpolate(dec.transpose(1, 2), scale_factor=2, mode="nearest")).transpose(1, 2)
        else:
            up = dec.transpose(1, 2)
        out = r.codec.decode(c)
    save("codec_decode", codes=c, emb=emb, dec=dec.transpose(1, 2), up=up, out=out)
    z = torch.randn(1, 303, 1024, generator=torch.Generator().manual_seed(7))
    q = r.codec.quantize(z)
    save("codec_quant", z=z, idx=q[0], out=q[1])

    x = torch.randn(1, 152, 1024, generator=torch.Generator().manual_seed(8))
    ylens = torch.tensor([200])
    r.length_regulator.eval()
    xp = r.length_regulator.content_in_proj(x)
    xi = torch.nn.functional.interpolate(xp.transpose(1, 2).contiguous(), size=200, mode="nearest")  # [1,512,200]
    xm = r.length_regulator.model(xi)  # [1,512,200]
    save("length_reg", x=x, xp=xp, xi=xi, xm=xm, out=r.length_regulator(x, ylens=ylens)[0])

    t = torch.randn(1, 16000 * 3, generator=torch.Generator().manual_seed(9))
    inp, am = r.featurizer(t, return_mask=True)
    h = r.w2v_bert(inp, am, return_layer=17)
    save("w2v_bert", feat=inp, mask=am, out=h)

    spk = torch.randn(1, 120, 1024, generator=torch.Generator().manual_seed(10))
    save("emo_cond", spk=spk, emovec=r.gpt.get_emovec(spk), merged=r.gpt.merge_emovec(spk, spk, alpha=0.5))

    g = torch.Generator()
    B, T = 2, 64
    x = torch.randn(B, 80, T, generator=g.manual_seed(11))
    p = torch.randn(B, 80, T, generator=g.manual_seed(12))
    c = torch.randn(B, T, 512, generator=g.manual_seed(13))
    s = torch.randn(B, 192, generator=g.manual_seed(14))
    tt = torch.randn(B, generator=g.manual_seed(15))
    lens = torch.tensor([T, T])
    r.dit.setup_caches(max_batch_size=2, max_seq_length=T)
    save("dit", x=x, p=p, c=c, s=s, t=tt, out=r.dit(x, p, lens, tt, s, c))

    spk = torch.randn(1, 30, 1024, generator=g.manual_seed(16))  # spk_cond [B,T,1024]
    s_in = torch.randn(1, 64, 1024, generator=g.manual_seed(17))  # codec latent [B,2T,1024]
    refmel = torch.randn(1, 80, 30, generator=g.manual_seed(18))
    style = torch.randn(1, 192, generator=g.manual_seed(19))
    mel = r.s2mel.inference(spk, s_in, refmel, style, n_timesteps=8, inference_cfg_rate=0.7)
    save("s2mel", spk=spk, s=s_in, refmel=refmel, style=style, out=mel)

    t = torch.randn(1, 80, 128, generator=g.manual_seed(20))
    save("bigvgan", mel=t, out=r.bigvgan(t))

    # --- GPT-AR ---
    style = torch.randn(1, 192, generator=g.manual_seed(21))
    emo = torch.randn(1, 1280, generator=g.manual_seed(22))
    conds = r.gpt.build_conds_latent(style, emo)
    tt = torch.tensor([[1, 5, 10, 20, 1]], dtype=torch.long)
    lang = torch.tensor([4], dtype=torch.long)  # ZH
    ids, emb, mask = r.gpt.prepare_gpt_inputs(conds, tt, lang)
    lg = r.gpt.prefill_logits_from_inputs(conds, tt, lang)
    save("gpt_prefill", conds=conds, tt=tt, lang=lang, ids=ids, emb=emb, mask=mask, logits=lg)

    for seed, beams, tag in ((23, 1, "greedy"), (24, 3, "beam3")):
        style = torch.randn(1, 192, generator=g.manual_seed(seed))
        emo = torch.randn(1, 1280, generator=g.manual_seed(seed + 1))
        conds = r.gpt.build_conds_latent(style, emo)
        codes = r.gpt.generate(conds, tt, lang, max_new_tokens=64, do_sample=beams > 1,
                               top_k=30, top_p=0.8, temperature=0.8, repetition_penalty=10.0,
                               num_beams=beams, stop_token=r.cfg.gpt.stop_mel_token, use_cuda_graph=False)
        save(f"gpt_{tag}", conds=conds, tt=tt, lang=lang, codes=codes)

    # --- end-to-end (test.wav) ---
    import torchaudio

    wav = "/Volumes/2T/WIndexTTS/test.wav"
    audio, sr = torchaudio.load(wav)
    a16 = torchaudio.transforms.Resample(sr, 16000)(audio)[:, : 16000 * 6]
    a22 = torchaudio.transforms.Resample(sr, 22050)(audio)[:, : 22050 * 6]
    spk_cond = r.extract_spk_cond(a16)
    style = r.extract_style(a16)
    refmel = r.mel_fn(a22)
    emo_vec = r.build_emo_vec(style, spk_cond)
    conds = r.gpt.build_conds_latent(style, emo_vec)

    from windextts.frontend.normalizer import TextNormalizer
    from windextts.frontend.tokenizer import build_tokenizer, lang_to_token

    text = TextNormalizer().normalize("你好，这是一个端到端对齐测试。")
    tok = build_tokenizer(model_dir=SRC)
    tt = torch.tensor(tok.encode(f"<|zh|> {text}", allowed_special="all") + [1], dtype=torch.long)[None]
    lang_id = torch.tensor([lang_to_token("ZH")], dtype=torch.long)
    codes = r.gpt.generate(conds, tt, lang_id, max_new_tokens=96, do_sample=False,
                           stop_token=r.cfg.gpt.stop_mel_token, use_cuda_graph=False)
    # per-step reference logits top-2 gap (tie evidence for the e2e code comparison;
    # used by test_e2e to exempt numerically-tied argmax flips instead of a code bug)
    with torch.no_grad():
        mdtype = next(r.gpt.parameters()).dtype
        S, am, kvs, cl = r.gpt._prefill(conds, tt, lang_id)
        gaps = []
        for step in range(96):
            lg = cl[0].float()
            gaps.append(float(torch.topk(lg, 2).values[0] - torch.topk(lg, 2).values[1]))
            nid = torch.argmax(lg).unsqueeze(0)
            if nid.item() == r.cfg.gpt.stop_mel_token:
                break
            am, kvs, cl = r.gpt._eager_step(nid, step, am, kvs, mdtype)
    gaps = np.array(gaps[: codes.shape[1]], dtype=np.float32)
    s_in = r.codec.decode(codes)
    # fixed-z CFM so the reference mel is reproducible across processes: torch
    # S2Mel.inference draws z internally, so drive solve_euler directly with a
    # seeded z (mlx test side injects the SAME z -> identical mel input).
    with torch.no_grad():
        _, _, cat = r.s2mel.length_regulate(spk_cond, s_in, refmel, 1.0)
        x_lens = torch.LongTensor([cat.shape[1]])
        zz = torch.randn(1, r.s2mel.cfm.in_channels, cat.shape[1], generator=torch.Generator().manual_seed(7))
        t_span = torch.linspace(0, 1, 9, dtype=cat.dtype)
        r.dit.setup_caches(2, cat.shape[1])
        vc = r.s2mel.cfm.solve_euler(zz, x_lens, refmel, cat, style, None, t_span, 0.7)
    mel = vc[:, :, refmel.shape[-1]:]
    aud = r.bigvgan(mel).squeeze(0).squeeze(0).clamp(-1, 1)
    save("e2e", spk_cond=spk_cond, style=style, refmel=refmel, emo_vec=emo_vec, conds=conds,
         tt=tt, lang=lang_id, codes=codes, codes_gap=gaps, s=s_in, z=zz, mel=mel, audio=aud)
    print("== cache written to", CACHE)


if __name__ == "__main__":
    main()
