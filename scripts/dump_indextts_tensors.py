"""Rebuild /root/windextts_dumps reference tensors from the OFFICIAL IndexTTS-2.5
env (/root/index-tts/.venv). Run:
  cd /root/index-tts && .venv/bin/python /root/WIndexTTS/scripts/dump_indextts_tensors.py

Two passes:
  A) manual staged pipeline on test.wav (codec / lr / s2mel DiT+CFM / bigvgan /
     mel_fn / campplus / w2v featurizer input)
  B) two greedy infer runs (plain, and emo_vector path) to capture GPT prefill
     logits + greedy codes + emo_vec_final / conds_latent assembly.
"""
import os, sys
import torch
import torchaudio

sys.path.insert(0, "/root/index-tts")
OUT = "/root/windextts_dumps"
TEST_WAV = "/root/WIndexTTS/test.wav"
os.makedirs(OUT, exist_ok=True)

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.tokenizer import get_tokenizer, lang_to_token

tts = IndexTTS2(
    model_dir="/root/IndexTTS-2.5",
    cfg_path="/root/IndexTTS-2.5/config.yaml",
    use_bf16=False,
    use_qwen_emo=False,
)
dev = tts.device

def sv(name, t):
    torch.save(t.detach().cpu() if torch.is_tensor(t) else t, f"{OUT}/{name}.pt")
    print(f"saved {name}: {tuple(t.shape) if torch.is_tensor(t) else type(t)}")

# ---------- frontend: tokenizer ----------
tok = get_tokenizer(multilingual=True, model_dir="/root/IndexTTS-2.5")
torch.save(torch.tensor(tok.encode("<|zh|> 欢迎大家来体验indextts2。", allowed_special="all"), dtype=torch.int64),
           f"{OUT}/frontend.tokens_zh.pt")
torch.save(torch.LongTensor([lang_to_token("zh")]), f"{OUT}/gpt.lang.pt")

# ---------- pass A: manual staged pipeline (mirrors the original dump recipe) ----------
print("\n===== PASS A: staged pipeline =====")
audio, sr = torchaudio.load(TEST_WAV)
audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
audio_22k, audio_16k = audio_22k.to(dev), audio_16k.to(dev)

# mel_fn + campplus + bigvgan I/O via hooks (single shot each)
cap = {}
def mk(name):
    def f(mod, inp, out):
        if name not in cap:
            cap[name] = {"in": [i.detach().cpu() if torch.is_tensor(i) else i for i in inp],
                         "out": (out[0].detach().cpu(), out[1].detach().cpu()) if isinstance(out, tuple) and len(out) == 2
                                else out.detach().cpu() if torch.is_tensor(out) else out}
    return f
hs = [tts.bigvgan.register_forward_hook(mk("bigvgan")),
      tts.campplus_model.register_forward_hook(mk("campplus"))]

with torch.no_grad():
    ref_mel = tts.mel_fn(audio_22k.float())                       # [1,80,T_ref]
    feat = torchaudio.compliance.kaldi.fbank(audio_16k.cpu(), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    style = tts.campplus_model(feat.unsqueeze(0).to(dev))         # [1,192]

    inputs = tts.extract_features(audio_16k.cpu(), sampling_rate=16000, return_tensors="pt")
    spk_cond_emb = tts.get_emb(inputs["input_features"].to(dev), inputs["attention_mask"].to(dev))
    codes, qfeat = tts.semantic_codec.quantize(spk_cond_emb)
    S_infer = tts.semantic_codec.decode(codes)

    prompt_condition = tts.s2mel.models["length_regulator"](
        spk_cond_emb, ylens=torch.LongTensor([ref_mel.size(2)]).to(dev), n_quantizers=3, f0=None)[0]
    target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72)]).to(dev)
    cond = tts.s2mel.models["length_regulator"](
        S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    # DiT single-step dump: emulate Euler step at t=0.5 (x zero beyond prompt)
    x = torch.zeros(1, 80, cat_condition.size(1), device=dev)
    prompt_x = torch.zeros_like(x)
    prompt_x[..., :ref_mel.size(-1)] = ref_mel[..., :ref_mel.size(-1)]
    t = torch.tensor([0.5], device=dev)
    dit_out = tts.s2mel.models["cfm"].estimator(
        x, prompt_x, torch.LongTensor([cat_condition.size(1)]).to(dev), t, style, cat_condition)

    # CFM end-to-end (seed=123 to match the original dump)
    torch.cuda.manual_seed(123)
    cfm_out = tts.s2mel.models["cfm"].inference(
        cat_condition, torch.LongTensor([cat_condition.size(1)]).to(dev),
        ref_mel, style, None, 25, inference_cfg_rate=0.7)

    # bigvgan input/output: use the generated mel (stripped) as vocoder input
    mel_for_voc = cfm_out[:, :, ref_mel.size(-1):]
    wav_out = tts.bigvgan(mel_for_voc.float())

sv("frontend.audio_22k", audio_22k.cpu())
sv("frontend.mel_fn_output", ref_mel.cpu())
sv("campplus.fbank_cm", feat.cpu())
sv("campplus.spk_emb_192", style.cpu())
sv("gpt.style", style.cpu())
sv("gpt.campplus_style", style.cpu())
sv("gpt.spk_cond_w2v", spk_cond_emb)
sv("w2v.cond_emb_normalized", spk_cond_emb)  # get_emb output IS the normalized h17
sv("codec.quantize_code", codes)
sv("codec.quantize_feat", qfeat)
sv("codec.decode_latent", S_infer)
sv("s2mel.S_infer", S_infer)
sv("s2mel.ref_mel", ref_mel)
sv("s2mel.style", style)
sv("s2mel.prompt_condition", prompt_condition)
sv("s2mel.cond", cond)
sv("s2mel.dit_input_x", x)
sv("s2mel.dit_input_prompt_x", prompt_x)
sv("s2mel.dit_input_cond", cat_condition)
sv("s2mel.dit_input_style", style)
sv("s2mel.dit_input_t", t)
sv("s2mel.dit_output", dit_out)
sv("s2mel.cfm_output_mel_seed123", cfm_out[:, :, ref_mel.size(-1):])
sv("bigvgan.input_mel", mel_for_voc)
sv("bigvgan.output_wav", wav_out)
for h_ in hs:
    h_.remove()

# ---------- pass B: greedy infer runs (GPT dumps) ----------
print("\n===== PASS B: greedy infer (GPT) =====")
TEXT, LANG = "大家好。", "zh"
greedy_toks = tok.encode(f"<|{LANG}|> {TEXT}", allowed_special="all") + [1]
torch.save(torch.tensor([greedy_toks], dtype=torch.int64), f"{OUT}/gpt.text_tokens_greedy.pt")

cap2 = {}
def mk2(name):
    def f(mod, inp, out):
        if name not in cap2:
            cap2[name] = {"in": [i.detach().cpu() if torch.is_tensor(i) else i for i in inp],
                          "out": out.logits.detach().cpu() if hasattr(out, "logits") else
                                 (out[0].detach().cpu() if isinstance(out, tuple) else out.detach().cpu() if torch.is_tensor(out) else out)}
    return f

gpt_cap = {}
orig_inference_speech = tts.gpt.inference_speech
def inference_speech_cap(speech_condition, text_inputs, langs=None, **kw):
    out, conds_latent = orig_inference_speech(speech_condition, text_inputs, langs, **kw)
    if "greedy_codes" not in gpt_cap:
        gpt_cap["greedy_codes"] = out.cpu() if torch.is_tensor(out) else out
        gpt_cap["conds_latent"] = kw_snoop.get("conds_latent", None)
    return out, conds_latent
# simpler: hook inference_model.forward for prefill logits

h_prefill = tts.gpt.inference_model.register_forward_hook(mk2("gpt_prefill"))
orig_merge = tts.gpt.merge_emovec
def merge_cap(spk, emo, cl, el, alpha=1.0):
    out = orig_merge(spk, emo, cl, el, alpha=alpha)
    if "emo_vec_final" not in cap2:
        cap2["emo_vec_final"] = out.detach().cpu()
    return out
tts.gpt.merge_emovec = merge_cap
# capture the EXACT prefill embeddings (store_mel_emb arg) for conds_latent
orig_store = tts.gpt.inference_model.store_mel_emb
def store_cap(mel_emb):
    if "prefill_embeds" not in cap2:
        cap2["prefill_embeds"] = mel_emb.detach().cpu()
    return orig_store(mel_emb)
tts.gpt.inference_model.store_mel_emb = store_cap
# capture the EXACT conds_latent + padded text the official prefill consumed
orig_pgi = tts.gpt.prepare_gpt_inputs
def pgi_cap(conditional_latents, text_inputs, langs=None):
    if "conds_latent" not in cap2:
        cap2["conds_latent"] = conditional_latents.detach().cpu()
        cap2["text_tokens_run"] = text_inputs.detach().cpu()
    return orig_pgi(conditional_latents, text_inputs, langs)
tts.gpt.prepare_gpt_inputs = pgi_cap

print(">> B1: plain greedy infer (prefill logits + emo_vec_final via merge_emovec)")
res = tts.infer(
    TEST_WAV, TEXT, f"{OUT}/_rebuild_b1.wav", LANG,
    do_sample=False, num_beams=1, repetition_penalty=1.0, max_mel_tokens=100,
    verbose=True,
)
tts.gpt.merge_emovec = orig_merge
h_prefill.remove()

# prefill logits: the first inference_model.forward is the prefill.
sv("gpt.prefill_logits", cap2["gpt_prefill"]["out"])
sv("gpt.emo_vec_final", cap2["emo_vec_final"])
# conds_latent: exact tensor prepare_gpt_inputs received (NOT sliced embeds —
# the embeds have a left-pad zero row in front)
sv("gpt.conds_latent", cap2["conds_latent"])
# style/campplus_style MUST come from B1's own cache (not pass A) so that
# build_conds_latent(style, emo_vec_final) reproduces the run's conds.
sv("gpt.style", tts.cache_s2mel_style)
sv("gpt.campplus_style", tts.cache_s2mel_style)
# 8-dim user emotion weights (input contract of the soft matrix-lookup test)
sv("gpt.emo_vec", torch.tensor([0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

# greedy codes [1,49]: strip stop token handling matches test (ends at 8193)
codes = gpt_cap.get("greedy_codes")
if codes is None:
    # fallback: re-run inference_speech directly to capture
    with torch.no_grad():
        out, scl = tts.gpt.inference_speech(
            spk_cond_emb, torch.tensor([greedy_toks], dtype=torch.int64, device=dev),
            torch.LongTensor([lang_to_token(LANG)]).to(dev),
            emo_speech_condition=spk_cond_emb,
            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=dev),
            emo_cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=dev),
            emo_vec=cap2["emo_vec_final"].to(dev),
            campplus_embedding=style,
            do_sample=False, num_beams=1, repetition_penalty=1.0,
            max_generate_length=100,
        )
    codes = out
sv("gpt.greedy_codes", codes)

# text_tokens_short for prefill test: the exact padded tokens the run consumed
sv("gpt.text_tokens_short", cap2["text_tokens_run"])

print("\n>> done. emo_ref/ needs scripts/dump_emo_ref_tensors.py separately.")
