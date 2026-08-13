"""Dump official emo_ref_audio path tensors for WIndexTTS alignment.

Uses the high-level merge_emovec() which internally handles all the
transpose/length bookkeeping. We hook sub-modules to capture intermediates.
"""
import sys, os
import torch

sys.path.insert(0, "/root/index-tts")
OUT = "/root/windextts_dumps/emo_ref"
os.makedirs(OUT, exist_ok=True)

from indextts.infer_v2_5 import IndexTTS2

tts = IndexTTS2(
    model_dir="/root/IndexTTS-2.5",
    cfg_path="/root/IndexTTS-2.5/config.yaml",
    use_bf16=False,
)

emo_path = "/root/WIndexTTS/test.wav"

# Hook sub-modules to capture their exact inputs/outputs
captured = {}
def make_hook(name):
    def hook(mod, inp, out):
        captured[name] = {"in": inp, "out": out}
    return hook

hooks = [
    tts.gpt.emo_conditioning_encoder.register_forward_hook(make_hook("conformer")),
    tts.gpt.emo_perceiver_encoder.register_forward_hook(make_hook("perceiver")),
    tts.gpt.emovec_layer.register_forward_hook(make_hook("emovec_layer")),
    tts.gpt.emo_layer.register_forward_hook(make_hook("emo_layer")),
]

# 1. extract emo_cond_emb (w2v-bert[17])
audio_16k, _ = tts._load_and_cut_audio(emo_path, 15, True, sr=16000)
emo_inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
emo_input_features = emo_inputs["input_features"].to(tts.device)
emo_attention_mask = emo_inputs["attention_mask"].to(tts.device)
emo_cond_emb = tts.get_emb(emo_input_features, emo_attention_mask)
torch.save({"emo_cond_emb": emo_cond_emb.cpu()}, f"{OUT}/emo_cond_emb.pt")
print(f"emo_cond_emb: {tuple(emo_cond_emb.shape)}")

# 2. Run merge_emovec (high-level, correct bookkeeping). Same audio for spk+emo.
with torch.no_grad():
    spk_cond_emb = emo_cond_emb  # same audio
    spk_T = spk_cond_emb.shape[1]
    emo_T = emo_cond_emb.shape[1]
    merged = tts.gpt.merge_emovec(
        spk_cond_emb, emo_cond_emb,
        torch.tensor([spk_T], device=tts.device),
        torch.tensor([emo_T], device=tts.device),
        alpha=0.65,
    )
torch.save({"merged_emovec_a065": merged.cpu()}, f"{OUT}/merged_emovec.pt")
print(f"merged_emovec (a=0.65): {tuple(merged.shape)} range [{merged.min():.3f},{merged.max():.3f}]")

# alpha=1.0 (pure emo ref)
with torch.no_grad():
    merged1 = tts.gpt.merge_emovec(
        spk_cond_emb, emo_cond_emb,
        torch.tensor([spk_T], device=tts.device),
        torch.tensor([emo_T], device=tts.device),
        alpha=1.0,
    )
torch.save({"merged_emovec_a100": merged1.cpu()}, f"{OUT}/merged_emovec_a100.pt")
print(f"merged_emovec (a=1.0): {tuple(merged1.shape)}")

# 3. Save captured intermediates
# conformer: out = (seq [B,T',512], mask [B,1,T'])
conf = captured["conformer"]["out"]
torch.save({"conformer_seq": conf[0].cpu(), "conformer_mask": conf[1].cpu()}, f"{OUT}/conformer_seq.pt")
print(f"conformer_seq: {tuple(conf[0].shape)}")

# perceiver out: [B, num_latents, 1024]
perc = captured["perceiver"]["out"]
torch.save({"perceiver_out": perc.cpu()}, f"{OUT}/perceiver_out.pt")
print(f"perceiver_out: {tuple(perc.shape)}")

# emovec_layer in [B,1024] -> out [B,1280]
evl = captured["emovec_layer"]
torch.save({"emovec_layer_in": evl["in"][0].cpu(), "emovec_layer_out": evl["out"].cpu()}, f"{OUT}/emovec_layer.pt")
print(f"emovec_layer: {tuple(evl['in'][0].shape)} -> {tuple(evl['out'].shape)}")

# emo_layer in [B,1280] -> out [B,1280]
ell = captured["emo_layer"]
torch.save({"emo_layer_in": ell["in"][0].cpu(), "emo_layer_out": ell["out"].cpu()}, f"{OUT}/emo_layer.pt")
print(f"emo_layer: {tuple(ell['in'][0].shape)} -> {tuple(ell['out'].shape)}")

# base_vec (spk only, via get_emovec on spk_cond_emb)
with torch.no_grad():
    base_vec = tts.gpt.get_emovec(spk_cond_emb, torch.tensor([spk_T], device=tts.device))
torch.save({"base_vec": base_vec.cpu()}, f"{OUT}/base_vec.pt")
print(f"base_vec: {tuple(base_vec.shape)}")

# Also dump the conformer input (what get_emo_conditioning receives)
# From hook: conformer["in"][0] is the [B, 1024, T] transposed input
conf_in = captured["conformer"]["in"][0]
torch.save({"conformer_in": conf_in.cpu()}, f"{OUT}/conformer_in.pt")
print(f"conformer_in: {tuple(conf_in.shape)}")

for h in hooks:
    h.remove()

print(f"\n✓ dumped to {OUT}/")
