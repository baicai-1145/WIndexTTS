# IndexTTS-2.5 Inference Tensor Seams

Authoritative reference for re-implementing IndexTTS-2.5 inference in pure torch.

**All shapes are empirically verified** (probe scripts under `/root/WIndexTTS/scripts/`,
reference env `/root/index-tts/.venv`, A10G, ref audio `/root/WIndexTTS/test.wav` 97259
samples @16kHz = 6.08s). When this doc conflicts with any summary/prose, **this doc wins**
— it is grounded in actual model runs.

Source: `/root/index-tts/indextts/infer_v2_5.py` + base classes `infer_v2.py` / `infer.py`.

---

## Pipeline (6 stages)

```
text → ① frontend(G2P+BPE) → text_tokens
ref_audio(16k) → ② w2v-bert feat → ③ EnhancedCodec.quantize → mel_codes (GPT target)
ref_audio → CAMPPlus → spk_emb(192)   ─┐
                                        ├→ ④ GPT-AR → codes
text_tokens, spk_emb, emovec ──────────┘
codes → EnhancedCodec.decode → S_infer(1024-d) → ⑤ S2Mel-CFM(25 steps) → mel
mel → ⑥ BigVGAN → 22050Hz audio
```

---

## ① Frontend (text → text_tokens)

`infer_v2_5.py` infer path. Construction: `tokenizer = tiktoken` over
`multilingual_zh_ja_yue_char_del.tiktoken` (58836 base + special tokens → vocab 60509).

Pipeline:
1. Text normalization: `jieba` + `cn2an` + `wetext` (zh), CMU for en.
2. Prepend language tag: `f"<|{lang.lower()}|> "` (e.g. `<|zh|> `).
3. `tokenizer.encode(prefix + text, allowed_special='all')`.
4. Pad right with **1** (`stop_text_token`) → `[1, T_text+1]`.

Verified shapes (text = "欢迎大家来体验indextts2..."):
- `text_tokens`: `[1, 31]` `int32`, range `1..58839` (values include lang tag 58839,
  BPE pieces; trailing 1 = stop_text).
- `lang`: `[1]` int64, scalar from `lang_to_token('ZH')` (e.g. ZH→1).
- `lang_embedding.weight`: `[107, 1280]` — 107 langs.

GPT embedding tables (all `[vocab, model_dim=1280]`):
- `text_embedding` `[60510, 1280]` (60509 + 1 pad)
- `mel_embedding` `[8194, 1280]` (8192 codes + start 8192 + stop 8193)
- `mel_head` `[8194, 1280]` (output projection)
- `text_pos_embedding.emb` `[602, 1280]`, `mel_pos_embedding.emb` `[1818, 1280]`

---

## ② w2v-bert feature extraction (the transformers-removal crux)

`infer_v2_5.py:282-289` (`get_emb`), `:173-179` (load + stats).

### 2a. SeamlessM4TFeatureExtractor preprocessing — `input_features`

SeamlessM4TFeatureExtractor does **Kaldi-style log-mel + stride-2 frame stacking**:

```python
# from transformers SeamlessM4TFeatureExtractor._extract_fbank_features
waveform = waveform * (2**15)           # Kaldi 16-bit signed int scaling
features = spectrogram(                # torchaudio kaldi spectrogram
    frame_length=400, hop_length=160, fft_length=512,
    power=2.0, center=False, preemphasis=0.97,
    mel_filters=<80-bin kaldi filterbank @16k>, log_mel="log",
    mel_floor=1.192092955078125e-7,
)
# then stack every 2 consecutive frames -> [T, 160]
features = np.stack([features[i], features[i+1]], axis=-1).reshape(T, 160)
```

Verified: `input_features [1, 303, 160]` float32 from 97259 samples @16k.
- 97259 / hop 160 ≈ 608 raw frames → after stride-2 stack → 303 (with center=False
  edge effects). `attention_mask [1, 303]` all 1.

**Re-implementation plan**: replicate via `torchaudio.compliance.kaldi.fbank` with
matching params + manual stride-2 stacking. High-risk point — mel filterbank table
and preemphasis must match exactly or `hidden_states[17]` won't align.

### 2b. Wav2Vec2BertModel forward — `hidden_states[17]`

```python
vq = semantic_model(input_features, attention_mask, output_hidden_states=True)
feat = vq.hidden_states[17]            # (B, T, 1024)  — THE seam magic number
feat = (feat - mean) / sqrt(var)       # stats from wav2vec2bert_stats.pt
```

Verified: `hidden_states[17]` `[1, 303, 1024]` float32 → normalized `[1, 303, 1024]`.
`mean`, `var` each `[1024]`. There are 25 hidden_states (embedding + 24 layers); we
take index 17.

Model: 24-layer conformer, hidden 1024, 16 heads, conv_depthwise_kernel=31,
relative positional encoding (`relative_key`), macaron FFN (ffn1 + ffn2 + conv_module
per layer). Weights: `hf_cache/w2v-bert-2.0/model.safetensors`, 773 keys, HF naming.

---

## ③ EnhancedCodec (semantic codec)

`infer_v2_5.py:292-297` (`get_scode`), `:851` (decode), module `codec/models.py`.

Construction: `EnhancedCodec(**cfg.semantic_codec)` with
`codebook_size=8192, hidden_size=1024, codebook_dim=8`, `num_quantizers=1`,
`downsample_scale=2`.

### quantize(feat) → (semantic_code, feat)
- input: normalized w2v-bert feat `[1, 303, 1024]`
- `semantic_code` `[1, 152]` int64, range `0..8191` (downsampled 2x: 303→152).
- `feat` `[1, 152, 1024]` float32 (quantized/reconstructed embeddings).

### decode(codes) → latent
- input: `semantic_code [1, 152]`
- output: `[1, 304, 1024]` float32 (upsampled 2x: 152→304). This is `S_infer`,
  the content latent fed to S2Mel.

Weights: `codec.pth` → `ckpt['model']`, 243 keys (encoder/decoder/quantizer).

---

## ④ GPT-AR (UnifiedVoice) — the 79% time stage

`gpt/model_v2.py` `UnifiedVoice`, `spk_cond_mode="campplus"` (v2.5).

### Speaker/emotion conditioning → `conditional_latents`

This is the resolved ambiguity (verified, supersedes earlier "w2v-bert 1024" claim).
Verified against `gpt/model_v2.py:740-775` (`inference_speech`, campplus branch):

```python
speech_conditioning_latent = self.spk_emb_proj(campplus_embedding)   # [1,1280]
emo_vec = self.emovec_layer(<pooled w2v feat>) | self.emo_layer(<emo>)  # [1,1280]
# campplus branch (v2.5):
conds_latent = torch.cat((
    speech_conditioning_latent + emo_vec.unsqueeze(1),           # 1 token: spk+emo merged
    torch.zeros(B, 2, 1280)                                       # 2 zero-pad tokens
), dim=1)                                                          # -> [1, 3, 1280]
```

So the 3 conditioning tokens are: **[spk_proj(campplus)+emovec, 0, 0]**. The spk
and emovec are *summed* into a single token, then 2 zero tokens pad to 3.

- `spk_emb_proj.weight` `[1280, 192]`, `emovec_layer.weight` `[1280, 1024]`,
  `emo_layer.weight` `[1280, 1280]`.
- `merge_emovec` (used in the infer path, not inference_speech) pools the w2v-bert
  1024-d sequence → `[1, 1024]`. (Needs exact pooling logic — TODO when implementing.)
- `emo_vec`: 8-dim `[happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]`,
  each `[0,1.2]`, via emo_matrix (feat2.pt [73,1280]) lookup → emo_layer.

Verified: `conditional_latents [1, 3, 1280]` fed to `inference_speech`.

### GPT body = standard GPT-2 (290 keys under `gpt.`)

`gpt.h.{0..23}` 24 layers, each 12 keys: `attn.{c_attn[1280,3840],c_proj[1280,1280]}`
+ `ln_1/ln_2[1280]` + `mlp.{c_fc[1280,5120],c_proj[5120,1280]}`. Plus `gpt.ln_f[1280]`.
This is vanilla GPT-2 — reimplementation is straightforward; the complexity is the
**AR decode loop** (HF `generate` in official; we must hand-write it for CUDA Graph).

### inference_speech(speech_condition, text_inputs, langs, ...) → codes

Uses **HuggingFace `generate`** (`GPT2InferenceModel`, vendored). Internally:
- `prepare_gpt_inputs`: builds `input_ids [1, 37]`, `inputs_embeds [1, 36, 1280]`
  (3 cond + 31 text + start mel + ...), `attention_mask [1, 37]`.
- Autoregressive decode, KV cache, emits mel codes until `stop_mel_token=8193`.

Verified output: `codes [1, 60]` int64, range `7..7554` (last token 4884, not 8193
in this run — run terminates by max_tokens or stop). `speech_cond_latent [1,1,1280]`.

GPT body: 24 layers, model_dim 1280, 20 heads (head_dim 64), condition via
prefix conditioning tokens (not cross-attn). 290 keys under `gpt.` prefix.

---

## ⑤ S2Mel-CFM (DiT + length_regulator)

`infer_v2_5.py:640-868`, module `s2mel/modules/` (flow_matching.py + diffusion_transformer.py + length_regulator.py). Weights `s2mel.pth` → `ckpt['net']` = `{cfm, length_regulator, gpt_layer}`.

### Inputs (all verified on test.wav)
- `S_infer = EnhancedCodec.decode(codes)` → `[1, 304, 1024]`
- `ref_mel = mel_fn(audio_22k)` → `[1, 80, 523]` (the prompt mel, also BigVGAN-style 80-bin log-mel at 22kHz)
- `style = CAMPPlus(fbank_16k_cm)` → `[1, 192]`
- `spk_cond` = normalized w2v-bert feat → `[1, 303, 1024]` (for prompt_condition)

### length_regulator(x, ylens, n_quantizers=3, f0=None)[0]
Two calls (length_regulator.py):
- `prompt_condition = lr(spk_cond_w2v[1,303,1024], ylens=ref_mel.size(2)=523, n_quantizers=3)[0]` → `[1, 523, 512]`
- `cond = lr(S_infer[1,304,1024], ylens=target_lengths, n_quantizers=3)[0]` → `[1, 522, 512]`
- `target_lengths = int(S_infer.shape[1] * 1.72 * duration_factor)` = 522 (dur=1.0)
- 22 keys: content_in_proj(1024→512), embedding(2048×512 codebook), mask_token, model.0..12 (HifiGAN-style conv1d/LayerNorm blocks)

### CFM inference
```python
cat_condition = cat([prompt_condition[1,523,512], cond[1,522,512]], dim=1)  # [1, 1045, 512]
vc_target = cfm.inference(cat_condition, x_lens=[1045], prompt=ref_mel[1,80,523],
                          style[1,192], f0=None, n_timesteps=25, inference_cfg_rate=0.7)
vc_target = vc_target[:, :, ref_mel.size(-1):]   # strip prompt -> [1, 80, 522]
```
`cfm.inference(mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5)`
- estimator = DiT (256 keys): x_embedder(conv80→512), cond_embedder(1024→512), t_embedder/t_embedder2,
  transformer(172 keys, 13 blocks), wavenet(51 keys, 8 layers), final_layer(adaLN),
  res_projection, skip_linear, cond_x_merge_linear(864→512), content_mask_embedder, conv1/conv2.
- **CFM**: 25 Euler steps, `inference_cfg_rate=0.7` (classifier-free guidance: v = (1-cfg)*v_uncond + cfg*v_cond, or similar).
- Output: mel `[1, 80, T_mel]` at 22050 Hz (hop 256). Verified output `[1, 80, 522]`.

### gpt_layer (only if use_gpt_latent; default off)
`latent = gpt_layer(latent)` then `S_infer = S_infer + latent`. 3 Linear: 1280→256→128→1024.
Default config does NOT use this (use_gpt_latent path), so skip for v1.

---

## ⑥ BigVGAN vocoder

`infer_v2_5.py:229` (`from_pretrained`), module `BigVGAN/`.

- input: mel `[1, 80, T_mel]`
- output: audio `[1, 1, T_audio]`, `T_audio = T_mel * 256` (hop_length).
- final audio resampled to **22050 Hz** mono (infer_v2_5.py:514).

Weights: `hf_cache/bigvgan/bigvgan_generator.pt` → unwrap `{'generator':{'model':sd}}`
→ 783 keys, weight-normalized (`weight_g`/`weight_v`), `conv_pre` + upsampling blocks.

---

## Critical "magic numbers" (do not invent — all verified)

| constant | value | source |
|---|---|---|
| w2v-bert layer index | `hidden_states[17]` | infer_v2_5.py:288 |
| w2v-bert norm | `(x - mean)/sqrt(var)` | infer_v2_5.py:289, stats wav2vec2bert_stats.pt |
| SeamlessM4T fbank | k=400,h=160,fft=512,preemph=0.97,log,80bins,stride2 | SeamlessM4TFeatureExtractor |
| codec downsample | 2x (303→152, 152→304) | EnhancedCodec |
| GPT spk cond | CAMPPlus 192 → spk_emb_proj → 1280 | model_v2.py:353,740-754 |
| start/stop mel token | 8192 / 8193 | config.yaml |
| text stop token | 1 (right-pad) | frontend |
| lang tag | `<\|{lang}|> ` prefix, lang_to_token scalar | frontend |
| S2Mel duration scale | 1.72 × duration_factor | infer_v2_5.py:855 |
| S2Mel n_quantizers | 3 | infer_v2_5.py:655,859 |
| CFM steps / cfg | 25 / 0.7 | infer_v2_5.py:849-850 |
| output audio | 22050 Hz mono | infer_v2_5.py:514 |
| ref audio resample | 22k(mel) / 16k(w2v,campplus) | infer_v2_5.py:628-629 |
