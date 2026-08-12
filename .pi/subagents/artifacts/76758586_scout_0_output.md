# IndexTTS-2.5 官方推理接缝清单（v2.5 主路径 = `infer_v2_5.py`）

> 所有 shape 均经 GPU 实测验证（`/root/index-tts/.venv`，A10G）。
> 关键结论速览：
> - 6 阶段：前端 → w2v-bert(hidden_states[17]) → CAMPPlus → GPT-AR → EnhancedCodec → S2Mel-CFM → BigVGAN。
> - `semantic_codec`(hf_cache, MaskGCT) 与 `codec.pth`(EnhancedCodec) 是**两个不同模型**，v2.5 推理只用 `codec.pth`。
> - w2v-bert 加载走 transformers `Wav2Vec2BertModel.from_pretrained(local_dir)`，用目录里的 `model.safetensors`；`conformer_shaw.pt` 是 fairseq 原版遗留文件，推理时不读。
> - v2.5 的 speaker 条件 = 归一化后的 `hidden_states[17]`（1024 维原始嵌入，**不量化**），与 v2（用 `S_ref` 量化码）不同。

---

## 0. 全局配置与权重文件（`/root/IndexTTS-2.5/`）

| 权重 | 大小 | 加载方式 | 对应模块 |
|---|---|---|---|
| `gpt.pth` | 3.26GB | `load_checkpoint`(utils/checkpoint.py) → `UnifiedVoice(**cfg.gpt, spk_cond_mode="campplus")` | GPT-AR |
| `codec.pth` | 607MB | `EnhancedCodec.load_checkpoint`（读 `ckpt['model']`） | EnhancedCodec（语义 codec） |
| `s2mel.pth` | 415MB | `load_checkpoint2`（读 `ckpt['net']`，按 `cfm`/`length_regulator`/`gpt_layer` 分键） | CFM+DiT+length_regulator |
| `wav2vec2bert_stats.pt` | 9KB | `torch.load` → `{'mean','var'}` 各 [1024] float32 | w2v-bert 归一化 |
| `feat1.pt`/`feat2.pt` | | spk/emo 参考矩阵（emo 向量最近邻检索用） | 情绪控制（非必需路径） |
| `hf_cache/w2v-bert-2.0/` | 2.32GB | `SeamlessM4TFeatureExtractor` + `Wav2Vec2BertModel` `from_pretrained(local_files_only=True)` | w2v-bert |
| `hf_cache/campplus_cn_common.bin` | | `CAMPPlus(feat_dim=80, embedding_size=192)` + `load_state_dict` | CAMPPlus |
| `hf_cache/bigvgan/`（config.json + bigvgan_generator.pt） | 449MB | `indextts.s2mel.modules.bigvgan.bigvgan.BigVGAN.from_pretrained(dir)` + `remove_weight_norm()` | BigVGAN |
| `hf_cache/semantic_codec/model.safetensors` | 177MB | 仅 v2 路径 / 数据预处理使用（见 Q1） | MaskGCT RepCodec（**v2.5 不用**） |
| `multilingual_zh_ja_yue_char_del.tiktoken` | | tiktoken BPE 词表（60509 基础 token） | 前端 tokenizer |

`config.yaml` 关键超参（`/root/IndexTTS-2.5/config.yaml`）：
- `gpt`: model_dim=1280, heads=20, layers=24, max_mel_tokens=1815, max_text_tokens=600, number_text_tokens=60509, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193, start_text_token=0, stop_text_token=1, condition_type=conformer_perceiver, emo_condition_module(output 512, 4 blocks)
- `semantic_codec`: codebook_size=8192, hidden_size=1024, codebook_dim=8, vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12（无 num_quantizers/downsample_scale → 用类默认 1 / 2）
- `s2mel`: preprocess sr=22050, n_fft=1024, win=1024, hop=256, n_mels=80; length_regulator channels=512, is_discrete=false, in_channels=1024, sampling_ratios=[1,1,1,1], n_codebooks=1; DiT hidden_dim=512, heads=8, depth=13, in_channels=80, content_dim=512, final_layer_type=wavenet, long_skip_connection=true, uvit_skip_connection=true, style_condition=true, class_dropout_prob=0.1, zero_prompt_speech_token=false; wavenet hidden_dim=512, num_layers=8, kernel_size=5

---

## 阶段 1：前端 — 文本 → token

入口：`IndexTTS2.infer_generator`（`/root/index-tts/indextts/infer_v2_5.py:570`），文本处理在 **L699–727**。

```
text (str, 任意语言) 
 └─ text_process.clean_pattern.sub(char_rep_map)      L700  （标点映射表 front.py:18-43）
 └─ 若 text_normalization: 
     · lang∈{zh,zhen,en} → text_process.normalize()   L703-704 （weText/tn，front.py:173-210；Linux 用 tn.chinese.normalizer + tn.english.normalizer）
     · lang∈{ja,es}      → nemo_text_normalize()      L705  （nemo_tn）
 └─ lang∈{ja,zh,zhen,en} → text.lower() / es → upper  L708-710
 └─ apply_pronunciation_annotations(text)             L711  （`<文字|发音>` → `<|SPECIAL_TOKEN_1|>`/`<|SPECIAL_TOKEN_2|>` 包裹，发音大写；is_kana 则不加包裹，L35-54）
 └─ lang==ja → ja_text_process.process_ja_text(text)  L713  （fugashi 分词；v2.5 g2p_ratio=0 → 只分段不替换读音）
 └─ re.sub 特殊 token 内字母转大写                    L714
 └─ split_text_by_tokens(text, 120, lang_prefix)      L715  （按 `_token_len` token 预算切段；容量=text_pos_embedding.emb.num_embeddings(602)-2）
 └─ 每段: toks = tokenizer.encode(f'<|{lang.lower()}|> ' + seg, allowed_special='all')   L725
         → torch.IntTensor(toks).unsqueeze(0) → [1, L] int32 → F.pad((0,1), value=1) → [1, L+1]  L726
         （value=1 = stop_text_token，pad 在尾部）
 └─ lang = torch.LongTensor([lang_to_token(lang)]).to(device)  L727  （LANGUAGE_DICT 索引：en=0,zh=1,de=2,es=3,ru=4,ko=5,fr=6,ja=7,...；utils/tokenizer.py:157）
```

**tokenizer 细节**（`/root/index-tts/indextts/utils/tokenizer.py`）：
- `get_tokenizer(multilingual=True, model_dir=model_dir)` L235 → `WhisperTokenizer`（whisper.tokenizer.Tokenizer 子类，tiktoken 驱动）。
- `get_encoding(name="multilingual_zh_ja_yue_char_del", num_languages=99, model_dir)` L52-89：读取 `{model_dir}/multilingual_zh_ja_yue_char_del.tiktoken`（60509 基础 token），按固定顺序追加 specials（`<|endoftext|>`、`<|startoftranscript|>`、99 个 `<|lang|>`、AUDIO_EVENT、EMOTION、`<|translate|>` 等、`<|SPECIAL_TOKEN_1..30|>`、TTS_Vocal_Token、1501 个 `<|0.00|>..<|30.00|>`）。text_embedding 维度 60510（= 60509*types(1)+1）。
- pat_str：`'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`。
- 注意：v2（infer_v2.py）用 sentencepiece `TextTokenizer(bpe.model)`；**v2.5 是 tiktoken**，别混。

**输出接缝**：`text_tokens` = `[1, L+1]` torch.int32, cuda，末位=1（stop_text_token）；`lang` = `[1]` int64。

---

## 阶段 2：w2v-bert 特征提取（语义前端，speaker/emo 条件）

初始化（infer_v2_5.py:173-179）：
```python
self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_dir, local_files_only=True)
self.semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_dir, local_files_only=True)
stat_mean_var = torch.load(model_dir/cfg.w2v_stat)   # wav2vec2bert_stats.pt
self.semantic_mean = stat_mean_var["mean"]           # [1024]
self.semantic_std  = torch.sqrt(stat_mean_var["var"])# [1024]
```

`get_emb`（L282-291）：
```python
vq_emb = self.semantic_model(input_features=..., attention_mask=..., output_hidden_states=True)
feat = vq_emb.hidden_states[17]   # (B, T, 1024)
feat = (feat - self.semantic_mean) / self.semantic_std
```

调用链（speaker 条件，L625-656）：
1. `audio, sr = _load_and_cut_audio(spk_audio_prompt, 15)` — librosa.load（默认 sr=None → 原始采样率），截断 15s，`[1, N]`。
2. `audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)` → `[1, N16]` float32。
3. `inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")` → **实测**：`input_features [1, 149, 160] float32`（3s@16k；帧数 ≈ N16/320 + 1，含 padding），`attention_mask [1, 149] int32`（1=有效，0=padding）。
4. `get_emb` → **实测**：`hidden_states[17] = [1, 149, 1024] float32`。w2v-bert config：24 conformer 层，hidden_size=1024，`num_hidden_layers=24` → hidden_states 共 25 项（[0]=CNN 特征提取器输出，[17]=第 17 层后，1-indexed）。
5. 归一化后 `spk_cond_emb = [1, T, 1024] float32`（同 device 作为 GPT 条件）。

emo 条件（L684-695）：同路径，参考音频按 sr=16000 直接 load → `emo_cond_emb = [1, T, 1024]`。

**rewrite 要点**：需要复刻 transformers Wav2Vec2BertModel 的 encoder_frontend（特征提取 CNN：conv1 128@10/5 stride → 3×conv 512@3/2 → 512@3/2 → 512@3/2 → hidden 1024@3/2 全部 @2 stride，conv_depthwise_kernel_size=31, feature_projection_input_dim=160）+ 24 层 conformer（relative_key 位置编码、rotary、FFN 4096）+ dropout(layerdrop)。**零 JIT 约束下需自写 conformer**（AGENTS.md 已定）。输入归一化/波形处理细节 = transformers `Wav2Vec2BertFeatureExtractor`（norm：`(x - mean)/sqrt(var)`，mean=0.0, var=1.0 → 等价直接除以 sqrt(160)；padding_value=1 在 mask 外）。

---

## 阶段 3：CAMPPlus（全局说话人风格向量）

初始化（infer_v2_5.py:219-221）：`CAMPPlus(feat_dim=80, embedding_size=192)`，加载 `campplus_cn_common.bin` state_dict。

推理（L643-649）：
```python
feat = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)  # [T_f, 80]
feat = feat - feat.mean(dim=0, keepdim=True)          # 逐特征维度去均值
style = self.campplus_model(feat.unsqueeze(0))        # [1, T_f, 80] → [1, 192]
```
- 输入：fbank 特征（16kHz，80 bins），`[1, T_f, 80]` float32。
- 输出：**`[1, 192]` float32**（实测）。
- 结构：FCM（2D CNN 下采样 + 1×80→ 通道 32）→ DTDNN blocks（TDNN 5→stride2 + 3 个 CAMDenseTDNNBlock (12/24/16 层, growth 32, dilation 1/2/2) + transit）→ StatsPool（均值+标准差）→ DenseLayer(2×channels → 192)。
- v2.5 中 `campplus_model` 输出**不做 L2 归一化**直接当 style。

---

## 阶段 4：GPT-AR（UnifiedVoice，`/root/index-tts/indextts/gpt/model_v2.py`）

初始化：`UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")`（infer_v2_5.py:160），加载 gpt.pth，`post_init_gpt2_config(kv_cache=True, half=False)` → 构造 `GPT2InferenceModel`（L337-366）。

### 4a. emovec 融合（infer_v2_5.py:760-767）
```python
emovec = self.gpt.merge_emovec(spk_cond_emb, emo_cond_emb,
             torch.tensor([spk_cond_emb.shape[-1]]), torch.tensor([emo_cond_emb.shape[-1]]), alpha=emo_alpha)
```
- `merge_emovec`（model_v2.py:742-749）：`base_vec = get_emovec(spk)`，`emo_vec = get_emovec(emo)`，`out = base_vec + alpha*(emo_vec - base_vec)`。
- `get_emovec`（model_v2.py:735-740）：`get_emo_conditioning` = `emo_conditioning_encoder`（ConformerEncoder 1024→512, 4 blocks）→ `emo_perceiver_encoder`（1 latent）→ `emovec_layer`(Linear 1024→1280) → `emo_layer`(Linear 1280→1280)。
- **实测输出 `emovec = [1, 1280] float32`**。

### 4b. inference_speech（model_v2.py:610-733，入口调用在 infer_v2_5.py:772-790）
参数：`spk_cond_emb [1,T,1024]`、`text_tokens [1,L+1] int32`、`lang [1]`、`emo_cond_emb [1,T,1024]`、`emo_vec [1,1280]`、`campplus_embedding [1,192]`、`max_generate_length=1500`、`do_sample=True, top_p=0.8, top_k=30, temperature=0.8, num_beams=3, repetition_penalty=10.0`。

内部流程（spk_cond_mode="campplus"）：
1. `speech_conditioning_latent = spk_emb_proj(campplus_embedding)` → Linear(192→1280) → unsqueeze → `[1,1,1280]`（L628-643）。
2. `conds_latent = cat([spk_latent + emovec.unsqueeze(1) (1,1,1280), zeros(1,2,1280)], dim=1)` → `[1,3,1280]`（L663-668）。3 个 cond token = [spk+emo, 0, 0]。
3. `prepare_gpt_inputs(conds_latent, text_inputs, langs)`（L509-573）：
   - 去 stop/start text token → 前后加 start_text_token(0)/stop_text_token(1) → `text_input` `[L+2]`。
   - `text_emb = text_embedding(text_input) + text_pos_embedding.emb(arange)`（**text_pos_embedding 容量 602 = 600+2**）；campplus 模式额外 `+ lang_embedding(langs[i])`（107 个）。
   - `mel_emb = cat([conds(3), text_emb(L+2)])` → `[3+L+2, 1280]` = target_len；padding 时左侧补 0 并把 attention_mask 对应位置 0。
   - 返回 `fake_inputs [1, target_len+1]`（全 1 + 末位=start_mel_token 8192）、`inputs_embeds [1, target_len, 1280]`、`attention_mask [1, target_len+1]`。
4. `inference_model.store_mel_emb(inputs_embeds)`；HF `generate(inputs, bos_token_id=8192, eos/pad=8193, max_length=trunc_index+1500, ...)`（L725-733）。accel 引擎（flash_attn，非必需）在 `use_accel` 时替换。
5. **返回 `codes = output[:, trunc_index:]` → `[1, M] int64`，以及 `speech_conditioning_latent [1,1,1280]`**（L737-738）。实测 codes 范围 115..8121（合法域 0..8193；8192=start，8193=stop/EOS）。

### 4c. latent 前向（仅 use_gpt_latent=True 时，infer_v2_5.py:829-844）
```python
latent = self.gpt(speech_conditioning_latent[1,1,1280], text_tokens, text_len,
                  codes [1,M], code_len, emo_cond_emb, cond_mel_lengths=..., emo_cond_mel_lengths=...,
                  emo_vec=emovec, use_speed=zeros(B).long())
```
- `UnifiedVoice.forward`（model_v2.py:555-636）：mel codes padding（stop_mel_token）+ 左右 start/stop 包裹 → `mel_emb = mel_embedding + mel_pos_embedding`；conds 同 4b；GPT2 前向；`get_logits(..., return_latent=True)` 返回 mel 位置 latent，`return mel_logits[:, :-2]`。
- **实测 `latent = [1, M, 1280] float32`**（M = codes 长度）。

**⚠️ 已实测发现的风险**：v2.5 的 `use_gpt_latent=True` 路径存在 shape 冲突 —— `latent` 经 `gpt_layer`(Linear 1280→256→128→1024) 为 `[1, M, 1024]`，而 `S_infer = codec.decode(codes)` 为 `[1, 2M, 1024]`，`S_infer + latent` 广播失败。**v2.5 默认 `use_gpt_latent=False`（构造器默认），重写只需对齐默认路径**；`use_gpt_latent` 分支按现状复刻或标注为不工作。

### 4d. 生成内部细节（GPT2InferenceModel.forward，model_v2.py:63-165）
- `self.gpt.wte = self.mel_embedding`（post_init_gpt2_config 末行，L366）。
- 首步：`input_ids[:, mel_len:]` 部分用 `mel_embedding(fake_ids) + mel_pos_embedding`（`text_pos_embedding` 参数名实为 mel pos emb），前面 mel_len 段替换为 cached `inputs_embeds`。
- 后续步：`emb = mel_embedding(input_ids) + mel_pos_embedding.get_fixed_embedding(attn_len - mel_len)`（KV cache 模式 input_ids 只取最后 1 个 token）。
- `lm_head = nn.Sequential(final_norm(LayerNorm 1280), mel_head(Linear 1280→8194))`。

**接缝**：GPT 输出 `codes`：`[1, M]` int64 cuda，值域 0..8193，以 8193(EO) 结束或截断于 max_generate_length。下游用它做 `codec.decode` 和（v2 风格）`vq2emb`。

---

## 阶段 5：EnhancedCodec（语义 codec，`/root/index-tts/indextts/codec/models.py`）

初始化（infer_v2_5.py:182-186）：`EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)` + `load_checkpoint(codec.pth)`。实测：`downsample_scale=2`、`num_quantizers=1`、codebook `[8192, 8]`（weight_norm in_project 8×1024、out_project 1024×8）。

### decode（生成路径用，models.py:146-171）
```python
S_infer = self.semantic_codec.decode(codes)   # infer_v2_5.py:851
```
- 输入 `codes [1, M] int64`（0..8192，含可能的 stop token）。
- `codes.unsqueeze(0)` → `[1, 1, M]` → `quantizer.vq2emb`：`F.embedding(code, codebook) [1, M, 8]` → transpose → `[1, 8, M]` → out_project(8→1024 Conv1d) → `[1, 1024, M]`。
- `decoder`（VocosBackbone(1024→384, 12 层 ConvNeXt, LayerNorm) + Linear(384→1024)）→ `[1, M, 1024]`。
- upsample：transpose → `[1, 1024, M]` → `F.interpolate(scale_factor=2, mode="nearest")` → `up`(Conv1d 1024→1024 k3 s1 p1) → transpose → **`[1, 2M, 1024] float32`（实测）**。

### quantize（训练/数据路径，v2.5 主流程不用；models.py:120-144）
- 输入 `x [1, T, 1024]`（= 归一化 hidden_states[17]）。
- `down`(Conv1d 1024→1024 k3 s2 p1) + GELU → `[1, T/2, 1024]`（奇数帧先 `x[:, :-1]`）；encoder（同上）→ ResidualVQ `[1, 1024, T/2]`。
- **实测输出 `codes [1, T/2] int64`（0..8191）、`quantized_out [1, T/2, 1024]`**（注意：quantize 的 codes 长度 = T/2，与 GPT 生成的 mel codes 长度不同——两套"code"概念，勿混）。
- 结构中 encoder/decoder 都是 `VocosBackbone`（ConvNeXt 深度可分离，**纯 torch 可复刻**，无编译依赖）。

---

## 阶段 6：S2Mel-CFM（`/root/index-tts/indextts/s2mel/modules/`）

`MyModel`（commons.py:471-508）：`ModuleDict{cfm: CFM(args), length_regulator: InterpolateRegulator}`；`use_gpt_latent=True` 时另有 `gpt_layer = Sequential(Linear(1280,256), Linear(256,128), Linear(128,1024))`。加载后调用 `cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)`（infer_v2_5.py:198，**必须**，否则 freqs_cis 断言失败）。

### 6a. prompt_condition（参考段，L651-658）
```python
prompt_condition = s2mel.models['length_regulator'](
    spk_cond_emb,             # [1, T_cond, 1024]  归一化 hidden_states[17]（未量化！）
    ylens=ref_target_lengths, # [1] = ref_mel.size(2)（80 维 mel 帧数，实测 3s→172 帧）
    n_quantizers=3, f0=None)[0]
```
- `InterpolateRegulator.forward`（length_regulator.py:127-169）：is_discrete=false → `content_in_proj`(Linear 1024→512)；`F.interpolate(x.T, size=ylens.max(), mode='nearest')`（sampling_ratios=[1,1,1,1] 非空 → interpolate=True）；`model = [Conv1d(512,512,3,1,1)+GroupNorm+Mish]×4 + Conv1d(512,512,1,1)`；`out = out * sequence_mask(ylens).unsqueeze(-1)`。
- **实测 `prompt_condition = [1, ref_mel_frames, 512] float32`**。
- `n_quantizers=3` 是 MaskGCT 遗留参数：v2.5 配置 `n_codebooks=1` 且 `is_discrete=false` → **纯 no-op**。

### 6b. content 条件（L851-858）
```python
S_infer = semantic_codec.decode(codes)                    # [1, 2M, 1024]
target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * duration_factor)])  # 默认 duration_factor=1.0
cond = s2mel.models['length_regulator'](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
```
- **实测 `cond = [1, target_len, 512] float32`**（target_len = round(2M×1.72)）。
- `cat_condition = torch.cat([prompt_condition, cond], dim=1)` → `[1, ref_frames + target_len, 512]`。

### 6c. CFM inference（flow_matching.py:37-59；调用 infer_v2_5.py:861-867）
```python
vc_target = cfm.inference(cat_condition,                       # mu   [1, R+T, 512]
                          torch.LongTensor([cat_condition.size(1)]),  # x_lens
                          ref_mel,                             # prompt [1, 80, R]
                          style,                               # [1, 192]
                          None,                                # f0
                          diffusion_steps=25, inference_cfg_rate=0.7)
vc_target = vc_target[:, :, ref_mel.size(-1):]                 # 去 prompt 段
```
`solve_euler`（L61-112）循环体（**25 步 Euler，t_span=linspace(0,1,26)**）：
1. `z = randn([B, 80, R+T]) * temperature(1.0)`；`prompt_x`：前 R 帧 = ref_mel，`x[:, :, :R] = 0`；每步后 `x[:, :, :R] = 0`（prompt 区强制 0）。
2. CFG：`stacked = cat([x,prompt_x,mu,style, t], [orig, zeros])` 一次前向（batch 翻倍）→ `dphi_dt = (1+0.7)*d - 0.7*d_null`。
3. `dphi_dt = self.estimator(stacked_x [2B,80,T], stacked_prompt_x, x_lens [B](不翻倍), stacked_t [2], stacked_style [2B,192], stacked_mu [2B,T,512])`。
4. `x = x + dt * dphi_dt`。
- **实测 `vc_target = [1, 80, R+T]`，sliced → `[1, 80, T]` float32**。

### 6d. DiT.forward（diffusion_transformer.py:222-288，6 个位置参数 + mask_content=False）
```
x [B,80,T], prompt_x [B,80,T], x_lens [B], t [B](float 0..1), style [B,192], cond(=mu) [B,T,512]
```
1. `t1 = t_embedder(t)`：sinusoidal，scale=1000 → MLP → `[B, 512]`。
2. `cond = cond_projection(cond)`（Linear 512→512，content_dim=512，**内容连续路径**；`content_type='discrete'` 的 `cond_embedder` 不用）。
3. `x_in = cat([x, prompt_x, cond], -1)`（80+80+512=672）→ `+ style[:,None,:].repeat(1,T,1)` → **864** → `cond_x_merge_linear`(Linear 864→512) → `[B, T, 512]`。
4. `x_mask = sequence_mask(x_lens, max_length=T)` → `x_mask_expanded [1,1,T,T]`（is_causal=false，非因果）。
5. `x_res = transformer(x_in, t1.unsqueeze(1) [B,1,512], input_pos=arange(T), mask=x_mask_expanded)`：LLaMA 式 13 层（RMSNorm + RoPE(base 10000) + SwiGLU FFN 2048 + uvit_skip_connection 前后半层拼接）。**必须已 setup_caches(1, 8192) 预计算 freqs_cis**。
6. `long_skip_connection=true`：`x_res = skip_linear(cat([x_res, x]))`（Linear 512+80→512）。
7. wavenet 头：`conv1`(Linear 512→512) → `wavenet(x, x_mask, g=t2.unsqueeze(2))`（8 层 WN，gin_channels=512，non-causal）+ `res_projection(x_res)` → `final_layer(x, t1)`(LayerNorm+adaLN+Linear 512→512) → `conv2`(Conv1d 512→80)。
8. 输出 `[B, 80, T]`。

---

## 阶段 7：BigVGAN（vocoder，`/root/index-tts/indextts/s2mel/modules/bigvgan/bigvgan.py`）

```python
wav = self.bigvgan(vc_target.float())   # infer_v2_5.py:872
```
- **注意用的是 s2mel 版 BigVGAN**（`forward(self, x)` 单参数，无 speaker encoder；`indextts/BigVGAN/bigvgan.py` 是带 ECAPA 的 v1 版本，v2.5 不用）。
- 输入 `[1, 80, T]` float32（T 个 mel 帧）。
- 结构：conv_pre(Conv1d 80→1536 k7) → 6 级 upsample ConvTranspose1d（rates [4,4,2,2,2,2]，kernels [8,8,4,4,4,4]，初始通道 1536 逐级减半）→ AMPBlock1（resblock_kernel_sizes [3,7,11]，dilation [[1,3,5]×3]，snakebeta，alias-free Activation1d torch 版）→ activation_post(snakebeta) → conv_post(→1, k7, bias=False) → `use_tanh_at_final=false` → **clamp[-1,1]**。
- **实测输出 `[1, 1, T*256] float32`**（256×上采样；344 帧 → 88064 样本）。
- 后处理（infer_v2_5.py:873-877）：`.squeeze().unsqueeze(0)` → `[1, T*256]` → `.squeeze(1)` 同形 → `wav = torch.clamp(32767 * wav, -32767.0, 32767.0)` → `torchaudio.save(..., wav.type(int16), 22050)`。
- 权重加载：`from_pretrained(dir)` 读 `config.json` + `bigvgan_generator.pt`（`_from_pretrained` L285-326），随后 `remove_weight_norm()`（L230）。

---

## 追加问题解答

### Q1: `hf_cache/semantic_codec/model.safetensors`（MaskGCT）与 `codec.pth` 是什么关系？
**两个不同模型，同架构家族、不同权重**：
- `model.safetensors`（177MB）来自 `amphion/MaskGCT` 仓库（model_download.py:189-192），加载进 `indextts/utils/maskgct/models/codec/kmeans/repcodec_model.py` 的 `RepCodec`（`codec/maskgct_codec.py:1-7`）。仅 **infer_v2.py（v2 路径，L137-140）** 与数据预处理脚本 `s2mel/wav2vecbert_extract.py:103-105` 使用。RepCodec 默认 `downsample_scale=1`。
- `codec.pth`（607MB）是 IndexTTS-2.5 的 `EnhancedCodec`（`codec/models.py`），`downsample_scale=2`（已从 ckpt 键验证：含 `down.weight/up.weight` 1024×1024×3）。**v2.5 推理只用这个**。
- 两者同为 VocosBackbone encoder/decoder + ResidualVQ(fvq, 8192×8 codebook, 1 quantizer)，但：结构细节不同（EnhancedCodec 有 down/up 2× 下采样；RepCodec 无）、权重不同、用途不同。
- **结论：v2.5 重写只需要 codec.pth（EnhancedCodec）；`hf_cache/semantic_codec/` 可忽略。**

### Q2: `conformer_shaw.pt` 怎么加载到 Wav2Vec2BertModel？
**不加载，直接忽略**。`w2v-bert-2.0/` 目录（= `facebook/w2v-bert-2.0` 仓库的 snapshot_download 结果，model_download.py:169-183）同时包含：
- `conformer_shaw.pt`（2.33GB）：fairseq 原版 checkpoint（reach-vb/conformer-shaw）。
- `model.safetensors`（2.32GB）：**已转换为 transformers 格式的权重**，`Wav2Vec2BertModel.from_pretrained(dir, local_files_only=True)` 只读它。
- `config.json`（1.9KB）：transformers Wav2Vec2Bert 配置（architectures=["Wav2Vec2BertModel"], hidden_size=1024, num_hidden_layers=24, feature_projection_input_dim=160, 等）。
- `preprocessor_config.json`：SeamlessM4TFeatureExtractor（sampling_rate=16000, num_mel_bins=80, stride=2, padding_value=1）。
- 仓库内没有任何把 conformer_shaw.pt 转成 model.safetensors 的脚本（grep 全仓库无 `conformer_shaw` 引用）；转换是 HF 侧 / 发布时完成的。`.msc`/`.mv` 是 ModelScope 下载元数据。
- **重写时直接以 config.json + model.safetensors（2.3GB）为准，用 `torch.load`/`safetensors` 加载，自写 conformer 网络结构对齐 transformers Wav2Vec2BertModel 的状态字典键名。**

---

## 端到端 shape 链路（默认 use_gpt_latent=False, batch=1, duration_factor=1.0）

```
text → [1, L+1] int32 + lang [1]
ref wav 16k → feature extractor → [1, F, 160] + mask [1, F] → w2v-bert → hs[17] [1, F, 1024] → norm → spk_cond_emb [1, F, 1024]
ref wav 16k → kaldi fbank [Tf, 80] → mean-sub → [1, Tf, 80] → CAMPPlus → style [1, 192]
ref wav 22k → mel_fn → ref_mel [1, 80, R]
length_regulator(spk_cond_emb, ylens=[R]) → prompt_condition [1, R, 512]
GPT.merge_emovec → emovec [1, 1280]
GPT.inference_speech(spk_cond_emb, text_tokens, lang, emo_cond_emb, emovec, style, max_gen=1500)
   → codes [1, M] int64 (0..8193), spk_latent [1, 1, 1280]
codec.decode(codes) → S_infer [1, 2M, 1024]
length_regulator(S_infer, ylens=[round(2M*1.72)]) → cond [1, round(3.44M), 512]
cat_condition [1, R + 3.44M, 512]  (mu)
CFM.inference(mu, [R+T], ref_mel, style, 25 步, cfg=0.7) → [1, 80, R+T] → slice → [1, 80, T]
BigVGAN([1, 80, T]) → [1, 1, 256T] → clamp(32767·x, ±32767) → int16 @22050
```

## 重写建议起点
1. **`/root/index-tts/indextts/infer_v2_5.py:570-910`（infer_generator）** — 唯一权威主流程，所有接缝的调用方。
2. 其次逐模块对齐：`gpt/model_v2.py`（GPT 结构最复杂，但生成可用 HF generate 语义自写 KV-cache 循环）、`codec/models.py`（简单，可直接复刻）、`s2mel/modules/{flow_matching, length_regulator, diffusion_transformer, gpt_fast/model, wavenet}.py`、`s2mel/modules/bigvgan/bigvgan.py`。
3. 每个模块的数值对齐测试物：参考音频 `/root/WIndexTTS/test.wav`，权重 `/root/IndexTTS-2.5/`，官方 venv `/root/index-tts/.venv`（Python 3.11）。

## 风险与注意点
- **`use_gpt_latent=True` 在 v2.5 有 shape 冲突**（2M vs M），默认 False；重写只需对齐默认路径。
- `n_quantizers=3` 在 v2.5 length_regulator 是死参数（n_codebooks=1 + 连续输入）。
- 1.72 系数作用于 **S_infer 长度（2M）**，即等效 3.44 mel 帧/code（与 v2 的 code_lens*1.72 不同）。
- w2v-bert 是全流程最大依赖（2.3GB、24 层 conformer），自写量最大；需严格对齐 `hidden_states[17]`（含 layerdrop=0.1 训练态随机性、position_embeddings_type=relative_key）。
- GPT 采样参数（top_p 0.8 / top_k 30 / temp 0.8 / rep_penalty 10.0 / num_beams=3, do_sample=True）在 `infer_generator` 内默认，重写时需一致才对齐听感；数值对齐测试建议用 do_sample=False 固定输出。
- 声音输出范围 [-1,1]（BigVGAN clamp），落盘 ×32767 → int16 @22050Hz。