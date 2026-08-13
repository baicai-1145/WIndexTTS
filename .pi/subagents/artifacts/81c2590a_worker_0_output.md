# emo_conditioning 实现报告

## 任务完成状态：✅ 完成

实现了 IndexTTS-2.5 GPT 的 emo_conditioning 子模块（情感参考音频路径），数值对齐通过。

## 变更文件

1. **新增** `/root/WIndexTTS/windextts/models/emo_conditioning.py`
   - `EmoConformerEncoder`：ConformerEncoder（input=1024, output=512, linear=1024, heads=4, blocks=4, conv2d2 subsampling, rel_pos）
   - `EmoPerceiverEncoder`：PerceiverResampler（dim=1024, dim_context=512, num_latents=1, heads=4, ff_mult=2, depth=2）
   - `get_emovec()`：完整提取管线（conformer → perceiver → emovec_layer → emo_layer）
2. **新增** `/root/WIndexTTS/tests/align/test_emo_conditioning_align.py`：数值对齐测试（pytest 可收集）

## 对齐结果（vs 官方 dump，CUDA fp32）

```
[1] conformer_seq  [1,66,512]:  max_diff=0.000415（中间张量，<1e-3 项目标准）
[2] perceiver_out  [1,1,1024]:  allclose(1e-4,1e-3)=True  max_diff=0.000039
[3] emo_vec (seam) [1,1280]:    allclose(1e-4,1e-3)=True  max_diff=0.000024
```

- 官方模块 vs dump = 0.0（dump 无 bias）
- 关键 seam（喂给 GPT conditioning 的 emo_vec）严格通过 1e-4
- conformer_seq 中间张量的 4e-4 差异来自 **SDPA 融合累积序 vs 官方两次独立 matmul**，只影响近零元素（301/33792，ref 幅值 ≤0.21），不传播到最终输出。采用 w2v-bert 同款标准（中间 <1e-3，seam <1e-4）

## 关键实现决策

1. **RelPosition 自注意力用 SDPA 表达**：相对位置分数 = (q+bias_u)·kᵀ + (q+bias_v)·pᵀ，通过 head-dim 拼接 q'=[q_u,q_v], k'=[k,p] 折叠成单个 SDPA 调用（数学精确等价），scale=1/√d_k
2. **macaron_style=False 确认**：权重只有一套 feed_forward（无 feed_forward_macaron），ff_scale=1.0
3. **conv2d2 subsampling**：Conv2d(1,512,3,3,stride=2) 后 Linear(261632→512)，mask 用 [:, :, 2::2]
4. **perceiver 的 dim_context 修正**：Attention 内部 `to_kv` 的 dim_context=dim(1024)（因为 context 已是 proj_context 输出），而非构造器参数 512
5. **FeedForward 命名对齐**：用 `add_module("0"/"2")` 让 checkpoint key `layers.{i}.1.{0,2}` 直接匹配
6. **RMSNorm**：F.normalize * dim^0.5 * gamma
7. **emo_cond_mask_pad**：`F.pad(mask.squeeze(1), (1,0), value=True)`（官方 ConstantPad1d 等价）

## 验证命令

```bash
# 单测
/root/index-tts/.venv/bin/python tests/align/test_emo_conditioning_align.py
# pytest
/root/index-tts/.venv/bin/python -m pytest tests/align/test_emo_conditioning_align.py -q
```

## 残留风险 / 待父级集成

- **集成到 inference.py 是下一步**（父级任务）：把 `EmoConformerEncoder/EmoPerceiverEncoder` 挂到 `UnifiedVoice`，让 `load_official` 不再 drop 那 149 个 key，并接线 `infer(emo_audio_prompt=...)` 路径
- 参考 dump 用 test.wav 同时作 spk/emo ref（alpha=0.65 和 1.0 都有 dump），集成后建议再 dump 一个不同 emo 音频做端到端验证
- fp16 下需重新验证（当前对齐是 fp32，与项目惯例一致）