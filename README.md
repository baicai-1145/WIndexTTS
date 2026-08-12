# WIndexTTS

Windows-priority, zero-JIT-compile, pure-torch accelerated inference engine for
**IndexTTS-2.5**.

目标：在 **Windows 上 `pip install` 即用**（无 MSVC / nvcc / triton 编译依赖）的前提下，
用纯 PyTorch 重写 IndexTTS-2.5 的神经网络推理，并实现比官方 `use_accel` 更彻底的 CUDA Graph 加速。

## 状态

**端到端推理已跑通**，纯 torch + GPT-AR CUDA Graph 加速，比官方 fp32 更快。

### 性能（A10G 24GB，fp32，稳态，4 段中文）

| 引擎 | 均值 | 最快 | 说明 |
|---|---|---|---|
| **WIndexTTS** | **1.68s** | **1.60s** | 纯 torch + GPT-AR CUDA Graph，零 JIT |
| 官方 IndexTTS | 2.06s | 1.91s | transformers + HF generate，fp32 |
| vLLM-Omni（参考上限） | ~3.8s RTF | — | 官方加速引擎（Linux/CUDA kernel） |

WIndexTTS 在零编译依赖下比官方 fp32 **快 ~20%**。

### 阶段进度

- ✅ **阶段1 完成**：全部神经模块纯 torch 重写并数值对齐（diff=0.0~9.5e-7）
  - w2v-bert(580M) / CAMPPlus / EnhancedCodec / GPT-AR / S2Mel-DiT(98M) / CFM / BigVGAN / LengthRegulator
- ✅ **阶段2 完成**：端到端流水线 `WIndexTTS.infer()` 跑通，产出有效语音
  - 前端：tiktoken tokenizer（6/6 精确匹配）/ HiFiGAN mel（9.5e-7）/ SeamlessM4T featurizer
- ✅ **阶段3 完成**：GPT-AR decode 引擎（KV cache）替代 HF generate，greedy 49/49 精确匹配
- ✅ **阶段3b 完成**：GPT-AR CUDA Graph，2.28x 加速（538→236ms），logits 逐位一致
- ✅ **阶段4 完成**：S2Mel CFM CUDA Graph 实现 + 对齐验证（小序列）
  - 注：全序列 T=1045 下 graph 捕获需 ~22GB（A10G OOM）；CFM 是计算密集非 launch 密集，graph 收益有限，默认走 eager
- ⏳ **阶段5**：流水线重叠（torch.cuda.Stream）—— 待做

### 各阶段耗时拆解（加速后）

```
w2v_spk       29ms   (一次性)
GPT-AR(graph) 406ms  ← CUDA Graph 加速 (was 1362ms)
codec          3ms
S2Mel-CFM    ~900ms  ← 当前最大瓶颈 (计算密集)
BigVGAN       86ms
─────────────────
总计        ~1.6s
```

### 已知限制

- **文本归一化**：第一版接受已处理文本；jieba/cn2an 中文归一化待补（非神经，独立任务）
- **emo audio 项**：用 emovec_mat only（省略 merge_emovec conformer 的 (1-sum)*audio 校正项）
- **CFM CUDA Graph**：全序列显存不足，默认 eager

## 运行

需要：模型权重（`/root/IndexTTS-2.5/`）、任意参考音频 wav。

> **注**：仓库不含权重和测试音频（见 `.gitignore`）。参考音频用任意 5-15s 干净人声 wav。

```bash
# 用官方 venv（有 torch/torchaudio/tiktoken/safetensors）
/root/index-tts/.venv/bin/python -c "
from windextts.inference import WIndexTTS
tts = WIndexTTS(device='cuda')
sr, wav = tts.infer('/path/to/ref.wav', '你好世界', 'ZH')
"

# benchmark（需准备 REF_AUDIO 指向的参考音频）
/root/index-tts/.venv/bin/python scripts/benchmark_e2e.py all
```

## 设计约束

见 [AGENTS.md](./AGENTS.md)。核心铁律：

- **纯 torch，零 JIT 编译**：核心推理路径在无 C/C++/CUDA 编译器的 Windows 上必须能跑。
- **不依赖 index-tts / transformers / modelscope**：所有 `nn.Module` 自己写，权重只用 `torch.load` + `safetensors`。
- **接缝魔数不可臆造**：唯一正确性标准是与官方输出 `torch.allclose(atol=1e-4, rtol=1e-3)`。
