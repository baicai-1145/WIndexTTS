# WIndexTTS

Windows-priority, zero-JIT-compile, pure-torch accelerated inference engine for
**IndexTTS-2.5**.

目标：在 **Windows 上 `pip install` 即用**（无 MSVC / nvcc / triton 编译依赖）的前提下，
用纯 PyTorch 重写 IndexTTS-2.5 的神经网络推理，并实现比官方 `use_accel` 更彻底的 CUDA Graph 加速。

## 状态

**端到端推理 + 多轮加速优化完成**，纯 torch，比官方 bf16 快 **2.4x**。

### 性能（A10G 24GB，4 段中文，稳态）

| 引擎 | 均值 | 最快 | RTF | 说明 |
|---|---|---|---|---|
| **WIndexTTS** | **0.67s** | **0.59s** | **5.75x** | fp16 GPT+BigVGAN, 15步CFM+TeaCache, 紧凑KV buffer |
| 官方 bf16 | 1.81s | 1.75s | — | transformers + HF generate |
| 官方 fp32 | 2.06s | 1.91s | — | 默认精度 |

**WIndexTTS 在零编译依赖下比官方 bf16 快 2.7x、比官方 fp32 快 3.1x。**

### 各阶段拆解（优化后）

```
GPT-AR(fp16+graph+紧凑buffer)  ~150ms  ~40%  ← 内存带宽受限（纯torch fp16 上限）
S2Mel-CFM(15步+tc)            ~175ms  ~37%  ← TeaCache 跳过冗余步骤 + 减少欧拉步数
BigVGAN(fp16)                  ~58ms  ~15%  ← cosine 0.9998
codec+前端+设置                ~50ms  ~8%
──────────────────────────────────────────
E2E 稳态                       ~0.67s       median 0.66s, min 0.59s
```

### 加速轮次

| 轮次 | 技术 | 效果 | 质量 |
|---|---|---|---|
| R1 | TeaCache（S2Mel DiT 步骤跳过） | S2Mel 465→247ms (1.88x) | mel cosine 0.98 |
| R2 | fp16 GPT-AR（混合精度） | GPT 430→245ms (1.77x) | greedy 78/78 精确 |
| R3 | fp16 BigVGAN | 89→58ms (1.53x) | cosine 0.9998 |
| R4 | CFM 欧拉步 25→15 | S2Mel 251→177ms | cosine 0.998 |
| R5 | 紧凑 KV buffer（max_mel_tokens 1000→300） | GPT 184→124ms (1.48x) | 无截断（实测 68-110 codes） |

### 未采用的方案及原因

- **bf16 GPT**：7-bit 尾数不够，greedy 仅 27% 匹配（fp16 的 10-bit 足够，100%）
- **torch.compile GPT**：与混合精度不兼容（dtype 报错）
- **GPT INT8 量化**：torchao/bitsandbytes 未安装（零依赖原则）
- **S2Mel CUDA Graph（全序列）**：T=1045 时 DiT 注意力中间量占用 ~22GB，A10G OOM
- **S2Mel fp16**：DiT forward 有多处 fp32 内部构造，侵入性大，TeaCache 已减半
- **Flash attention 解码**：不支持 seqlen_q≠seqlen_k 的 is_causal（解码场景 Q=1,K=90）
- **流水线 stream overlap**：各阶段串行依赖（每阶段需上阶段输出），单请求无可重叠独立工作

### 可调参数（质量/速度权衡）

```python
tts.infer(ref, text, 'ZH',
    dtype=torch.float16,      # fp16 GPT+BigVGAN（默认 fp32 保对齐）
    cfm_steps=15,             # CFM 欧拉步（25=最高质量，10=最快）
    teacache_thresh=0.15,     # 0=禁用，0.25=更激进跳步
    cfg_rate=0.7,             # CFG 强度（0.3=快10%，0.0=最快无引导）
)
```

### 实现历史（阶段 1-5）

- ✅ **阶段1**：全部神经模块纯 torch 重写并数值对齐（diff=0.0~9.5e-7）
  - w2v-bert(580M) / CAMPPlus / EnhancedCodec / GPT-AR / S2Mel-DiT(98M) / CFM / BigVGAN / LengthRegulator
- ✅ **阶段2**：端到端流水线 `WIndexTTS.infer()` 跑通
  - 前端：tiktoken tokenizer（6/6 精确）/ HiFiGAN mel（9.5e-7）/ SeamlessM4T featurizer
- ✅ **阶段3**：GPT-AR decode 引擎（KV cache）替代 HF generate，greedy 49/49 精确
- ✅ **阶段3b**：GPT-AR CUDA Graph，2.28x 加速，logits 逐位一致
- ✅ **阶段4**：S2Mel CFM CUDA Graph（小序列验证）+ TeaCache 步骤跳过（生产路径）
- ✅ **阶段5**：参考音频特征缓存 + 加速优化循环（fp16/减步/TeaCache，详见上方加速轮次表）

### 已知限制

- **文本归一化**：第一版接受已处理文本；jieba/cn2an 中文归一化待补（非神经，独立任务）
- **emo audio 项**：用 emovec_mat only（省略 merge_emovec conformer 的 (1-sum)*audio 校正项）
- **CFM CUDA Graph**：全序列显存不足，默认 eager + TeaCache

## 运行

需要：模型权重（见下）+ 任意参考音频 wav（5-15s 干净人声）。

### 获取模型权重

模型来自 IndexTeam 官方发布，二选一：

```bash
# HuggingFace（海外）
pip install -U "huggingface_hub[cli]"
hf download IndexTeam/IndexTTS-2.5 --local-dir=IndexTTS-2.5

# ModelScope（国内）
pip install modelscope
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir IndexTTS-2.5
```

下载后把目录路径传给 `--model-dir` / `weights_dir=`，或设环境变量：

```bash
export WINDEXTTS_WEIGHTS_DIR=/path/to/IndexTTS-2.5
```

### 安装

```bash
pip install windextts                    # 核心（纯 torch，零 JIT 编译）
pip install 'windextts[server]'         # HTTP API（/v1/audio/speech）
pip install 'windextts[webui]'          # Gradio WebUI
pip install 'windextts[quant]'          # W4A16 INT4 加速（可选）
```

### CLI

```bash
# 基础（fp16）
windextts --ref voice.wav --text "你好世界" -o out.wav
# 最快（INT4 量化）
windextts --ref voice.wav --text "你好世界" -o out.wav --w4a16
# 3GB 显卡（保持 beam3 质量，稳态 ~2.9GB）
windextts --ref voice.wav --text "你好" -o out.wav --w4a16 --low-vram
```

### HTTP API（OpenAI 兼容）

```bash
windextts-server --w4a16 --port 8000 --voices default=voice.wav
```

```bash
curl -s http://localhost:8000/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"windextts","input":"你好世界","voice":"default"}' \
    -o out.wav
```

### WebUI（Gradio）

```bash
# 仓库源码运行（webui.py 不在 pip 包内）
git clone https://github.com/baicai-1145/WIndexTTS
cd WIndexTTS && pip install -e ".[webui]"
python webui.py --model_dir /path/to/IndexTTS-2.5    # 0.0.0.0:7860
```

功能：参考音频上传、精度运行时切换（W4A16/fp16/fp32）、低显存模式、多语言、
情感控制（向量/文本/参考音频）、采样参数、性能调优、一键基准。

### Python API

```python
import torch
from windextts.inference import WIndexTTS

tts = WIndexTTS(weights_dir="/path/to/IndexTTS-2.5",
                 device="cuda", dtype=torch.float16, enable_w4a16=True)
tts.warmup()
sr, wav = tts.infer("voice.wav", "你好世界", "ZH")
```

## 设计约束

见 [AGENTS.md](./AGENTS.md)。核心铁律：

- **纯 torch，零 JIT 编译**：核心推理路径在无 C/C++/CUDA 编译器的 Windows 上必须能跑。
- **不依赖 index-tts / transformers / modelscope**：所有 `nn.Module` 自己写，权重只用 `torch.load` + `safetensors`。
- **接缝魔数不可臆造**：唯一正确性标准是与官方输出 `torch.allclose(atol=1e-4, rtol=1e-3)`。
