# WIndexTTS

Windows-priority, zero-JIT-compile, pure-torch accelerated inference engine for
**IndexTTS-2.5**.

目标：在 **Windows 上 `pip install` 即用**（无 MSVC / nvcc / triton 编译依赖）的前提下，
用纯 PyTorch 重写 IndexTTS-2.5 的神经网络推理，并实现比官方 `use_accel` 更彻底的 CUDA Graph 加速。

## 快速开始（Windows）

Windows 用户两种方式任选：

### 方式一：整合包（解压即用，推荐）

直接下载整合包（已内置模型权重，无需 Python / 编译工具链）：

<https://www.modelscope.cn/models/baicai1145/WIndexTTS-Releases/resolve/master/WIndexTTS-Windows-Portable.7z>

下载后解压，按包内说明运行即可。

### 方式二：pip 安装

```bash
pip install windextts
```

模型权重需单独下载，见下方「获取模型权重」。

## 状态

**已发布**：PyPI `windextts` v0.1.1 + Windows 整合包（解压即用、内置权重）。
核心推理纯 torch、零 JIT 编译依赖，端到端推理 + 多轮加速优化完成，
A10G 上默认档（beam3 官方质量）约 0.96s、fp16 极速档约 0.72s（最快 0.48s）、
W4A16 极速档最快 0.43s——比官方快约 2 倍，最快延迟追平并超过 vLLM-Omni fast。

能力：CLI / OpenAI 兼容 HTTP API / Gradio WebUI / Python API 四入口；
W4A16 INT4 量化（可选）、低显存模式（3GB 显卡可用）、中文文本归一化（jieba/cn2an/tn）、
多语言、情感控制（向量/文本/参考音频）。

### 性能（A10G 24GB，同协议实测：warmup + 4 段中文）

| 引擎 | 均值 | 最快 | RTF | 说明 |
|---|---|---|---|---|
| **WIndexTTS**（W4A16 greedy 极速） | **0.69s** | **0.43s** | **8.3x** | INT4 GPT + CUDA Graph（最快延迟） |
| **WIndexTTS**（W4A16 beam3 默认） | **0.76s** | **0.51s** | **7.0x** | INT4 GPT（比 fp16 快 ~1.25x） |
| **WIndexTTS**（fp16 greedy 极速） | **0.72s** | **0.48s** | **7.9x** | S2Mel CUDA Graph（fp16 DiT）+ 12 步 |
| **WIndexTTS**（fp16 beam3 默认） | 0.96s | 0.64s | 6.0x | R14 CUDA-Graph beam search（官方质量档） |
| vLLM-Omni fast（参照） | 0.51s | 0.47s | — | FlashInfer/Triton 全 graph（R14 记录） |
| 官方 accel+bf16 | 1.13s | 1.09s | — | 官方 CUDA Graph 加速版（R14 记录） |
| 官方 bf16 | 1.89s | 1.67s | — | transformers + HF generate |
| 官方 fp32 | 1.74s | 1.65s | — | 默认精度 |

**零编译依赖下：默认档比官方 fp32 快 ~1.8x、比官方 bf16 快 ~2.0x；fp16 极速档最快 0.48s、W4A16 极速档最快 0.43s，最快延迟追平并超过 vLLM-Omni fast（0.47s）。**

> WIndexTTS / 官方 fp32 / 官方 bf16 五行为同环境同协议实测（warmup + 4 段中文）；vLLM-Omni fast 与官方 accel
> 为 docs/PERFORMANCE.md 的 R14 记录（该文档为仓库本地文件，未随远端发布）。
>
> **W4A16 为有损加速档**：INT4 舍入使 mel codes 与 fp16/官方从首个 token 起即不同（波形 cosine
> ~0.02，不满足数值对齐标准），但人工试听正常、自然度可用；追求与官方逐位一致请用 fp16/fp32。

### 各阶段拆解（R14 记录，beam3 默认档）

```
GPT-AR beam3 (CUDA Graph 静态 batch K=3)          ~490ms  ~62%  ← eager beam3 1464ms → 3.0x
S2Mel-CFM (CUDA Graph, 15步；graph 与 TeaCache 互斥)  ~165ms  ~17%  ← fp16 DiT + graph（TeaCache 经校准+听感验证不可用，默认禁用）
BigVGAN (fp16 + remove_weight_norm)                ~59ms   ~8%
codec + 前端 + Python 设置                          ~25ms   ~3%
─────────────────────────────────────────────────────────
E2E beam3 默认档（今日实测）                       mean 0.96s / min 0.64s
E2E greedy 极速档（今日实测）                      mean 0.72s / min 0.48s
```

### 加速轮次（R1-R14 精选）

| 轮次 | 技术 | 效果 | 质量 |
|---|---|---|---|
| R1 | TeaCache（S2Mel DiT 步骤跳过） | S2Mel 465→247ms (1.88x) | mel cosine 0.98 |
| R2 | fp16 GPT-AR（混合精度） | GPT 430→245ms (1.77x) | greedy 78/78 精确 |
| R3 | fp16 BigVGAN | 89→58ms (1.53x) | cosine 0.9998 |
| R4 | CFM 欧拉步 25→15 | S2Mel 251→177ms | cosine 0.998 |
| R5 | 紧凑 KV buffer（max_mel_tokens 1000→300） | GPT 184→124ms (1.48x) | 无截断 |
| R8 | TeaCache 阈值 0.15→0.25 | S2Mel 185→165ms | cosine 0.999 |
| R9 | CFM 步数 15→12 | S2Mel 168→137ms | cosine 0.9995 |
| R12 | **S2Mel CUDA Graph 修复并启用**（dt_buf GC + freqs_cis rebuild 两个根因 bug） | eager 488→graph 442ms | 0/75 板砖，21/21 对齐 |
| R13 | **fp16-native DiT**（移除 .float() 精度守卫） | S2Mel 433→400ms | cosine 0.9997 |
| R14 | **CUDA-Graph beam search**（静态 batch K=3，无 KV 重排） | eager beam3 1464→~490ms，e2e 1.4→0.65s | 9/10 seed 位级一致 |

### 未采用/已回退的方案及原因

- **bf16 GPT**：7-bit 尾数不够，greedy 仅 27% 匹配（fp16 的 10-bit 足够，100%）
- **torch.compile GPT**：与混合精度不兼容（dtype 报错）；**torch.compile S2Mel**：TeaCache 动态状态触发持续重编译，慢 14x
- **GPT INT8 量化**：torchao/bitsandbytes 未安装（零外部依赖原则）
- **BigVGAN CUDA Graph**：纯 torch conv 实测 0.89x 反而变慢，且 fp16 引入数值偏差
- **bf16 DiT autocast**：autocast dispatch 开销在 batch=1 下净变慢 32ms（profiler-free A/B）
- **Flash attention 解码**：不支持 seqlen_q≠seqlen_k 的 is_causal（解码场景 Q=1,K≈90）
- **流水线 stream overlap**：各阶段串行依赖，单请求无可重叠独立工作
- **CFG=0**：可省 ~50ms 但偏离训练分布，保留为可调参数

> S2Mel CUDA Graph / fp16 DiT / beam3 Graph 曾被判定不可行，R12-R14 已逐一攻克启用，
> 根因分析与数值证据见仓库内 docs/PERFORMANCE.md（本地文件，未随远端发布）。

### 可调参数（质量/速度权衡）

```python
tts = WIndexTTS(weights_dir=..., device="cuda", dtype=torch.float16,  # fp16 最快保真；默认 fp32
                enable_w4a16=True)  # INT4 GPT（更快，有损，听感可用；需 pip install 'windextts[quant]'）

tts.warmup()  # 预捕获 CUDA Graph，把 ~1s 冷启动成本移出首请求
tts.infer(ref, text, 'ZH',
    num_beams=3,          # 默认官方质量档；1 = greedy 极速（最快）
    cfm_steps=15,         # CFM 欧拉步（15=听感无损下限；12=极速，音节末尾呼吸处轻微变差；25=最高）
    teacache_thresh=0.0,  # 0=禁用（默认，听感验证结论：任何有效跳步率均可感知劣化；graph 全量步已是最优）
    cfg_rate=0.7,         # CFG 强度（0.3=快10%，0.0=最快无引导）
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

- **emo audio 项**：用 emovec_mat only（省略 merge_emovec conformer 的 (1-sum)*audio 校正项）
- **S2Mel CUDA Graph 与 TeaCache 互斥**：graph 路径自动禁用 TeaCache（data-dependent 分支无法 capture）；
  eager + TeaCache 作为备选路径保留

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
# 任意环境（wheel 内置 webui，四个入口点均可 pip 安装）
pip install 'windextts[webui]'
windextts-webui --model_dir /path/to/IndexTTS-2.5    # 0.0.0.0:7860

# 或仓库源码运行
python webui.py --model_dir /path/to/IndexTTS-2.5
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
