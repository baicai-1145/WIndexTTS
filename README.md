# WIndexTTS

Windows-priority, zero-JIT-compile, pure-torch accelerated inference engine for
**IndexTTS-2.5**.

目标：在 **Windows 上 `pip install` 即用**（无 MSVC / nvcc / triton 编译依赖）的前提下，
用纯 PyTorch 重写 IndexTTS-2.5 的神经网络推理，并实现比官方 `use_accel` 更彻底的 CUDA Graph 加速。

**vs 官方 index-tts：核心推理代码少 1w+ 行（19513 → 2315 纯代码），速度更快（最高 4.6x），功能更多（CUDA Graph 全覆盖 / W4A16 / 低显存）。**

**Apple Silicon**：同时提供纯 MLX（Metal）推理后端 `windextts_mlx` —— 无需 torch/CUDA，支持
W4A16 INT4 量化、kernel 融合与低内存运行，详见下方「MLX 适配」。

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

**已发布**：PyPI `windextts` v0.4.0 + Windows 整合包（解压即用、内置权重）。最快延迟追平并超过 vLLM-Omni fast。

本版亮点：**统一 CLI 后端自动选择**（NVIDIA GPU → CUDA torch，Apple Silicon → 纯 MLX/Metal，`WINDEXTTS_BACKEND` 可强制）；**MLX 后端**随包分发（W4A16 INT4 + kernel 融合 + 前端用后释放，实测推理峰值 ~3.2GB）；性能持续优化（SDPA 注意力融合 / BigVGAN 子图编译，累计提速 ~31%）。

能力：CLI / OpenAI 兼容 HTTP API / Gradio WebUI / Python API 四入口；
W4A16 INT4 量化（可选）、低显存模式（3GB 显卡可用）、中文文本归一化（jieba/cn2an/tn）、
多语言、情感控制（向量/文本/参考音频）。

### 性能（A10G 24GB，统一协议实测：4 段长中文（成音 5.3-8.7s）、beam3 采样、warmup 后每档 12 次计时）

| 引擎 | 精度 | CFM 步数 | 解码 | RTF 均值 | 显存 | vs 官方 fp32 |
|---|---|---|---|---|---|---|
| 官方 fp32 | fp32 | 25 | beam3 采样 | 0.62 | 7.4G | 1.0x |
| 官方 bf16 | bf16 | 25 | beam3 采样 | 0.68 | 5.8G | 0.91x（更慢） |
| WIndexTTS 保真档 | fp32 | 25 | beam3 采样 | 0.55 | 8.9G | 1.13x |
| WIndexTTS fp32 | fp32 | 15 | beam3 采样 | 0.49 | 8.9G | 1.28x |
| WIndexTTS fp16 | fp16 | 25 | beam3 采样 | 0.30 | 4.7G | 2.04x |
| **WIndexTTS 默认档** | **fp16** | **15** | beam3 采样 | **0.24** | **5.5G** | **2.6x** |
| WIndexTTS W4A16 | fp16 | 12 | beam1 采样 | 0.17 | 4.5G | 3.6x |
| WIndexTTS W4A16 极速 | fp16 | 12 | greedy | **0.14** | **4.2G** | **4.6x** |
| WIndexTTS fp16 低显存 | fp16 | 15 | beam3 采样 | 0.19 | **3.6G** | 3.2x |
| WIndexTTS W4A16 低显存极速 | fp16 | 12 | greedy | **0.12** | **2.9G** | **5.3x** |
| vLLM-Omni（默认 deploy） | bf16 | 25 | beam1 采样 | 0.20 | ~18G* | 3.10x |

> ⚠️ **greedy（beam1+argmax）对效果影响较大**

### MLX 适配（Apple Silicon）

仓库随 `windextts` 一起分发**纯 MLX 推理后端 `windextts_mlx/`**：在 Apple Silicon（Metal）上原生运行，
**不依赖 torch / CUDA**；复用 torch 包的纯 Python 前端（tokenizer/normalizer/segmenter）与 config，
`pip install windextts` 即包含。统一入口 `windextts` 在 Apple Silicon 上会自动选用本后端
（详见下方「后端自动选择与 WINDEXTTS_BACKEND」）。

能力：**W4A16 INT4 量化**（GPT body 4bit）、**kernel 融合**（SDPA 注意力 / BigVGAN 子图编译）、
统一内存缓存上限控制、**前端用后释放**（首段推理后释放 w2v-bert/cam++/QwenEmotion，常驻 -1.1GB）。
回退开关：`WINDEXTTS_NO_ATTN_SDPA=1` / `WINDEXTTS_NO_ACT_COMPILE=1` / `WINDEXTTS_KEEP_FRONTEND=1` 等。

实测（M4，统一内存 16GB，greedy，RTF = 耗时/音频时长，>1 慢于实时，
与上方 A10G 性能表的 CUDA 协议 RTF 口径**不同**）：

| 配置 | 解码 | RTF | 推理峰值内存 |
|---|---|---|---|
| fp16 | greedy | 1.42x | ~3.9 GB |
| **fp16 + W4A16（最低内存）** | greedy | **1.14x** | **~3.2 GB** |
| fp16 + W4A16 | beam3 采样 | 1.53x | ~3.2 GB |

说明：RTF 为保守上界（含系统内存压力残留），更干净的窗口下可再降低；
W4A16 量化同时省内存（-0.7GB）且更快（int4 带宽减半）。

### 各阶段拆解（fp16 beam3 @15步，短句协议；fp32 对照见括号）

```
GPT-AR beam3 (CUDA Graph 静态 batch K=3)          ~526ms  ~51%  （fp32: 984ms，占 64%）
S2Mel-CFM (CUDA Graph, 15步)                      ~154ms  ~15%  （fp32: 382ms）
BigVGAN (fp16 + remove_weight_norm)                ~59ms   ~6%  （fp32: 80ms）
codec + 前端 + Python 设置                          ~25ms   ~3%
─────────────────────────────────────────────────────────
E2E（短句参考）                                    ~1.0s
长文本统一协议见上表（默认档 1.62s / 极速档 1.01s）
```

fp32 档瓶颈：GEMM 计算密度（GPT-AR 984ms 占 64%），graph 等手段只消调度开销；fp16 GEMM 2x 是最大单台阶。

### 加速轮次（R1-R14 精选）

| 轮次 | 技术 | 效果 | 质量 |
|---|---|---|---|
| R1 | ~~TeaCache（S2Mel DiT 步骤跳过）~~（已回退；代码已在 v0.2.1 移除） | S2Mel 465→247ms (1.88x) | mel cosine 0.98 |
| R2 | fp16 GPT-AR（混合精度） | GPT 430→245ms (1.77x) | greedy 78/78 精确 |
| R3 | fp16 BigVGAN | 89→58ms (1.53x) | cosine 0.9998 |
| R4 | CFM 欧拉步 25→15 | S2Mel 251→177ms | cosine 0.998 |
| R5 | 紧凑 KV buffer（max_mel_tokens 1000→300） | GPT 184→124ms (1.48x) | 无截断 |
| R8 | ~~TeaCache 阈值 0.15→0.25~~（已回退；代码已在 v0.2.1 移除） | S2Mel 185→165ms | cosine 0.999 |
| R9 | CFM 步数 15→12 | S2Mel 168→137ms | cosine 0.9995 |
| R12 | **S2Mel CUDA Graph 修复并启用**（dt_buf GC + freqs_cis rebuild 两个根因 bug） | eager 488→graph 442ms | 0/75 板砖，21/21 对齐 |
| R13 | **fp16-native DiT**（移除 .float() 精度守卫） | S2Mel 433→400ms | cosine 0.9997 |
| R14 | **CUDA-Graph beam search**（静态 batch K=3，无 KV 重排） | eager beam3 1464→~490ms，e2e 1.4→0.65s | 9/10 seed 位级一致 |

### 可调参数（质量/速度权衡）

```python
tts = WIndexTTS(weights_dir=..., device="cuda", dtype=torch.float16,  # fp16 最快保真；默认 fp32
                enable_w4a16=True)  # INT4 GPT（更快，有损，听感可用；需 pip install 'windextts[quant]'）

tts.warmup()  # 预捕获 CUDA Graph，把 ~1s 冷启动成本移出首请求
tts.infer(ref, text, 'ZH',
    num_beams=3,          # 默认官方质量档；1+do_sample=True = 极速且自然；
                          # 1+do_sample=False = 纯 greedy（最快但韵律机械拖沓，慎用）
    cfm_steps=15,         # CFM 欧拉步（15=听感无损下限；12=极速，音节末尾呼吸处轻微变差；25=最高）
    cfg_rate=0.7,         # CFG 强度（0.3=快10%，0.0=最快无引导）
)
```

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

# Apple Silicon（MLX 后端，Metal 加速）：核心包同样包含 windextts_mlx 子包，
# 用 [mlx] extra 一条命令装好 MLX 运行时
pip install 'windextts[mlx]'
```

### CLI

```bash
# 基础（fp16）
windextts --ref voice.wav --text "你好世界" -o out.wav
# 最快（INT4 量化）
windextts --ref voice.wav --text "你好世界" -o out.wav --quantize   # 或旧别名 --w4a16
# 精度 / 束搜索控制（--fp32 / --greedy 为等价旧别名）
windextts --ref voice.wav --text "你好" -o out.wav --dtype fp16 --beams 1
# 情感向量 + 时长
windextts --ref voice.wav --text "你好" -o out.wav --emo-vector 0.8,0,0,0,0,0,0.2,0 --duration 1.1

# 仅 CUDA：低显存模式（保持 beam3 质量，稳态 ~2.9GB）
windextts --ref voice.wav --text "你好" -o out.wav --quantize --low-vram

### 后端自动选择与 WINDEXTTS_BACKEND

同一条 `windextts` 命令按平台自动选后端：**NVIDIA GPU → CUDA torch；Apple Silicon（arm64 且已装 mlx）→ MLX**。
检测不导入 torch/mlx（platform + find_spec），零开销。可用环境变量强制覆盖：

```bash
WINDEXTTS_BACKEND=mlx windextts ...    # 强制 MLX
WINDEXTTS_BACKEND=torch windextts ... # 强制 torch（Apple Silicon 上无 NVIDIA 卡会明确报“CUDA 不可用”）
WINDEXTTS_BACKEND=cuda windextts ...  # 同 torch
```

两侧参数一致（--text-file/-o/--model-dir/--duration/--top-p/--top-k/--temperature/
--no-normalize/--segment-tokens/--emo-vector|--emo-text|--emo-ref/--beams/--cfm-steps 等）；
仅 CUDA 支持的 `--low-vram` 在 MLX 后端会被忽略并告警，`--install-model` 则明确报错退出。
旧的独立入口仍然可用：

```bash
python -m windextts_mlx --ref voice.wav --text "你好世界" --quantize -o out.wav
# MLX 参数与统一入口一致：--dtype fp32|fp16、--quantize（W4A16）、--beams（1=greedy）、--cfm-steps、--lang、--model-dir（MLX 权重目录）
```
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

# Apple Silicon（纯 MLX，无需 torch/CUDA）
from windextts_mlx import WIndexTTSMLX

tts = WIndexTTSMLX(weights_dir="/path/to/IndexTTS-2.5-mlx",
                   dtype="fp16", quantize=True)  # quantize=True = W4A16
sr, wav = tts.infer("voice.wav", "你好世界", "ZH")
```
