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

**已发布**：PyPI `windextts` v0.1.1 + Windows 整合包（解压即用、内置权重）。最快延迟追平并超过 vLLM-Omni fast。

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
| R1 | ~~TeaCache（S2Mel DiT 步骤跳过）~~（已回退，见下） | S2Mel 465→247ms (1.88x) | mel cosine 0.98 |
| R2 | fp16 GPT-AR（混合精度） | GPT 430→245ms (1.77x) | greedy 78/78 精确 |
| R3 | fp16 BigVGAN | 89→58ms (1.53x) | cosine 0.9998 |
| R4 | CFM 欧拉步 25→15 | S2Mel 251→177ms | cosine 0.998 |
| R5 | 紧凑 KV buffer（max_mel_tokens 1000→300） | GPT 184→124ms (1.48x) | 无截断 |
| R8 | ~~TeaCache 阈值 0.15→0.25~~（已回退） | S2Mel 185→165ms | cosine 0.999 |
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
