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

### 性能（A10G 24GB，统一协议实测：4 段长中文（成音 5.3-8.7s）、beam3 采样、warmup 后每档 12 次计时）

| 引擎 | 精度 | CFM 步数 | 解码 | 均值 | 最快 | vs 官方 fp32 |
|---|---|---|---|---|---|---|
| 官方 fp32 | fp32 | 25 | beam3 采样 | 3.66s | 3.07s | 1.0x |
| 官方 bf16 | bf16 | 25 | beam3 采样 | 4.03s | 3.39s | 0.91x（更慢） |
| WIndexTTS 保真档 | fp32 | 25 | beam3 采样 | 3.12s | 2.68s | 1.17x |
| WIndexTTS fp32 | fp32 | 15 | beam3 采样 | 2.77s | 2.40s | 1.32x |
| WIndexTTS fp16 | fp16 | 25 | beam3 采样 | 1.73s | 1.34s | 2.12x |
| **WIndexTTS 默认档** | **fp16** | **15** | beam3 采样 | **1.62s** | 1.23s | **2.26x** |
| WIndexTTS W4A16 | fp16 | 12 | beam1 采样 | 1.30s | 0.93s | 2.82x |
| WIndexTTS W4A16 极速 | fp16 | 12 | greedy | **1.01s** | **0.88s** | **3.62x** |
| vLLM-Omni（默认 deploy） | bf16 | 25 | beam1 采样 | 1.15s | 0.90s | 3.18x |

**零编译依赖下：默认档比官方 fp32 快 2.3x；W4A16 极速档最快 0.88s，全场最低延迟（vLLM-Omni 默认配置 0.90s）。**

> 协议说明：本轮为长文本统一协议（音频 5.3-8.7s），与早期短句协议（~4s 成音）数字不可直接比较。
> vLLM-Omni 为其默认 `indextts2_5.yaml` deploy 配置（beam1 采样 25 步 bf16+compile，DiT graph 关）；
> 早期记录的 0.51s 均值来自手动开启 graph 的调优配置，非默认。
> fp32@25 档（1.17x）为与官方逐位对齐的保真档：fp32 GEMM 计算密度是瓶颈（GPT-AR 984ms 占 64%），
> 加速手段只消调度开销；要速度请下台阶到 fp16（GEMM 2x）。
>
> ⚠️ **greedy（beam1+argmax）对效果影响较大**：无采样的确定性解码韵律机械拖沓、语速系统性偏慢（codes 显著多于
> beam3 采样档），听感劣化程度远超 fp16/W4A16/12步等有损加速项。极速档若在意质量，建议用 `num_beams=1,
> do_sample=True`（速度接近 greedy，保留采样自然度）；beam3+采样为官方质量档。
>
> **W4A16 为有损加速档**：INT4 舍入使 mel codes 与 fp16/官方从首个 token 起即不同（波形 cosine
> ~0.02，不满足数值对齐标准），但人工试听正常、自然度可用；追求与官方逐位一致请用 fp16/fp32。

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
    num_beams=3,          # 默认官方质量档；1+do_sample=True = 极速且自然；
                          # 1+do_sample=False = 纯 greedy（最快但韵律机械拖沓，慎用）
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
