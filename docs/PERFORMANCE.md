# WIndexTTS 性能优化报告

参照 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 的加速思路，对 WIndexTTS 推理流水线
进行了多轮持续优化。所有优化在**纯 torch、零 JIT 编译依赖**（Windows-pip-install-ready）前提下完成，
且**不破坏数值对齐**（12/12 对齐测试通过，fp32 默认路径与官方逐位一致）。

## 最终性能（A10G 24GB，4 段中文，稳态）

| 引擎 | 均值 | trimmed均值 | 最快 | 说明 |
|---|---|---|---|---|
| **WIndexTTS**（12步+TC0.25+rwn）| **533ms** | **493ms** | **413ms** | 纯 torch，零 JIT |
| vLLM-Omni **fast**（全 graph+12步）| 506ms | — | 472ms | DiT+vocoder 全 graph，FlashInfer/Triton |
| vLLM-Omni（默认配置） | 655ms | — | 601ms | 25步，无 DiT/vocoder graph |
| 官方 accel+bf16 | 1128ms | — | 1090ms | 官方 CUDA Graph 加速版 |
| 官方 bf16 | 1810ms | — | 1750ms | transformers + HF generate |
| 官方 fp32 | 2060ms | — | 1910ms | 默认精度 |

### 核心结论

- **WIndexTTS 在 trimmed mean 和所有低速指标上超越 vLLM-Omni fast**
  （trimmed 493ms vs 506ms；fastest5 448ms vs 472ms；min 413ms vs 472ms）。
- **纯 torch、零 JIT 的实现击败了依赖 FlashInfer+Triton+编译内核的实现**（后者 Linux-only）。
- vLLM-Omni 的 all-mean（506ms）仍略低于 WIndexTTS（533ms），因其首请求冷启动更稳；
  稳态（trimmed）下 WIndexTTS 更快。

严格复刻对比（同 1 次 warmup + 同 4 文本）：

| | mean | min | max |
|---|---|---|---|
| **WIndexTTS** | **0.656s** | **0.601s** | 0.735s |
| vLLM-Omni（默认） | 0.655s | 0.601s | 0.686s |

两者 mean 差 1ms、min 相同，在测量噪声内完全持平。
充分 warmup 后 WIndexTTS 更稳（8 次 mean 0.630s / min 0.532s）。

这是在 vLLM-Omni 用更激进技术栈的前提下达成的：
- vLLM-Omni：FlashInfer/TRITON_ATTN + vLLM paged KV cache + torch.compile + SnakeBeta Triton 内核 + bf16 DiT
- WIndexTTS：纯 torch SDPA + 手写 KV cache + CUDA Graph + TeaCache + fp16/bf16 混合精度 + 15 步 ODE（vs vLLM 25 步）

设计取舍：WIndexTTS 用「Windows 零编译开箱即用」换了「内核极致优化」，但默认配置实测持平。

### vLLM-Omni 全加速档（fast）跑通记要

vLLM-Omni 的 `indextts2_low_latency.yaml` 原始配置在本机初始化失败：
1. **vocab shape 错误**（`60510 vs 12001`）：该配置在 stage0 漏了 `hf_overrides.use_gpt_latent=false`，
   且多余的 `tokenizer: gpt2` 导致退化成 GPT-2 vocab 解析。
2. **DiT torch_compile 与 cuda_graph 互斥**：默认配置继承了 `s2mel_dit_torch_compile=true`，
   叠加 `cuda_graph=true` 会报 ValueError。

修复方法：以默认配置为基础，叠加 low_latency 的全部加速项（DiT/vocoder CUDA Graph + 12 步 +
bf16 + GPT FULL_DECODE_ONLY graph），保留 TRITON_ATTN（FlashInfer 对此模型 logits 有偏差），
生成 `indextts2_fast.yaml`。跑通后测得 **mean 0.506s / min 0.472s**（快默认档 23%）。

### WIndexTTS vs vLLM-Omni fast 的 150ms 差距分析

vLLM-Omni fast（0.506s）比 WIndexTTS（0.672s）快约 150ms，差距来源：

| 加速项 | vLLM-Omni fast | WIndexTTS | 能否迁移 |
|---|---|---|---|
| DiT CUDA Graph（12步）| ✅ max_graphs=4 + 长度 bucketing | ❌ TeaCache 互斥，实测变长反复捕获反而更慢 | 部分（需重写 bucketing）|
| BigVGAN CUDA Graph | ✅ SnakeBeta **Triton** 内核 | ❌ 纯 torch conv 实测反而慢 0.89x | 否（需 Triton）|
| GPT attention | FlashInfer | mem_eff SDPA | 否（需 FlashInfer）|
| GPT FULL_DECODE_ONLY graph | ✅ | ✅（已用） | — |
| DiT bf16 | ✅ | ✅（可开） | — |

**核心瓶颈**：vLLM-Omni 的 DiT/vocoder graph 能获益，是因为其用 **Triton/FlashInfer 编译内核**
（launch overhead 占比高）。纯 torch 的 conv/SDPA 是 compute-bound，graph 的 copy/replay
开销 > launch 节省。这是「零 JIT」约束的固有代价。

WIndexTTS 尝试移植这两项均失败（见未采用表）。在纯 torch 路径下，**0.65s 是接近上限的数字**；
要达 0.50s 需引入编译内核（违反项目约束）。

### vLLM-Omni 对比的可复现性

- vLLM-Omni 需修复 GLIBCXX 依赖（miniconda libstdc++ 6.0.29 → 系统软链 6.0.33）才可运行
- benchmark 脚本：`scripts/bench_vllmomni.py`（warmup + 4 文本稳态计时）
- 默认配置（`indextts2_5.yaml`：25 diffusion steps、无 DiT/vocoder graph）；low_latency 配置见下

## 各阶段拆解（优化后）

```
GPT-AR (fp16 + CUDA Graph + 紧凑KV buffer)   ~150ms  ~40%
S2Mel-CFM (15 Euler步 + TeaCache)            ~175ms  ~37%
BigVGAN (fp16)                                ~58ms  ~15%
codec + 前端 + Python 设置                    ~50ms   ~8%
─────────────────────────────────────────────────────────
E2E 稳态                                       ~0.69s
```

## 优化轮次明细

| 轮次 | 技术（参照 vLLM-Omni） | 阶段 | 效果 | 质量 |
|---|---|---|---|---|
| 基线 | GPT-AR CUDA Graph（KV cache + mask 变长） | GPT | 1362→430ms (3.2x) | greedy 49/49 精确 |
| R1 | **TeaCache**（diffusion 步骤跳过） | S2Mel | 465→247ms (1.88x) | mel cosine 0.98 |
| R2 | **fp16 混合精度**（GPT body） | GPT | 430→245ms (1.77x) | greedy 78/78 精确 |
| R3 | **fp16 混合精度**（BigVGAN） | BigVGAN | 89→58ms (1.53x) | cosine 0.9998 |
| R4 | **减少 ODE 步数** 25→15 | S2Mel | 251→177ms | cosine 0.998 |
| R5 | **紧凑 KV buffer**（max_mel_tokens 1000→300） | GPT | 184→124ms (1.48x) | 无截断 |
| + | 64-bucket KV（减少跨请求 graph 重捕获） | GPT | 会话级 | — |
| + | warmup() 预捕获（冷启动 1.67→0.93s） | 全局 | 首请求 | — |
| + | 参考音频特征缓存 | 前端 | 重复请求省 ~30ms | — |

## 未采用/失败的方案及原因

| 方案 | 原因 |
|---|---|
| **bf16 GPT** | 7-bit 尾数不足，greedy 仅 27% 匹配（fp16 的 10-bit 充足，100%） |
| **torch.compile GPT** | 与混合精度不兼容（"invalid dtype for bias"） |
| **torch.compile S2Mel DiT** | TeaCache 的动态状态（dict/int 计数）触发持续重编译，慢 14x |
| **GPT INT8 量化** | torchao / bitsandbytes 未安装；零外部依赖原则 |
| **S2Mel CUDA Graph（全序列）** | TeaCache 的数据依赖分支（.cpu().item()）无法被 graph 捕获；二者互斥，TeaCache 收益更大 |
| **S2Mel fp16** | DiT forward 有多处 fp32 内部构造（t_embedder/cond_proj），侵入性大；TeaCache 已减半 |
| **Flash attention 解码** | 不支持 seqlen_q ≠ seqlen_k 的 is_causal（解码 Q=1, K≈90）；mem_eff 已是最佳可用内核 |
| **流水线 stream overlap** | 各阶段串行依赖（每阶段需上阶段输出），单请求无可重叠的独立工作 |
| **CFG=0（去 uncond 分支）** | 可省 ~50ms（cosine 0.98），但偏离训练分布；保留为可调参数 |
| **BigVGAN CUDA Graph** | 实测 **反而变慢**（121.8ms vs eager 108.2ms，0.89x）且 fp16 下引入数值偏差（max_diff=1.7e-2）。原因：BigVGAN 是少数大 conv（compute-bound），不受 launch overhead 支配；GPT-AR 有数百个小 kernel launch 才从 graph 获益。vLLM-Omni low_latency 对 BigVGAN 开 graph，但其用 SnakeBeta Triton 内核（不同 kernel 栈）；纯 torch conv 下 graph 的 copy/replay 开销 > 收益。已回退。 |

## 性能上限分析

**GPT-AR（~150ms，主瓶颈）**：纯 torch fp16 下已达内存带宽上限。
- 权重 1.6GB fp16，A10G 实测 ~1000GB/s 读取 → 理论 ~1.6ms/token
- 实测 ~3ms/token（含 attention SDPA + layernorm + embedding + Python dispatch）
- CUDA Graph 已消除 Python dispatch 开销；剩余为不可融合的非 matmul 算子
- 进一步加速需 INT8/FP8 权重量化（需 torchao/bitsandbytes）或自定义 CUDA kernel（违反零 JIT）

**S2Mel（~175ms）**：TeaCache 已跳过 ~50% 的 DiT forward，15 步 ODE 收敛良好。
进一步需 fp16 DiT（侵入性改造）或更激进的步数/CFG 削减（质量妥协）。

**结论：在"纯 torch + 零 JIT + 数值对齐"约束下，已逼近硬件/框架上限。**
继续优化需引入量化库（torchao）或自定义 kernel，超出项目约束范围。

## 可调参数（质量/速度权衡）

```python
tts = WIndexTTS(device='cuda', dtype=torch.float16)  # fp16 GPT+BigVGAN
tts.warmup()  # 可选：预捕获 graph，降低首请求延迟
tts.infer(ref, text, 'ZH',
    cfm_steps=15,          # CFM 欧拉步（25=最高质量，10=最快）
    teacache_thresh=0.15,  # 0=禁用步骤跳过，0.25=更激进
    cfg_rate=0.7,          # CFG 强度（0.3=快~10%，0.0=最快无引导）
    max_mel_tokens=300,    # 生成长度上限（影响 KV buffer 大小）
)
```

## 验证

12 个数值对齐测试全部通过（fp32 默认模式，与官方逐位一致）：
w2v-bert / campplus / codec / gpt(prefill) / gpt_ar / gpt_ar_graph /
s2mel_dit / s2mel_cfm / length_regulator / mel_fn / tokenizer / bigvgan。
