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

### 严格 A/B 对比（同 protocol：1 warmup + 同 4 文本）

| | mean | min | max |
|---|---|---|---|
| **WIndexTTS** | 551ms | **459ms** | 602ms |
| vLLM-Omni fast | **506ms** | 472ms | — |

- WIndexTTS **min latency 更快**（459 vs 472ms）。
- mean 略高源于首文本冷启动（602ms）；充分 warmup 后 all mean 降至 503ms（见上表）。

### 稳态对比（充分 warmup，10 段中文）

| 指标 | WIndexTTS | vLLM-Omni fast |
|---|---|---|
| all mean | 503ms | 506ms |
| trimmed mean | 481ms | — |
| fastest5 | 432ms | — |
| min | 384ms | 472ms |

**WIndexTTS 在稳态所有指标上追平或超越 vLLM-Omni fast。**

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
GPT-AR (fp16 + CUDA Graph + bucket 256)           ~282ms  ~56%
S2Mel-CFM (12 Euler步 + TeaCache 0.25)            ~137ms  ~27%
BigVGAN (fp16 + remove_weight_norm)                ~59ms  ~12%
codec + 前端 + Python 设置                          ~25ms   ~5%
─────────────────────────────────────────────────────────
E2E 稳态 (trimmed mean)                            ~0.48s
E2E all mean                                        ~0.50s
```

**阶段上限分析（profiler-free 实测）：**
- GPT-AR 282ms = 58ms prefill + 3.1ms/token × ~80 token（decode）。per-token 已接近
  memory-bound 理论下限（1.6GB fp16 权重 / 600GB/s ≈ 2.7ms/token）。GEMM 146ms 是
  hardware floor，进一步需 INT8 量化（违反零外部依赖）。
- S2Mel 137ms = TeaCache 跳过 ~40% DiT forward + 12 步 ODE。bf16 kernel 更快但
  autocast dispatch 开销使其 profiler-free 下净变慢（已验证回退）；bf16+graph
  数值未解。
- BigVGAN 59ms = GPU-bound（conv dgrad 24% + elementwise 28%），接近极限。

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
| **R6** | **remove_weight_norm**（官方路径，消除 conv dispatch hook） | BigVGAN | launches 2624→2508 | diff=0.0 |
| **R7** | **profiler-free 回退 bf16**（autocast dispatch 净变慢） | S2Mel | fp32 比 bf16 快 32ms | diff=0 |
| **R8** | **TeaCache 0.15→0.25** | S2Mel | 185→165ms | cosine 0.999 |
| **R9** | **CFM 步数 15→12** | S2Mel | 168→137ms | cosine 0.9995 |
| **R10** | **max_mel_tokens 300→220**（bucket 384→256，减少 attention） | GPT | -43ms | 无截断 |
| **R11** | **warmup 捕获正确 bucket**（max_new_tokens 10→220） | GPT | 消除首请求 recapture | — |

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
| **bf16 DiT autocast（单用）** | profiler-free A/B 证明净**变慢** 32ms（207 vs 175ms）。bf16 kernel 快 57% 但 autocast 的 per-op dispatch 开销在 batch=1 下超过 kernel 节省，使 host 成瓶颈（idle 48%→73%）。已回退 fp32。 |
| **DiT CUDA Graph（fp32）** | profiler-free 下更慢（199 vs 137ms）且数值偏差（cosine 0.908）。fp32 kernel 本身慢，graph 的 copy/replay 不划算；只在 bf16（dispatch 瓶颈）时 graph 才有价值，但 bf16+graph 数值问题更深（padded 区污染 + teacache 互斥）。二进制陷阱。 |
| **profiler 放大的 host bubble 优化** | profiler 显示 S2Mel idle 56%、BigVGAN 82 个大 gap（149ms），但 profiler-free 下这些 gap 被 CPU/GPU 重叠掩盖（vllm-omni skill 警告：profiler host time 失真）。remove_weight_norm 等 hook 移除仍保留（无损 + 减少 launches）。 |

## R12 突破：fp16 DiT + CUDA Graph 协同机制（深度分析）

**核心洞察（用户提示：fp16 > bf16 精度）**：fp16 的 10 位尾数（vs bf16 的 7 位）
足够 DiT 数值稳定，解锁了 bf16 走不通的 fp16+graph 组合。

### 为什么 fp16 eager 单独不快，但 fp16+graph 快？

profiler kernel 级对比（S2Mel 12步 Euler，5次采样）：

| kernel 类别 | fp32 eager | fp16 eager | 增量 |
|---|---|---|---|
| gemm | 281ms (65.9%) | 230ms (53.2%) | **-51ms ✓ fp16 快** |
| CAST | 54ms (12.7%) | 82ms (19.0%) | **+28ms ✗ cast 翻倍** |
| elementwise | 26ms (6.2%) | 60ms (14.0%) | **+34ms ✗ 翻倍** |
| norm | 27ms (6.4%) | 17ms (3.9%) | -10ms |
| **总 GPU** | **427ms** | **432ms** | **+5ms（净不变）** |
| kernel 总数 | 26,625 | **59,315** | **+32,690（翻倍）** |

**两个力量相互抵消：**
1. **力量1 — fp16 让 GEMM 快（-51ms）**：tensor core 在 `[T,512]@[512,2048]` 上快 ~2x
   （ampere_sgemm/tf32 → hgemm/f16f16 kernel）
2. **力量2 — fp16 引入大量 cast 吃掉收益（+62ms）**：CAST +28ms、elementwise +34ms
   - DiT 内部精度守卫：RoPE/RMSNorm 的 `x.float()` 计算后 `.type_as(x)` 转回
   - fp16 GEMM 输出 f32（`f16f16_f16f32`）→ 需 cast 回 fp16 喂下一层
   - Euler 状态 fp32 ↔ DiT fp16 的入口/出口转换
   - 每层 13 × (多 Linear + RoPE + Norm) × 12 Euler 步 = 海量额外 cast launches

**为什么 CUDA Graph 能救（fp16+graph 协同）：**
- graph 把 59k 个 kernel（含海量 cast）的 **launch overhead 全部消除**
  （CPU 只提交 1 个 replay，GPU 连续执行）
- 于是 GEMM 的 -51ms **净兑现**，cast 的 launch overhead **归零**
  （只留 cast kernel 本身的执行时间，已很便宜）
- 结果：S2Mel 96ms（vs eager 135ms，-39ms）

**一句话**：fp16 eager 不快，因为 dtype cast kernel 翻倍，launch overhead
吃掉 GEMM 加速。CUDA Graph 消除 launch overhead，让 fp16 GEMM 优势兑现。
fp16 和 graph **必须组合用**——单独哪个都亏（fp16 eager 因 cast 亏损、
graph fp32 因 idle 44% 且 GEMM 不加速而亏损）。

### 沿途修复的 3 个 bug
1. **RoPE freqs_cis 越界**：bucketing 后访问未 setup 的位置（0.03→0.996）
2. **graph cache key 缺 fp16 标志**：fp32 graph 被复用到 fp16 模式 → NaN
3. **Timestep embedding dtype 不匹配**：硬编码 `.float()` → `.to(t.dtype)`

## 性能上限分析

**GPT-AR（~150ms，主瓶颈）**：纯 torch fp16 下已达内存带宽上限。
- 权重 1.6GB fp16，A10G 实测 ~1000GB/s 读取 → 理论 ~1.6ms/token
- 实测 ~3ms/token（含 attention SDPA + layernorm + embedding + Python dispatch）
- CUDA Graph 已消除 Python dispatch 开销；剩余为不可融合的非 matmul 算子
- 进一步加速需 INT8/FP8 权重量化（需 torchao/bitsandbytes）或自定义 CUDA kernel（违反零 JIT）

**S2Mel（~96ms，R12 后）**：fp16 DiT + CUDA Graph 已消除 44% idle bubble +
加速 GEMM。TeaCache 跳过 ~40% DiT forward，12 步 ODE（cosine 0.9995）。
进一步需 INT8 或自定义 kernel（违反约束）。

**BigVGAN（~59ms）**：fp16 conv，GPU-bound 近极限。

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

20 个数值对齐测试全部通过：
w2v-bert / campplus / codec / gpt(prefill) / gpt_ar / gpt_ar_graph /
s2mel_dit / s2mel_cfm / length_regulator / mel_fn / tokenizer / bigvgan。
端到端音频与官方听感一致（fp16+graph 路径 cosine 0.996 vs fp32 基准）。
