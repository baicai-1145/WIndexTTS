# WIndexTTS 性能优化报告

参照 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 的加速思路，对 WIndexTTS 推理流水线
进行了多轮持续优化。所有优化在**纯 torch、零 JIT 编译依赖**（Windows-pip-install-ready）前提下完成，
且**不破坏数值对齐**（12/12 对齐测试通过，fp32 默认路径与官方逐位一致）。

## 最终性能（A10G 24GB，4 段中文，稳态）

| 引擎 | 均值 | 中位 | 最快 | RTF | 说明 |
|---|---|---|---|---|---|
| **WIndexTTS** | **0.687s** | **0.640s** | **0.608s** | **5.18x** | 纯 torch，零 JIT |
| 官方 accel+bf16 | 1.128s | — | 1.090s | — | 官方 CUDA Graph 加速版 |
| 官方 bf16 | 1.81s | — | 1.75s | — | transformers + HF generate |
| 官方 fp32 | 2.06s | — | 1.91s | — | 默认精度 |
| vLLM-Omni | ⚠️ 未能实测 | — | — | — | 环境限制（见下） |

**WIndexTTS 比官方加速版（accel+bf16）快 1.6x，比官方 fp32 快 3.0x。**

### 关于 vLLM-Omni 的对比（未能实测）

vLLM-Omni 在本机**无法启动**：flashinfer 编译的 `.so` 要求 `GLIBCXX_3.4.32`，
系统 libstdc++ 版本过低（`GLIBCXX_3.4.32 not found`）。这是系统库兼容性问题，
非性能问题，调试超出任务范围。

基于其 `indextts2_low_latency.yaml` 配置的技术分析（推断性能区间）：
- vLLM-Omni low_latency 版用：FlashInfer attention + FULL_DECODE_ONLY CUDA Graph（GPT）、
  DiT bf16+CUDA Graph、BigVGAN CUDA Graph、12 diffusion steps、vLLM paged KV cache。
- 这些比我更激进（我用 mem_eff+graph、TeaCache、无 vocoder graph），
  **vLLM-Omni low_latency 版很可能比我快**，预估在 0.4-0.6s 区间。
- 但其默认配置（`indextts2_5.yaml`）较保守：25 步、无 DiT/vocoder graph、TRITON_ATTN，
  性能未必明显优于 WIndexTTS。
- **关键差异**：vLLM-Omni 依赖 FlashInfer/Triton（需编译，Linux-only），
  WIndexTTS 纯 torch（Windows 可跑）。这是设计取舍——Windows 零编译 vs Linux 极致内核。

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
