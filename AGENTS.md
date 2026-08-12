# WIndexTTS — 项目全局约束

> Windows-priority, zero-JIT-compile, pure-torch accelerated inference engine for
> IndexTTS-2.5. This file is loaded as global project context for every session.

## 项目定位

用纯 PyTorch 从零重写 IndexTTS-2.5 的神经网络推理，参照 vLLM-Omni 的加速思路，做一个
**Windows 优先、零 JIT 编译依赖、`pip install` 即用** 的加速实现。

差异化不是"比 vllm-omni 快"，而是：Windows 开箱即用、更彻底的 CUDA Graph、干净的纯 torch 代码。

## 铁律

### 纯 torch，零 JIT 编译依赖
- ✅ 允许：`torch` / `F.scaled_dot_product_attention` / `torch.cuda.CUDAGraph` 等原生 API
- ❌ 禁止：`torch.utils.cpp_extension.load`（需 MSVC+nvcc，Windows 痛点）
- ❌ 禁止：`@triton.jit`（官方 triton 不支持 Windows）
- ⚠️ 可选：`flash_attn` 库 / `torch.compile`——仅作可选加速，核心推理不开启时也必须能跑
- 判定标准：核心推理路径在**没有 C/C++/CUDA 编译器**的 Windows 上必须能运行

### 不依赖 index-tts / transformers / modelscope
- 自己写所有 `nn.Module`，不 `import indextts`
- 权重加载只用 `torch.load` + `safetensors`（0.5MB），不用 `transformers`（504MB 内存）
- w2v-bert 的 conformer encoder 自己手写
- 可阅读 `/root/index-tts/` 源码作为参考，但代码必须独立

### 接缝魔数不可臆造
重写的唯一正确性标准是与 index-tts 官方输出数值对齐。任何不确定的张量形状/数值，
**去 `/root/index-tts/indextts/` 查源码，不要猜**。关键约定见源码：
- `infer_v2_5.py`（主流程 + 魔数：w2v-bert 取 `hidden_states[17]`、S2Mel `1.72` 缩放、`n_quantizers=3`、CFM `25步`/`cfg_rate=0.7`）
- `config.yaml`（vocab/special token/层数维度等超参）

## 正确性验证

每个 `nn.Module` 必须能通过数值对齐测试：同一输入下，你的输出 vs index-tts 官方输出
`torch.allclose(atol=1e-4, rtol=1e-3)`。**不要等端到端听音频判断**——音频对齐是听感问题，
调试周期以天计。

## 默认测试物料
- 参考音频：`/root/WIndexTTS/test.wav`
- 模型权重：`/root/IndexTTS-2.5/`

参考环境（用于对齐与基准）：
- 官方 index-tts：`/root/index-tts/.venv`（Python 3.11）
- vLLM-Omni（性能上限参照）：`/root/vllm-omni/.venv`（Python 3.12）
- 模型权重：`/root/IndexTTS-2.5/`

## 非神经部分调最小成熟库，不自己造
- BPE 分词：`tiktoken`（不手写）
- 权重加载：`safetensors`（不手写二进制解析）
- 中文归一化：`jieba` + `cn2an` + `wetext`（语言学规则，torch 管不了）
- mel 特征：`torchaudio.compliance.kaldi.fbank`
- 音频 IO：`soundfile` + `torchaudio`

## 编码规范
- Python 3.10+，UTF-8，类型注解必填
- 注释英文为主（国际开源惯例），歧义处补中文
- 每个 `nn.Module` 附 `if __name__ == "__main__":` 的数值对齐测试
- CLI 和 Python API 优先；webui 直接复用 index-tts 官方的，不在本仓库重写

## VCS 约束
- 允许本地 commit，**禁止 push 到任何远端**
- 仓库内的 benchmark 脚本可以提交；**禁止提交**一次性临时测试脚本、草稿笔记等本地-only 文件
- 提交前用 `git status` 核对，确保没有混入临时产物

## 运行时输出
- **不要输出到 `/tmp/`**（系统盘仅 60GB，易写满）
- 临时输出（测试中间张量、临时音频、日志等）写到仓库外的数据盘路径，或用完即删，不要写进仓库目录，避免不经意间存留垃圾或被误提交