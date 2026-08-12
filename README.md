# WIndexTTS

Windows-priority, zero-JIT-compile, pure-torch accelerated inference engine for
**IndexTTS-2.5**.

目标：在 **Windows 上 `pip install` 即用**（无 MSVC / nvcc / triton 编译依赖）的前提下，
用纯 PyTorch 重写 IndexTTS-2.5 的神经网络推理，并实现比官方 `use_accel` 更彻底的 CUDA Graph 加速。

## 状态

阶段1 进行中：权重加载骨架 + 逐模块 `nn.Module` 实现 + 数值对齐测试。

## 设计约束

见 [AGENTS.md](./AGENTS.md)。核心铁律：

- **纯 torch，零 JIT 编译**：核心推理路径在无 C/C++/CUDA 编译器的 Windows 上必须能跑。
- **不依赖 index-tts / transformers / modelscope**：所有 `nn.Module` 自己写，权重只用 `torch.load` + `safetensors`。
- **接缝魔数不可臆造**：唯一正确性标准是与官方输出 `torch.allclose(atol=1e-4, rtol=1e-3)`。
