# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
实现 IndexTTS-2.5 GPT 的 emo_conditioning 子模块（ConformerEncoder + PerceiverResampler），用于情感参考音频路径。

## 背景
WIndexTTS 是 IndexTTS-2.5 的纯 torch 重写（见 /root/WIndexTTS/AGENTS.md）。当前 GPT 只支持 emo_vector（8维向量）路径，缺少 emo_ref_audio（从参考音频提取情感）路径。
需要实现官方 /root/index-tts/indextts/gpt/ 里的两个模块，加载 gpt.pth 中的 149 个 emo_conditioning 权重。

## 任务
创建文件 /root/WIndexTTS/windextts/models/emo_conditioning.py，包含两个 nn.Module：

### 1. EmoConformerEncoder（ConformerEncoder，与官方 indextts/gpt/conformer_encoder.py 对齐）
- input_size=1024, output_size=512, linear_units=1024, attention_heads=4, num_blocks=4, input_layer='conv2d2'
- conv2d2 embed: Conv2d(1,512,3,3,stride=2) + reshape + Linear(261632,512) + pos_enc（与官方一致，subsample_rate=2）
- 4 层 ConformerEncoderLayer（macaron FFN + RelPositionMultiHeadedAttention + ConvModule）
- after_norm (LayerNorm 512)
- forward(xs, xs_lens) → (out_seq [B,T',512], mask [B,1,T'])

### 2. EmoPerceiverEncoder（PerceiverResampler，与官方 indextts/gpt/perceiver.py 对齐）
- dim=1024, dim_context=512, num_latents=1, dim_head=64, heads=4, ff_mult=2, depth=2
- proj_context: Linear(512,1024)
- latents: Parameter [1,1024]
- 2 层 cross-attention（Attention with cross_attn_include_queries=True）+ FeedForward
- norm: RMSNorm(1024)
- forward(x [B,S,512], mask [B,1,S]) → latents [B,1,1024]

## 权重规格（已确认，来自 gpt.pth）
运行此命令查看完整权重 key 和 shape：
```bash
/root/index-tts/.venv/bin/python -c "import torch; sd=torch.load('/root/IndexTTS-2.5/gpt.pth',map_location='cpu',weights_only=False); [print(f'{k}: {tuple(sd[k].shape)}') for k in sorted(sd) if k.startswith('emo_conditioning_encoder') or k.startswith('emo_perceiver_encoder')]" 2>/dev/null
```

关键映射点：
- embed.conv.0 = Conv2d(1,512,3,3,stride=2); embed.out.0 = Linear(261632→512); embed.pos_enc.pe = [1,5000,512]
- encoders.{0-3} 每层 31 keys: self_attn(q/k/v/linear_out/linear_pos/pos_bias_u/pos_bias_v), feed_forward(w_1 512→1024, w_2 1024→512), feed_forward_macaron(同 feed_forward 命名?), conv_module(pointwise_conv1 512→1024, depthwise_conv 512×15, pointwise_conv2 1024→512, norm=BatchNorm1d), norm_ff/norm_mha/norm_conv/norm_final (LayerNorm 512)
- 注意：官方 macaron 有 feed_forward_macaron 吗？检查权重 key 数：每层 31 keys，如果只有一套 feed_forward 则无 macaron 的第二个 FFN。实际看 encoders.0.feed_forward 只有一套（w_1,w_2），所以**有 macaron 但权重复用？不对**——看 ff_scale。需要仔细读官方 ConformerEncoderLayer 源码 /root/index-tts/indextts/gpt/conformer_encoder.py line 170-240 确认。

## 参考源码（必读）
- /root/index-tts/indextts/gpt/conformer_encoder.py（ConformerEncoder + ConformerEncoderLayer + RelPositionMultiHeadedAttention + ConvolutionModule + PositionwiseFeedForward）
- /root/index-tts/indextts/gpt/perceiver.py（PerceiverResampler + Attention + FeedForward + RMSNorm）
- /root/index-tts/indextts/utils/common.py（make_pad_mask, RelPositionalEncoding）
- /root/WIndexTTS/windextts/models/w2v2_bert.py（我们已有的 conformer 实现，可参考但注意这里的 conformer 结构不同）

## 对齐测试
创建 /root/WIndexTTS/tests/align/test_emo_conditioning_align.py：
- 加载 gpt.pth 的 emo 权重到新模块
- 加载 /root/windextts_dumps/emo_ref/ 下的参考张量
- 测试：conformer_in [1,133,1024] + lens → conformer_seq [1,66,512]（allclose atol=1e-4, rtol=1e-3）
- 测试：conformer_seq + mask → perceiver_out [1,1,1024]（allclose）
用 /root/index-tts/.venv/bin/python 运行测试。

## 约束
- 纯 torch，零 JIT/cuda扩展/triton（AGENTS.md 铁律）
- 不 import transformers/modelscope/indextts（可读源码参考，但代码独立）
- 类型注解必填，注释英文为主
- ConformerEncoder 的 self-attention 用 F.scaled_dot_product_attention（不用 flash_attn 库），但要正确处理 RelPosition（相对位置编码）

## 完成标准
- 两个模块实现 + 数值对齐测试通过（vs 官方 dump）
- 用 /root/index-tts/.venv/bin/python tests/align/test_emo_conditioning_align.py 验证

先读官方源码理解结构，再实现。数值对齐是唯一正确性标准。

---
**Output:**
Write your findings to exactly this path: /root/WIndexTTS/.pi/subagents/artifacts/outputs/81c2590a/inline
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```