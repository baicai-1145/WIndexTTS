# Task for scout

目标：为"用纯 torch 重写 IndexTTS-2.5 推理"项目，摸清官方推理的完整数据流和每个神经模块的精确输入/输出接缝（tensor shape / dtype / device / 取哪一层），用于后续逐模块数值对齐。

只读分析，不要改任何文件。重点源码目录：
- /root/index-tts/indextts/infer_v2_5.py  （主流程，最关键）
- /root/index-tts/indextts/infer_v2.py, infer.py（基类，提供 _load_model 等基础设施）
- /root/index-tts/indextts/gpt/model_v2.py（GPT-AR）
- /root/index-tts/indextts/codec/models.py（EnhancedCodec）
- /root/index-tts/indextts/s2mel/modules/（CFM + DiT + length_regulator + campplus）
- /root/index-tts/indextts/BigVGAN/（vocoder）

请产出一份精确的"接缝清单"，按推理 6 阶段顺序组织：
1. 前端：文本→token 的完整路径（G2P/归一化/BPE），输入输出 dtype/shape
2. w2v-bert 特征提取：原始 wav(16kHz) 进，hidden_states[17] 出——精确到：怎么调用的（transformers Wav2Vec2BertModel?）、输入要做的预处理（归一化用 stats）、输出 hidden_states[17] 的 shape
3. CAMPPlus：输入输出 shape（emb 192维？）
4. GPT-AR：输入(text_emb, mel_condition)，输出 mel codes 的 shape、dtype、范围(0~8193?)；inference 用的是 HF generate 还是自定义？关键 mask/condition 接缝
5. EnhancedCodec：输入 mel codes，输出什么？（codec codes? audio latent?）shape/dtype
6. S2Mel-CFM：输入到 DiT 的所有张量（content code、style、mel target shape）、CFM 25 步 Euler 的循环结构、length_regulator(n_quantizers=3) 输入输出
7. BigVGAN：输入 mel，输出 audio

对每个模块，标注：
- 官方类名 + 文件路径 + 关键行号
- 入口函数签名（forward/inference 的参数）
- 输入输出张量的 shape/dtype 约定
- 任何"魔数"（缩放系数、特殊 token、层数索引等）

同时回答：
- semantic_codec(hf_cache/semantic_codec/model.safetensors) 和 codec.pth 是什么关系？是两个不同模型还是同一个？
- w2v-bert 的 conformer_shaw.pt 怎么加载到 transformers Wav2Vec2BertModel？（config.json 内容）

返回格式：结构化 markdown，按阶段分节，每节含"文件:行号 + 接缝"。不要泛泛而谈，要精确到能照着写代码。

---
**Output:**
Write your findings to exactly this path: /root/WIndexTTS/.pi/subagents/artifacts/outputs/76758586/context.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

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