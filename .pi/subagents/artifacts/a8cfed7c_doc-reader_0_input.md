# Task for doc-reader

精确梳理 IndexTTS-2.5 官方推理的完整张量数据流，目标是让另一个团队能照着重写。所有结论必须带文件:行号，所有 shape 必须从源码或实测确认（不要猜）。

只读分析，不要改源码。如果某处 shape 源码看不出来，可以写探测脚本在 /root/index-tts/.venv 里实际加载模型、喂 /root/WIndexTTS/test.wav 跑一遍 print 中间张量 shape（脚本写到 /root/WIndexTTS/scripts/probe_*.py，跑完报告即可）。

重点文件：
- /root/index-tts/indextts/infer_v2_5.py（主流程，重点 infer()）
- /root/index-tts/indextts/infer_v2.py, infer.py（基类，G2P/前端/encode_speaker）
- /root/index-tts/indextts/gpt/model_v2.py（GPT-AR forward + inference）
- /root/index-tts/indextts/codec/models.py（EnhancedCodec.quantize/decode/vq2emb）
- /root/index-tts/indextts/s2mel/（CFM、length_regulator、estimator/DiT、style_encoder）

必须回答的 8 个问题（每题给 文件:行号 + shape/dtype/值域）：

1. 前端：text→text_tokens 完整步骤。G2P 用什么？BPE 词表是 multilingual_zh_ja_yue_char_del.tiktoken 吗？text_tokens dtype/shape/值域。

2. w2v-bert 特征（get_emb, infer_v2_5.py:282-289）：SeamlessM4TFeatureExtractor 对 16kHz wav 做了什么预处理？输出 input_features shape（[B,T,80] 还是 [B,T]？）必须实测。attention_mask 怎么算。hidden_states[17] shape=[B,T_w2v,1024]，T_w2v 与采样点数关系。

3. EnhancedCodec.quantize（infer_v2_5.py:294）：输入是归一化的 w2v-bert feat [B,T,1024] 吗？输出 semantic_code shape/dtype/值域(0~8191?)、feat（量化后）shape。

4. GPT-AR 输入接缝（最关键）：text_emb、mel condition、lang emb 各是什么。v2.5 speaker 条件到底是什么——scout 报告说=归一化 hidden_states[17]（1024维不量化），但 gpt.pth 里 spk_emb_proj.weight 是 [1280,192]，说明 spk 条件输入是 192 维！这个 192 vs 1024 矛盾必须查清：spk 条件到底是 CAMPPlus 192维 emb，还是 w2v-bert 1024维，两者怎么配合？CAMPPlus 192维 emb 怎么算（对 16kHz wav？对 mel？输入输出 shape）。

5. GPT-AR 输出：infer 路径用 HF generate 还是自定义？输出 mel codes shape [B,T_mel]？值域(0~8193 含 stop 8193?)。fill_codes_to_level / remove_long_silence 对 codes 做了什么。

6. EnhancedCodec.decode（infer_v2_5.py:851）：输入 codes shape，输出 S_infer shape（[B,C,T]? C=?）。

7. S2Mel-CFM（infer_v2_5.py:849-855 附近）：length_regulator(n_quantizers=3) 输入/输出 shape。DiT estimator 输入(content code、style、t、x0/mel)各 shape。1.72*duration_factor 用在哪步。CFM 25步 Euler 循环结构(cfg_rate=0.7 怎么用)。style_encoder 输入输出(CAMPPlus 192维→style?)。

8. BigVGAN：输入 mel [B,80,T_mel]，输出 audio [B,1,T_audio]，T_audio=T_mel*256?

返回格式：按 8 题分节，每节含 文件:行号 + 关键代码片段(2-5行) + 张量接缝表(变量名|shape|dtype|值域) + 魔数。
特别强调第 4 题 spk 条件 192/1024 矛盾——这是 GPT-AR 重写核心歧义，必须查到 ground truth，源码看不出来就实际跑一遍 print GPT.inference 收到的每个输入 shape。

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return a concise result and residual risks when applicable

Required evidence: manual-notes, residual-risks

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