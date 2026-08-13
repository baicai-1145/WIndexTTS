# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
目标: 为 WIndexTTS 实现长文本分段模块 windextts/frontend/segmenter.py。

## 背景
WIndexTTS 的 GPT-AR 有 max_text_tokens 上限(默认600)。用户输入长文本(如一篇文章)必须按标点+token预算切成多段, 分别合成后拼接(段间加静音)。

## 参考 (只读)
- 官方分段逻辑: /root/index-tts/indextts/infer_v2_5.py 的 split_text_by_tokens 方法 (约413-464行) 和 _split_atomic_pieces
- 官方 split_text_by_punctuation (约467-490行): 低显存模式的按字符数切分
- WIndexTTS 的 tokenizer: /root/WIndexTTS/windextts/frontend/tokenizer.py (有 encode 方法)

## 要实现
1. split_text_by_tokens(text, tokenizer, max_tokens, lang_prefix='') -> list[str]
   - 完全对齐官方 split_text_by_tokens 逻辑
   - 按标点 [，。！？、；：,.!?;:\n] 切分
   - SPLIT_PROTECTED_PATTERN 保护某些片段(从官方复制这个正则)
   - 用 tokenizer.encode(text) 计算每段 token 数, 确保不超 budget
   - 超长段递归按字符切, 再按 budget 合并

2. split_text_by_punctuation(text, max_chars=40) -> list[str]
   - 官方低显存模式用的, 按标点切, 每段最多 max_chars 字符

## 关键: SPLIT_PROTECTED_PATTERN
从官方 infer_v2_5.py 复制这个正则常量(它保护 <字|读音> 这类标注不被切段)。搜索 SPLIT_PROTECTED_PATTERN 找到它。

## 验证 (必须跑)
```python
import sys; sys.path.insert(0,'/root/WIndexTTS')
from windextts.frontend.tokenizer import WhisperTokenizer
tok = WhisperTokenizer('/root/IndexTTS-2.5/multilingual_zh_ja_yue_char_del.tiktoken')
from windextts.frontend.segmenter import split_text_by_tokens
long = '这是第一句话。这是第二句话，它比较长用来测试分段。第三句！最后一句。'
for seg in split_text_by_tokens(long, tok, max_tokens=20): print(repr(seg))
```
用 /root/index-tts/.venv/bin/python 运行。

## 返回
报告: 文件路径、实现的方法、SPLIT_PROTECTED_PATTERN 来源、测试结果、风险。

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