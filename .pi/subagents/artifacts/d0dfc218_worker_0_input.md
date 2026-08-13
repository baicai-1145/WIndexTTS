# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
目标: 为 WIndexTTS 实现文本归一化模块 windextts/frontend/normalizer.py, 移植官方 /root/index-tts/indextts/utils/front.py 的 TextNormalizer。

## 背景
WIndexTTS 是 IndexTTS-2.5 的纯 torch 重写。当前 infer() 的 text 参数要求'已归一化', 但 webui 用户输入是任意中文(含数字/标点/英文), 必须先归一化。

## 参考 (只读, 不要修改)
- 官方 TextNormalizer: /root/index-tts/indextts/utils/front.py (756行)
- 官方 common.py 的 tokenize_by_CJK_char / de_tokenized_by_CJK_char: /root/index-tts/indextts/utils/common.py
- 参考 env: /root/index-tts/.venv (Python 3.11, 已装 jieba/cn2an/tn)

## 关键约束
- 不引入 transformers/modelscope (违反铁律)
- 用 tn (NeMo text processing) 做底层 TN, 不用 wetext 包装层 (当前环境 tn 已装, wetext 未装)。官方 Linux 用 tn, Mac/Win 用 wetext; 我们统一用 tn 底层 API:
    from tn.chinese.normalizer import Normalizer as NormalizerZh
    from tn.english.normalizer import Normalizer as NormalizerEn
    zh = NormalizerZh(cache_dir=<tmp>, remove_interjections=False, remove_erhua=False, overwrite_cache=False)
    en = NormalizerEn(overwrite_cache=False)
- cache_dir 用 tempfile.gettempdir()/windextts_tn_cache, 自动创建

## 要移植的核心方法 (front.py 里的 TextNormalizer 类)
- normalize(text): 主入口, 按语言分发 (use_chinese 判断), 保护人名/技术词/拼音声调/发音标注, 调 TN, 恢复
- use_chinese(s): 判断是中文还是英文 (用于选择 zh/en normalizer)
- save_names / restore_names: 保护人名不被 TN 破坏 (jieba 分词识别)
- save_tech_terms / restore_tech_terms: 保护技术术语 (如 GPT-5-nano)
- save_pinyin_tones / restore_pinyin_tones / correct_pinyin: 保护拼音声调 (如 xing2)
- apply_glossary_terms: 术语词典应用 (enable_glossary 可选, 先实现基础版)
- _protect_pronunciation_annotations / _restore_pronunciation_annotations: 保护 <字|读音> 标注
- load(): 延迟加载 zh/en normalizer

## 注意
- 不需要移植 SentencePiece 相关 (那是旧版用的, v2.5 用 tiktoken)
- 人名/技术词的正则和映射表 (name_pattern, tech_term_pattern, char_rep_map, zh_char_rep_map) 要从 front.py 完整复制过来
- normalizer 的全部常量和正则都在 __init__ 里定义, 要完整复制

## 验证 (必须跑)
写一个 if __name__=='__main__' 块, 测试以下输入:
```python
norm = TextNormalizer()
tests = ['我有123本书', '3.14%的利润', '2024年1月1日', '你好world', 'GPT-5的参数', 'IT行业薪资', '我叫张三']
for t in tests: print(f'{t!r} -> {norm.normalize(t)!r}')
```
期望: 数字转中文(一百二十三)、百分号转(百分之)、日期转(二零二四年一月一日)、技术词保留(GPT-5)、人名保留。

## 返回
报告: 创建的文件路径、移植了哪些方法、哪些省略、测试结果(输入→输出对照)、与官方的差异(如有)、风险。
用 /root/index-tts/.venv/bin/python 运行测试。

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