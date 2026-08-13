The grep exit 1 confirms the module has **zero third-party imports at module level** (only stdlib: `re`, `os`, `tempfile`, `traceback`, `functools`; `tn` is lazily imported inside `load()`). Everything is verified.

## Acceptance Report

**完成：`windextts/frontend/normalizer.py`** — 官方 `TextNormalizer` 的完整移植，41 个官方用例逐字一致。

### 移植的方法（与官方 front.py 1:1 对应）
- `normalize()` — 主入口，按 `use_chinese` 分发 zh/en 路径，含发音标注保护 → TN → 人名/拼音/技术词恢复 → 标点映射
- `use_chinese()` / `match_email()` — 语言判断
- `load()` — 延迟加载 zh/en TN（用 `tn` 底层 API，非 wetext 包装层；cache 在 `/tmp/windextts_tn_cache`）
- `save_names/restore_names`、`save_tech_terms/restore_tech_terms`、`save_pinyin_tones/restore_pinyin_tones`、`correct_pinyin` — 保护机制
- `apply_glossary_terms` / `load_glossary` / `load_glossary_from_yaml` / `save_glossary_to_yaml` — 术语词典
- `_protect_pronunciation_annotations` / `_restore_pronunciation_annotations` — G2P 发音标注 `<字|读音>`
- 全部常量：`char_rep_map`、`zh_char_rep_map`、`PINYIN_TONE_PATTERN`、`NAME_PATTERN`、`TECH_TERM_PATTERN`、`ENGLISH_CONTRACTION_PATTERN`、`G2P_PRONUNCIATION_ANNOTATION_PATTERN`

### 省略
- `TextTokenizer` 类（SentencePiece 旧版机制，v2.5 用 tiktoken，任务明确不需要）

### 验证结果
- **41 个官方用例 vs 官方 `TextNormalizer`：0 差异**（数字/日期/百分比/拼音声调/技术术语/人名/英文缩写/长句）
- 边界功能：发音标注保护 ✓、glossary ✓、yaml 往返 ✓、use_chinese 邮箱/纯数字 ✓
- `py_compile` ✓、零 transformers/modelscope 依赖 ✓、tn cache 在 /tmp 不入仓库 ✓

### 风险
- `tn`（NeMo）在 Windows 上是否可用未验证（AGENTS.md 的 Windows 优先原则下，Windows 用户可能需要 wetext 路径；当前 `load()` 统一用 tn，pyproject 已列 wetext 作为备选）
- 首次调用 `load()` 构建 zh FST 约需几秒（一次性，缓存后秒级）