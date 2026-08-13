The file is created and untracked (not staged — correct). Here is the acceptance report.

## Task result

Implemented `windextts/frontend/segmenter.py` — long-text segmentation for the GPT-AR stage, replicating official `infer_v2_5.py` `IndexTTS2.split_text_by_tokens` / `split_text_by_punctuation` exactly.

**API note**: The task referenced `WhisperTokenizer`, but WIndexTTS's tokenizer module exposes `build_tokenizer()` (returns a `tiktoken.Encoding`). The segmenter takes a generic `tokenizer: Callable[[str], list[int]]` so it's decoupled from the encoder — verified with `build_tokenizer()`.

**Implementation highlights**:
- `split_text_by_tokens(text, tokenizer, max_tokens, lang_prefix="", capacity=602)` — token-budget segmentation: protected atomic pieces → punctuation split (delimiter attached to preceding segment) → char-level force split for over-budget parts → greedy budget merge. `capacity` mirrors official `gpt.text_pos_embedding.emb.num_embeddings` (= max_text_tokens 600 + 2), overridable by callers.
- `split_text_by_punctuation(text, max_chars=40)` — low-VRAM char-based split.
- `SPLIT_PROTECTED_PATTERN` copied verbatim from `infer_v2_5.py` (`<|SPECIAL_TOKEN_\d+|>...<|SPECIAL_TOKEN_\d+|>`).

**Validation**: 32/32 A/B alignment cases PASS against a verbatim replica of the official `split_text_by_tokens` (8 texts × 4 budgets, with `<|zh|>` lang prefix, including protected-token, mixed-language, no-punctuation long-English, and force-split cases). Module `__main__` self-test passes.