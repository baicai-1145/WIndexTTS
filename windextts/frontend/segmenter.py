"""Long-text segmentation for the GPT-AR stage.

Replicates official ``infer_v2_5.py`` ``IndexTTS2.split_text_by_tokens`` /
``split_text_by_punctuation`` (lines ~413-490) exactly.

The GPT-AR model has a fixed ``max_text_tokens`` positional-embedding capacity
(WIndexTTS default 600 → capacity 602). User text longer than the budget must
be split into segments at punctuation boundaries (keeping the delimiter
attached to the preceding segment), each within the token budget, then
synthesized separately and concatenated with silence gaps by the caller.
"""

from __future__ import annotations

import re
from typing import Callable

# Protect atomic pieces like <|SPECIAL_TOKEN_1|>...</|SPECIAL_TOKEN_2|> from
# being split. Copied verbatim from infer_v2_5.py (IndexTTS2.SPLIT_PROTECTED_PATTERN).
SPLIT_PROTECTED_PATTERN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")

# Split after these punctuation marks, keeping the delimiter on the preceding
# segment (lookbehind). Copied from official split_text_by_tokens / by_punctuation.
_PUNCT_SPLIT = re.compile(r"(?<=[，。！？、；：,\.!\?;:\n])")

# Default positional-embedding capacity: WIndexTTS gpt.max_text_tokens=600 -> +2.
_DEFAULT_CAPACITY = 602


def _token_len(text: str, tokenizer: Callable[[str], list[int]]) -> int:
    """Token count of ``text`` under the given encoder callable."""
    return len(tokenizer(text))


def _split_atomic_pieces(text: str) -> list[tuple[str, bool]]:
    """Split into (chunk, is_atomic) pieces; atomic pieces are never re-split.

    Mirrors official ``_split_atomic_pieces``.
    """
    pieces: list[tuple[str, bool]] = []
    pos = 0
    for m in SPLIT_PROTECTED_PATTERN.finditer(text):
        if m.start() > pos:
            pieces.append((text[pos : m.start()], False))
        pieces.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def split_text_by_tokens(
    text: str,
    tokenizer: Callable[[str], list[int]],
    max_tokens: int,
    lang_prefix: str = "",
    capacity: int = _DEFAULT_CAPACITY,
) -> list[str]:
    """Split ``text`` into segments each within ``max_tokens`` tokens.

    Mirrors official ``IndexTTS2.split_text_by_tokens``:

    1. ``budget = min(max_tokens, capacity - 2) - len(tokens(lang_prefix))``
    2. If the whole text fits, return ``[text]``.
    3. Split into atomic-protected pieces, then punctuation parts, then by
       character for any part still over budget.
    4. Greedily merge chunks into segments that fit the budget.

    ``tokenizer`` is any callable ``str -> list[int]``; pass e.g.
    ``lambda s: build_tokenizer().encode(s, allowed_special="all")``.
    """
    budget = min(max_tokens, capacity - 2) - _token_len(lang_prefix, tokenizer)
    budget = max(1, budget)
    if _token_len(text, tokenizer) <= budget:
        return [text]

    chunks: list[str] = []
    for piece, atomic in _split_atomic_pieces(text):
        if atomic:
            chunks.append(piece)
            continue
        for part in _PUNCT_SPLIT.split(piece):
            if not part:
                continue
            if _token_len(part, tokenizer) <= budget:
                chunks.append(part)
                continue
            current = ""
            for ch in part:
                if current and _token_len(current + ch, tokenizer) > budget:
                    chunks.append(current)
                    current = ch
                else:
                    current += ch
            if current:
                chunks.append(current)

    segments: list[str] = []
    current = ""
    for chunk in chunks:
        if current and _token_len(current + chunk, tokenizer) > budget:
            segments.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        segments.append(current)
    return segments or [text]


def split_text_by_punctuation(text: str, max_chars: int = 40) -> list[str]:
    """Split ``text`` into segments of at most ``max_chars`` characters.

    Mirrors official ``IndexTTS2.split_text_by_punctuation`` (low-VRAM mode):
    break at punctuation boundaries, keeping the delimiter attached to the
    preceding segment. A punctuation-free over-long part is kept as-is to avoid
    mid-word splits.
    """
    parts = _PUNCT_SPLIT.split(text)
    segments: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            if current:
                segments.append(current)
            current = part
    if current:
        segments.append(current)
    return segments


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from windextts.frontend.tokenizer import build_tokenizer

    _tok = build_tokenizer()
    enc = lambda s: _tok.encode(s, allowed_special="all")  # noqa: E731

    long = "这是第一句话。这是第二句话，它比较长用来测试分段。第三句！最后一句。"
    print("== split_text_by_tokens (max_tokens=20) ==")
    for seg in split_text_by_tokens(long, enc, max_tokens=20):
        print(f"  {seg!r} ({len(enc(seg))} tok)")

    print("== split_text_by_punctuation (max_chars=12) ==")
    for seg in split_text_by_punctuation(long, max_chars=12):
        print(f"  {seg!r}")

    # over-budget single sentence: must be force-split by character
    very_long = "人工智能正在改变世界的方方面面" * 10
    segs = split_text_by_tokens(very_long, enc, max_tokens=8)
    print("== over-budget force-split ==")
    print(f"  {len(segs)} segments, all <= 8 tok: "
          f"{all(len(enc(s)) <= 8 for s in segs)}")
