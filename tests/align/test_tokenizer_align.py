"""Numerical alignment test: pure-tiktoken tokenizer vs official WhisperTokenizer.

Validates that our ``windextts.frontend.tokenizer.build_tokenizer`` (pure tiktoken,
no whisper/transformers) produces bit-identical token ids to the official
``indextts.utils.tokenizer.get_tokenizer(multilingual=True)`` for the v2.5
inference path (``encode(text, allowed_special='all')``).

The dump ``frontend.tokens_zh.pt`` is the official output for
``'<|zh|> 欢迎大家来体验indextts2。'`` (14 tokens).
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"

# Text cases covering zh / en / mixed / punctuation. Each prefixed with the
# language tag exactly as v2.5 does (f"<|{lang}|> {text}").
CASES = [
    ("zh", "欢迎大家来体验indextts2。"),
    ("en", "Hello world, this is a test."),
    ("zh", "2002年的第一场雪，下在了2003年"),
    ("en", "See you at 8:00 AM"),
    ("zh", "速度是10km/h，电话：135-4567-8900"),
    ("ja", "そうですね、ほんと1年前"),
]


def _official_tokenizer():
    from indextts.utils.tokenizer import get_tokenizer

    return get_tokenizer(multilingual=True, model_dir="/root/IndexTTS-2.5")


def _ours():
    from windextts.frontend.tokenizer import build_tokenizer

    return build_tokenizer(model_dir="/root/IndexTTS-2.5")


def test_matches_dump():
    from windextts.frontend.tokenizer import build_tokenizer

    enc = build_tokenizer(model_dir="/root/IndexTTS-2.5")
    toks = enc.encode("<|zh|> 欢迎大家来体验indextts2。", allowed_special="all")
    ref = torch.load(f"{DUMPS}/frontend.tokens_zh.pt", weights_only=False).tolist()
    assert toks == ref, f"dump mismatch: {toks} != {ref}"
    print(f"[align] dump case match ({len(toks)} tokens)")


def test_multiple_cases():
    off = _official_tokenizer()
    ours = _ours()
    ok = 0
    for lang, text in CASES:
        s = f"<|{lang}|> {text}"
        ref_ids = off.encode(s, allowed_special="all")
        my_ids = ours.encode(s, allowed_special="all")
        status = "OK" if ref_ids == my_ids else "MISMATCH"
        if ref_ids == my_ids:
            ok += 1
        print(f"[align] {lang}: {status} {len(ref_ids)} ids -> {my_ids[:8]}...")
        assert ref_ids == my_ids, f"lang={lang} mismatch:\n ref={ref_ids}\n my ={my_ids}"
    print(f"[align] {ok}/{len(CASES)} cases exact match")


def test_vocab_size():
    from windextts.frontend.tokenizer import build_tokenizer

    enc = build_tokenizer(model_dir="/root/IndexTTS-2.5")
    # base 58836 + specials: 2 + 99 + 11 + 4 + 6 + 30 + 18 + 1501 = 1671?
    # Actual: official n_vocab=60509 -> 60509-58836 = 1673 specials. Verify against
    # the official config's number_text_tokens=60509 (config.yaml gpt section).
    assert enc.n_vocab == 60509, f"n_vocab={enc.n_vocab}"
    print(f"[align] n_vocab={enc.n_vocab} (== official config number_text_tokens)")


if __name__ == "__main__":
    test_matches_dump()
    test_multiple_cases()
    test_vocab_size()
    print("TOKENIZER ALIGN OK")
