"""IndexTTS-2.5 multilingual text tokenizer — pure tiktoken, zero whisper/transformers.

Re-implements ``indextts/utils/tokenizer.py`` ``get_encoding`` for the v2.5
inference path. v2.5 only calls ``tokenizer.encode('<|{lang}|> ' + text,
allowed_special='all')`` — the WhisperTokenizer ``tokenize``/``convert_*``
machinery is v1-only and NOT needed.

The tokenizer is a plain ``tiktoken.Encoding``:
  - BPE merge ranks from ``multilingual_zh_ja_yue_char_del.tiktoken`` (58836 base vocab).
  - Special tokens appended in the exact official order (their ids = 58836 + index):
    <|endoftext|>, <|startoftranscript|>, <|lang|> x num_languages(99),
    <|audio_event|> x 11, <|emotion|> x 4, <|translate|>, <|transcribe|>,
    <|startoflm|>, <|startofprev|>, <|nospeech|>, <|notimestamps|>,
    <|SPECIAL_TOKEN_{1..30}|>, <|TTS/...|> x 18, <|{i*0.02:.2f}|> x 1501.

pat_str and everything else match official exactly → encode() is bit-identical.
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

import tiktoken

__all__ = ["LANGUAGES", "AUDIO_EVENT", "EMOTION", "TTS_Vocal_Token", "lang_to_token", "build_tokenizer"]

# ---------------------------------------------------------------------------
# Constants — copied 1:1 from indextts/utils/tokenizer.py (order matters:
# the special-token id assignment depends on iteration order).
# ---------------------------------------------------------------------------

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
    "common": "common",
}

# language code lookup by name (used by lang_to_token fallback in official code)
LANGUAGE_DICT = {lang: index for index, lang in enumerate(LANGUAGES.keys())}

AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech",
    "/Speech": "/Speech",
    "BGM": "BGM",
    "/BGM": "/BGM",
    "Laughter": "Laughter",
    "/Laughter": "/Laughter",
    "Applause": "Applause",
    "/Applause": "/Applause",
}

EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}

TTS_Vocal_Token = {
    "TTS/B": "TTS/B",
    "TTS/O": "TTS/O",
    "TTS/Q": "TTS/Q",
    "TTS/A": "TTS/A",
    "TTS/CO": "TTS/CO",
    "TTS/CL": "TTS/CL",
    "TTS/H": "TTS/H",
    **{f"TTS/SP{i:02d}": f"TTS/SP{i:02d}" for i in range(1, 14)},
}

# NOTE: default num_languages=99 — the first 99 keys of LANGUAGES (through "su").
# yue/minnan/wuyu/dialect/zh-en/en-zh/common are NOT registered as special tokens,
# matching official get_tokenizer(multilingual=True) defaults.
_PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def lang_to_token(lang: str) -> int:
    """Language -> lang_embedding row index (for GPT lang_embedding), NOT a token id."""
    lang = lang.lower()
    if lang not in LANGUAGE_DICT:
        lang = "common"
    return LANGUAGE_DICT[lang]


def _build_specials(num_languages: int) -> list[str]:
    """Exact special-token list in official order (tokenizer.py:196-209)."""
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        *[f"<|{audio_event}|>" for audio_event in list(AUDIO_EVENT.keys())],
        *[f"<|{emotion}|>" for emotion in list(EMOTION.keys())],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],
        *[f"<|{tts}|>" for tts in list(TTS_Vocal_Token.keys())],
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    return specials


def _load_ranks(vocab_path: str | Path) -> dict[bytes, int]:
    """Load BPE merge ranks from the .tiktoken file (base64-encoded byte -> rank)."""
    ranks: dict[bytes, int] = {}
    with open(vocab_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


@lru_cache(maxsize=None)
def build_tokenizer(
    name: str = "multilingual_zh_ja_yue_char_del",
    num_languages: int = 99,
    model_dir: str | None = None,
) -> tiktoken.Encoding:
    """Build the tiktoken Encoding identical to official get_encoding.

    Mirrors indextts/utils/tokenizer.py get_encoding() exactly:
    ranks + specials in official order -> tiktoken.Encoding(name, explicit_n_vocab,
    pat_str, mergeable_ranks, special_tokens).

    model_dir defaults to the WINDEXTTS_WEIGHTS_DIR env var (or the dev-box
    canonical path); pass explicitly or set the env var.
    """
    if model_dir is None:
        model_dir = os.environ.get("WINDEXTTS_WEIGHTS_DIR", "/root/IndexTTS-2.5")
    vocab_path = os.path.join(model_dir, f"{name}.tiktoken")
    ranks = _load_ranks(vocab_path)
    n_vocab = len(ranks)

    special_tokens = {}
    for token in _build_specials(num_languages):
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=_PAT_STR,
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


if __name__ == "__main__":
    import torch

    enc = build_tokenizer()
    toks = enc.encode("<|zh|> 欢迎大家来体验indextts2。", allowed_special="all")
    print("tokens:", toks)
    print("total:", len(toks))
    ref = torch.load("/root/windextts_dumps/frontend.tokens_zh.pt", weights_only=False)
    ref_list = ref.tolist()
    print("match frontend.tokens_zh.pt:", toks == ref_list)
    # round-trip decode sanity
    print("decode:", enc.decode(toks)[:50])
