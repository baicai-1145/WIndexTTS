"""QwenEmotion — text→emotion-vector predictor (pure torch, no transformers).

Reimplements ``indextts/infer_v2_5.py::QwenEmotion`` using:
  - ``tokenizers`` (lightweight Rust BPE) for the Qwen2 tokenizer
  - our pure-torch ``Qwen3ForCausalLM`` (qwen3.py) for generation
  - JSON parsing of the greedy-decoded emotion dict

The QwenEmotion model (qwen0.6bemo4-merge) is a Qwen3-0.6B fine-tune that reads
a text and emits a JSON dict of 8 emotion scores. We replicate the official
chat template (system="文本情感分类") and greedy generation, then parse the
output into the 8-dim ``emo_vector`` consumed by IndexTTS GPT conditioning.

No transformers/modelscope dependency.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .qwen3 import load_qwen3

# 8 emotions in the fixed order IndexTTS expects (happy, angry, sad, afraid,
# disgusted, melancholic, surprised, calm). Maps the Chinese keys emitted by
# QwenEmotion to the English names used downstream.
CN_KEY_TO_EN = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}
DESIRED_VECTOR_ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

# QwenEmotion confuses "悲伤"(sad) with "低落"(melancholic). When the input
# text contains melancholic keywords, swap those vectors. (Official workaround.)
MELANCHOLIC_WORDS = {"低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"}

EOS_TOKEN_ID = 151643  # <|endoftext|>
THINK_END_ID = 151668  # </think> (present only when enable_thinking=True)

# Greedy max tokens — emotion JSON is short (~70 tokens). Cap generously.
MAX_NEW_TOKENS = 512


def _build_chat_prompt(text_input: str) -> str:
    """Replicate the QwenEmotion chat template (enable_thinking=False).

    The official jinja template renders to a fixed string form. We hardcode it
    to avoid a jinja dependency:
        System: 文本情感分类<|endoftext|>\\nHuman: {text}<|endoftext|>\\nAssistant:
    """
    return (
        f"System: 文本情感分类{chr(0)}\nHuman: {text_input}{chr(0)}\nAssistant:"
    ).replace(chr(0), "<|endoftext|>")


class QwenEmotion:
    """Text→emotion predictor (pure torch Qwen3 + lightweight tokenizer).

    Args:
        model_dir: path to the qwen0.6bemo4-merge directory (config.json +
            model.safetensors + tokenizer.json).
        device: torch device.
        dtype: model dtype (fp16 by default, like the official load).
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from tokenizers import Tokenizer  # lightweight Rust BPE (not transformers)

        model_dir = Path(model_dir)
        self.model = load_qwen3(model_dir, device=device, dtype=dtype)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.device = device
        self.max_score = 1.2
        self.min_score = 0.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.2, value))

    def _normalize_content(self, content) -> dict:
        """Normalize the parsed emotion output into a {cn_key: score} dict."""
        if isinstance(content, dict):
            normalized = dict(content)
        else:
            normalized = {}

        def label_to_cn_key(value):
            if not isinstance(value, str):
                return None
            value = value.strip()
            if value in CN_KEY_TO_EN:
                return value
            for cn_key, en_key in CN_KEY_TO_EN.items():
                if value.lower() == en_key:
                    return cn_key
            return None

        # single-string label
        detected = label_to_cn_key(content) if isinstance(content, str) else None
        if detected is None:
            for alias in ("emotion", "emotion_label", "label", "情感", "情绪"):
                detected = label_to_cn_key(normalized.get(alias))
                if detected is not None:
                    break
        if detected is not None and all(k not in normalized for k in DESIRED_VECTOR_ORDER):
            normalized[detected] = 1.0

        for cn_key in DESIRED_VECTOR_ORDER:
            det = label_to_cn_key(normalized.get(cn_key))
            if det is not None:
                normalized[cn_key] = 1.0 if det == cn_key else 0.0
                if det != cn_key:
                    normalized[det] = 1.0
        return normalized

    def _convert(self, content) -> list[float]:
        """content dict → 8-dim emo_vector in DESIRED_VECTOR_ORDER."""
        content = self._normalize_content(content)
        vec = [
            self._clamp(float(content.get(k, 0.0)))
            for k in DESIRED_VECTOR_ORDER
        ]
        # default to calm if all-zero
        if all(v <= 0.0 for v in vec):
            vec[-1] = 1.0  # calm
        return vec

    @torch.no_grad()
    def inference(self, text_input: str) -> list[float]:
        """Predict 8-dim emotion vector from text (greedy Qwen3 decode + parse)."""
        prompt = _build_chat_prompt(text_input)
        enc = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([enc.ids], device=self.device, dtype=torch.long)

        full = self.model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID
        )
        gen_ids = full[0, input_ids.size(1) :].tolist()

        # strip <think>...</think> if present (enable_thinking=False usually omits it)
        try:
            idx = len(gen_ids) - gen_ids[::-1].index(THINK_END_ID)
        except ValueError:
            idx = 0
        content = self.tokenizer.decode(gen_ids[idx:])

        # parse JSON emotion dict
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # fallback: manual "key": number parsing
            parsed = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }

        # melancholic workaround: swap 悲伤/低落 for melancholic keywords
        if any(w in text_input.lower() for w in MELANCHOLIC_WORDS):
            parsed["悲伤"], parsed["低落"] = parsed.get("低落", 0.0), parsed.get("悲伤", 0.0)

        return self._convert(parsed)
