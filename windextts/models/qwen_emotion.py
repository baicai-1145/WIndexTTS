"""QwenEmotion — text→emotion-vector predictor (pure torch, no transformers).

Reimplements ``indextts/infer_v2_5.py::QwenEmotion``: qwen0.6bemo4-merge is a
Qwen3-0.6B fine-tune that emits a JSON dict of 8 emotion scores. We hardcode
the official chat template (System=文本情感分类, enable_thinking=False), greedy
decode via our pure-torch Qwen3ForCausalLM (qwen3.py), then parse the JSON into
the 8-dim emo_vector consumed by IndexTTS GPT conditioning. No transformers /
modelscope dependency.
"""

import json
import re
from pathlib import Path

import torch

from .qwen3 import load_qwen3

# 8 emotions in the fixed order IndexTTS expects (happy, angry, sad, afraid,
# disgusted, melancholic, surprised, calm). Maps the Chinese keys the model
# emits to the English names used downstream.
CN_KEY_TO_EN = {"高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid",
                "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised", "自然": "calm"}
DESIRED_VECTOR_ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

# QwenEmotion confuses 悲伤(sad) with 低落(melancholic); when the input text
# contains melancholic keywords, swap those two scores. (Official workaround.)
MELANCHOLIC_WORDS = {"低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"}

EOS_TOKEN_ID = 151643  # <|endoftext|>
THINK_END_ID = 151668  # </think> (present only when enable_thinking=True)

# Greedy max tokens — emotion JSON is highly templated: 76-77 tokens across 12
# diverse inputs (the fixed JSON skeleton + 8 scores). 80 leaves a safe margin
# and keeps KV/attention footprint small for faster graph decode.
MAX_NEW_TOKENS = 80


def _build_chat_prompt(text_input: str) -> str:
    # Hardcoded render of the official jinja template (enable_thinking=False),
    # replicating the template's \x00 -> <|endoftext|> substitution exactly:
    return (f"System: 文本情感分类{chr(0)}\nHuman: {text_input}{chr(0)}\nAssistant:"
            .replace(chr(0), "<|endoftext|>"))


class QwenEmotion:
    def __init__(self, model_dir: str | Path, device: str = "cuda", dtype: torch.dtype = torch.float16):
        from tokenizers import Tokenizer  # lightweight Rust BPE (not transformers)

        self.model = load_qwen3(model_dir, device=device, dtype=dtype)
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.device = device
        self.max_score, self.min_score = 1.2, 0.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.2, value))

    def _normalize_content(self, content) -> dict:
        # -> {cn_key: score}: accepts a dict, or a single label (string / alias
        # keys like "emotion"/"label"), with en-alias -> cn mapping.
        def to_cn(v):
            if isinstance(v, str):
                v = v.strip()
                if v in CN_KEY_TO_EN:
                    return v
                for k, e in CN_KEY_TO_EN.items():
                    if v.lower() == e:
                        return k
            return None

        n = dict(content) if isinstance(content, dict) else {}
        d = to_cn(content) if isinstance(content, str) else None
        if d is None:
            for a in ("emotion", "emotion_label", "label", "情感", "情绪"):
                if (d := to_cn(n.get(a))) is not None:
                    break
        if d is not None and all(k not in n for k in DESIRED_VECTOR_ORDER):
            n[d] = 1.0
        for k in DESIRED_VECTOR_ORDER:
            det = to_cn(n.get(k))
            if det is not None:
                n[k] = 1.0 if det == k else 0.0
                if det != k:
                    n[det] = 1.0
        return n

    def _convert(self, content) -> list[float]:
        # content dict -> 8-dim emo_vector in DESIRED_VECTOR_ORDER; calm if all-zero
        content = self._normalize_content(content)
        vec = [self._clamp(float(content.get(k, 0.0))) for k in DESIRED_VECTOR_ORDER]
        if all(v <= 0.0 for v in vec):
            vec[-1] = 1.0  # calm
        return vec

    @torch.no_grad()
    def inference(self, text_input: str) -> list[float]:
        # Greedy Qwen3 decode of the chat prompt, then JSON-parse the emotion dict.
        enc = self.tokenizer.encode(_build_chat_prompt(text_input))
        input_ids = torch.tensor([enc.ids], device=self.device, dtype=torch.long)
        full = self.model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID,
            use_cuda_graph=True,
        )
        gen = full[0, input_ids.size(1):].tolist()
        try:  # strip <think>…</think> if present (usually omitted, keep safe)
            idx = len(gen) - gen[::-1].index(THINK_END_ID)
        except ValueError:
            idx = 0
        content = self.tokenizer.decode(gen[idx:])
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:  # fallback: manual "key": number parsing
            parsed = {m.group(1): float(m.group(2))
                      for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)}
        if any(w in text_input.lower() for w in MELANCHOLIC_WORDS):
            parsed["悲伤"], parsed["低落"] = parsed.get("低落", 0.0), parsed.get("悲伤", 0.0)
        return self._convert(parsed)
