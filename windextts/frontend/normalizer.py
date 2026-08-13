"""Text normalization (TN) for IndexTTS-2.5 — zero transformers/modelscope.

Port of ``indextts/utils/front.py`` ``TextNormalizer`` (used by ``infer_v2_5.py``
before tokenization). Normalizes raw user text (digits, dates, percents, pinyin
tone marks, technical terms, names) into the spoken form the tiktoken
tokenizer expects.

Adaptations vs official:
  - Uses ``tn`` (NeMo text processing) directly for zh/en TN instead of the
    ``wetext`` wrapper (same underlying grammar; wetext is a thin wrapper and is
    not installed in the reference env). Official uses wetext on Mac/Windows and
    tn on Linux; we unify on tn.
  - TN grammar cache lives in ``{tempdir}/windextts_tn_cache`` (not in-repo).
  - No ``TextTokenizer``/SentencePiece machinery (v2.5 uses tiktoken).
  - Normalizer auto-loads lazily on first ``normalize()`` call.
"""
from __future__ import annotations

import os
import re
import tempfile
import traceback
from functools import lru_cache


__all__ = ["TextNormalizer"]


class TextNormalizer:
    def __init__(self, enable_glossary: bool = False):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }
        self.clean_pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        self.enable_glossary = enable_glossary
        # 术语词汇表：用户可自定义专业术语的读法
        # 格式: {"原始术语": {"en": "英文读法", "zh": "中文读法"}}
        self.term_glossary = dict()

    def match_email(self, email: str) -> bool:
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声
    例如：xuan4, jve2, ying1, zhong4, shang5
    不匹配：beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    """
    匹配人名，格式：中文·中文，中文·中文-中文
    例如：克里斯托弗·诺兰，约瑟夫·高登-莱维特
    """
    TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
    """
    匹配技术术语，格式：字母开头+(字母或数字)*+(-字母或数字)+
    例如：GPT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2
    必须以字母开头，避免匹配纯数字（如电话号码 135-4567-8900）
    """
    # 匹配常见英语缩写 's，仅用于替换为 is，不匹配所有 's
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"

    def use_chinese(self, s: str) -> bool:
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self) -> None:
        """Lazily build the zh/en TN normalizers (NeMo ``tn`` grammar).

        Unified on ``tn`` (the underlying grammar wetext wraps) so we do not
        depend on the wetext packaging layer. The zh grammar needs a writable
        cache dir for the compiled tagger rules; we use ``{tmp}/windextts_tn_cache``.
        """
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        from tn.chinese.normalizer import Normalizer as NormalizerZh
        from tn.english.normalizer import Normalizer as NormalizerEn

        cache_dir = os.path.join(tempfile.gettempdir(), "windextts_tn_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.zh_normalizer = NormalizerZh(
            cache_dir=cache_dir, remove_interjections=False, remove_erhua=False,
            overwrite_cache=False,
        )
        self.en_normalizer = NormalizerEn(overwrite_cache=False)

    G2P_PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")

    def _protect_pronunciation_annotations(self, text: str):
        """
        在 normalize 之前调用：将 <字|读音> 标注替换为纯字母占位符，
        防止 normalizer 把标注内的数字/符号展开（如 XING2 -> XING二）。
        返回 (替换后文本, 占位符字典)。
        """
        placeholders = {}

        def _idx_to_alpha(n):
            s = ""
            while True:
                s = chr(ord("a") + n % 26) + s
                n = n // 26 - 1
                if n < 0:
                    break
            return s

        def _replacer(m):
            tag = _idx_to_alpha(len(placeholders))
            key = f"PRONPLACEHOLDER{tag}PRONPLACEHOLDER"
            placeholders[key] = m.group(0)
            return key

        text = self.G2P_PRONUNCIATION_ANNOTATION_PATTERN.sub(_replacer, text)
        return text, placeholders

    @staticmethod
    def _restore_pronunciation_annotations(text: str, placeholders: dict) -> str:
        """在 normalize 之后调用：将占位符还原为原始 <字|读音> 标注。"""
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def normalize(self, text: str) -> str:
        if self.zh_normalizer is None or self.en_normalizer is None:
            self.load()
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        # 保护 G2P 发音标注 <word|pronunciation>，防止被 normalizer 破坏
        text, _pron_placeholders = self._protect_pronunciation_annotations(text)
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            # 应用术语词汇表（优先级最高，在所有保护之前）
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang="zh")
            # 保护技术术语（如 GPT-5-nano）避免被中文normalizer错误处理
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)
            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # 恢复人名
            result = self.restore_names(result, original_name_list)
            # 恢复拼音声调
            result = self.restore_pinyin_tones(result, pinyin_list)
            # 恢复技术术语
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                # 应用术语词汇表（优先级最高，在所有保护之前）
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                # 保护技术术语（如 GPT-5-Nano）避免被英文normalizer错误处理
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                # 恢复技术术语
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)

        # 恢复 G2P 发音标注
        result = self._restore_pronunciation_annotations(result, _pron_placeholders)
        return result

    def correct_pinyin(self, pinyin: str) -> str:
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # 匹配 jqx 的韵母为 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text: str):
        """
        替换人名为占位符 <n_a>、 <n_b>, ...
        例如：克里斯托弗·诺兰 -> <n_a>
        """
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 替换占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")
        return transformed_text, original_name_list

    def restore_names(self, normalized_text: str, original_name_list):
        """
        恢复人名为原来的文字
        例如：<n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_tech_terms(self, original_text: str):
        """
        保护技术术语中的连字符，防止被中文normalizer解析为减号
        策略：将术语中的连字符替换为特殊占位符<H>，数字仍可被正常处理
        例如：GPT-5-nano -> GPT<H>5<H>nano，然后 5 被转换为 五
        最终恢复为：GPT-五-nano
        """
        tech_pattern = re.compile(TextNormalizer.TECH_TERM_PATTERN)
        original_tech_list = tech_pattern.findall(original_text)
        if len(original_tech_list) == 0:
            return (original_text, None)
        # 去重并按长度降序排列（避免短匹配先替换导致问题）
        original_tech_list = sorted(set(original_tech_list), key=len, reverse=True)
        transformed_text = original_text
        for term in original_tech_list:
            protected_term = term.replace("-", "<H>")
            transformed_text = transformed_text.replace(term, protected_term)
        return transformed_text, original_tech_list

    def restore_tech_terms(self, normalized_text: str, original_tech_list):
        """
        恢复技术术语中的连字符
        将占位符 <H> 恢复为连字符 -
        同时清理 normalizer 可能在占位符周围添加的多余空格
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text
        # 清理 <H> 周围可能的空格，然后恢复为连字符
        transformed_text = re.sub(r"\s*<H>\s*", "-", normalized_text)
        return transformed_text

    def apply_glossary_terms(self, text: str, lang: str = "zh") -> str:
        """
        应用术语词汇表，将专业术语替换为对应语言的读法

        Args:
            text: 待处理文本
            lang: 语言类型 "zh" 或 "en"

        Returns:
            处理后的文本
        """
        if not self.term_glossary:
            return text
        # 按术语长度降序排列，避免短术语先匹配导致长术语无法匹配
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)

        @lru_cache(maxsize=42)
        def get_term_pattern(term: str):
            return re.compile(re.escape(term), re.IGNORECASE)

        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)
        return transformed_text

    def load_glossary(self, glossary_dict: dict) -> None:
        """
        加载外部术语词汇表

        Args:
            glossary_dict: 术语词典，格式为 {"术语": {"en": "英文读法", "zh": "中文读法"}}
        """
        if glossary_dict and isinstance(glossary_dict, dict):
            self.term_glossary.update(glossary_dict)

    def load_glossary_from_yaml(self, glossary_path: str) -> bool:
        """
        从 YAML 文件加载术语词汇表。YAML 格式:
            M.2:
              en: M dot two
              zh: M 二
            NVMe: N-V-M-E
        """
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, "r", encoding="utf-8") as f:
                external_glossary = yaml.safe_load(f)
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_glossary_to_yaml(self, glossary_path: str) -> None:
        """保存术语词汇表到 YAML 文件。"""
        import yaml
        with open(glossary_path, "w", encoding="utf-8") as f:
            yaml.dump(self.term_glossary, f, allow_unicode=True, default_flow_style=False)

    def save_pinyin_tones(self, original_text: str):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text: str, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        return transformed_text


if __name__ == "__main__":
    norm = TextNormalizer()
    tests = [
        "我有123本书",
        "3.14%的利润",
        "2024年1月1日",
        "你好world",
        "GPT-5的参数",
        "IT行业薪资",
        "我叫张三",
    ]
    for t in tests:
        print(f"{t!r} -> {norm.normalize(t)!r}")

    print("\n--- 拼音声调 ---")
    for t in ["晕XUAN4是一种GAN3觉", "受不liao3你了"]:
        print(f"{t!r} -> {norm.normalize(t)!r}")

    print("\n--- 技术术语 ---")
    for t in ["GPT-5-Nano 是 GPT-5 模型家族中最小且速度最快的变体", "2025/09/08 IndexTTS-2 全球发布"]:
        print(f"{t!r} -> {norm.normalize(t)!r}")

    print("\n--- 人名 ---")
    for t in ["克里斯托弗·诺兰执导了《盗梦空间》", "约瑟夫·高登-莱维特"]:
        print(f"{t!r} -> {norm.normalize(t)!r}")

    print("\n--- 英文 ---")
    for t in ["I love you!", "See you at 8:00 AM", "This sales for 2.5% off, only $12.5."]:
        print(f"{t!r} -> {norm.normalize(t)!r}")
