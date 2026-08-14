"""Typed access to the IndexTTS-2.5 config.yaml.

This is the single source of truth for hyperparameters. The ``Config`` object
is constructed once from ``/root/IndexTTS-2.5/config.yaml`` (or a copied file)
and passed to every module. No module should hardcode magic numbers that exist
in config.yaml — read them here.

Magic numbers that live in source code (infer_v2_5.py), not config.yaml — e.g.
``w2v-bert hidden_states[17]``, ``1.72`` duration scale, ``diffusion_steps=25`` —
are collected in :class:`RuntimeConstants` so they are also centralized.
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Runtime constants — these come from infer_v2_5.py source, NOT config.yaml.
# Every value must cite its source line. Do NOT invent or tweak.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeConstants:
    """Constants hardcoded in indextts/infer_v2_5.py (not in config.yaml).

    Each field cites the source line in ``infer_v2_5.py``. These are the
    "seam magic numbers" — changing any of them produces wrong output.
    """

    # w2v-bert: which hidden_states layer to use as ref audio feature.
    # infer_v2_5.py:288  hidden_states[17]
    w2v_layer: int = 17

    # w2v-bert feature normalization: (feat - mean) / sqrt(var)
    # stats loaded from wav2vec2bert_stats.pt {mean:[1024], var:[1024]}
    # infer_v2_5.py:177-179, 289
    # (handled in weights/frontend, kept here only as documentation anchor)

    # S2Mel duration scaling. infer_v2_5.py:855  1.72 * duration_factor
    s2mel_duration_scale: float = 1.72

    # S2Mel length_regulator n_quantizers. infer_v2_5.py:655,859
    s2mel_n_quantizers: int = 3

    # S2Mel CFM. infer_v2_5.py:849-850  diffusion_steps=25, inference_cfg_rate=0.7
    cfm_diffusion_steps: int = 25
    cfm_inference_cfg_rate: float = 0.7

    # Output audio. infer_v2_5.py:514  22050 Hz mono
    output_sr: int = 22050

    # Reference audio resample targets. infer_v2_5.py:628-629
    ref_sr_mel: int = 22000      # for mel / mel-condition path
    ref_sr_w2v: int = 16000      # for w2v-bert & campplus

    # Max reference audio length taken (seconds). infer_v2_5.py path
    ref_max_seconds: float = 15.0


# ---------------------------------------------------------------------------
# config.yaml sections — mirrors /root/IndexTTS-2.5/config.yaml structure.
# Kept loosely typed (nested dataclasses of plain values) to avoid coupling to
# any external library's config schema.
# ---------------------------------------------------------------------------


@dataclass
class MelConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    mel_fmin: float = 0.0
    normalize: bool = False


@dataclass
class GPTConfig:
    model_dim: int = 1280
    max_mel_tokens: int = 1815
    max_text_tokens: int = 600
    heads: int = 20
    use_mel_codes_as_input: bool = True
    mel_length_compression: int = 1024
    layers: int = 24
    number_text_tokens: int = 60509
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    start_text_token: int = 0
    stop_text_token: int = 1
    train_solo_embeddings: bool = False
    condition_type: str = "conformer_perceiver"
    # condition_module / emo_condition_module are nested dicts; kept as-is.
    condition_module: dict = field(default_factory=dict)
    emo_condition_module: dict = field(default_factory=dict)


@dataclass
class SemanticCodecConfig:
    codebook_size: int = 8192
    hidden_size: int = 1024
    codebook_dim: int = 8
    vocos_dim: int = 384
    vocos_intermediate_dim: int = 2048
    vocos_num_layers: int = 12


@dataclass
class S2MelDiTConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    depth: int = 13
    class_dropout_prob: float = 0.1
    block_size: int = 8192
    in_channels: int = 80
    style_condition: bool = True
    final_layer_type: str = "wavenet"
    target: str = "mel"
    content_dim: int = 512
    content_codebook_size: int = 1024
    content_type: str = "discrete"
    f0_condition: bool = False
    n_f0_bins: int = 512
    content_codebooks: int = 1
    is_causal: bool = False
    long_skip_connection: bool = True
    zero_prompt_speech_token: bool = False
    time_as_token: bool = False
    style_as_token: bool = False
    uvit_skip_connection: bool = True
    add_resblock_in_transformer: bool = False


@dataclass
class S2MelLengthRegConfig:
    channels: int = 512
    is_discrete: bool = False
    in_channels: int = 1024
    content_codebook_size: int = 2048
    sampling_ratios: tuple = (1, 1, 1, 1)
    vector_quantize: bool = False
    n_codebooks: int = 1
    quantizer_dropout: float = 0.0
    f0_condition: bool = False
    n_f0_bins: int = 512


@dataclass
class S2MelConfig:
    # preprocess_params.spect_params
    sr: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None
    dit_type: str = "DiT"
    reg_loss_type: str = "l1"
    style_encoder_dim: int = 192
    length_reg: S2MelLengthRegConfig = field(default_factory=S2MelLengthRegConfig)
    dit: S2MelDiTConfig = field(default_factory=S2MelDiTConfig)
    wavenet_hidden_dim: int = 512
    wavenet_num_layers: int = 8
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_p_dropout: float = 0.2
    wavenet_style_condition: bool = True


@dataclass
class Config:
    """Top-level config mirroring IndexTTS-2.5 config.yaml."""

    # dataset.mel
    mel: MelConfig = field(default_factory=MelConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    semantic_codec: SemanticCodecConfig = field(default_factory=SemanticCodecConfig)
    s2mel: S2MelConfig = field(default_factory=S2MelConfig)
    # vocoder
    vocoder_type: str = "bigvgan"
    vocoder_name: str = "bigvgan_generator.pt"
    version: str = "2.5"
    # raw dict fallback for anything not yet dataclassed
    _raw: dict = field(default_factory=dict, repr=False)
    # runtime constants from infer_v2_5.py
    rt: RuntimeConstants = field(default_factory=RuntimeConstants)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        ds = raw.get("dataset", {}) or {}
        mel_raw = ds.get("mel", {}) or {}
        mel = MelConfig(
            sample_rate=mel_raw.get("sample_rate", 24000),
            n_fft=mel_raw.get("n_fft", 1024),
            hop_length=mel_raw.get("hop_length", 256),
            win_length=mel_raw.get("win_length", 1024),
            n_mels=mel_raw.get("n_mels", 100),
            mel_fmin=mel_raw.get("mel_fmin", 0.0),
            normalize=mel_raw.get("normalize", False),
        )

        g = raw.get("gpt", {}) or {}
        gpt = GPTConfig(
            model_dim=g.get("model_dim", 1280),
            max_mel_tokens=g.get("max_mel_tokens", 1815),
            max_text_tokens=g.get("max_text_tokens", 600),
            heads=g.get("heads", 20),
            use_mel_codes_as_input=g.get("use_mel_codes_as_input", True),
            mel_length_compression=g.get("mel_length_compression", 1024),
            layers=g.get("layers", 24),
            number_text_tokens=g.get("number_text_tokens", 60509),
            number_mel_codes=g.get("number_mel_codes", 8194),
            start_mel_token=g.get("start_mel_token", 8192),
            stop_mel_token=g.get("stop_mel_token", 8193),
            start_text_token=g.get("start_text_token", 0),
            stop_text_token=g.get("stop_text_token", 1),
            train_solo_embeddings=g.get("train_solo_embeddings", False),
            condition_type=g.get("condition_type", "conformer_perceiver"),
            condition_module=g.get("condition_module", {}) or {},
            emo_condition_module=g.get("emo_condition_module", {}) or {},
        )

        sc = raw.get("semantic_codec", {}) or {}
        semantic_codec = SemanticCodecConfig(
            codebook_size=sc.get("codebook_size", 8192),
            hidden_size=sc.get("hidden_size", 1024),
            codebook_dim=sc.get("codebook_dim", 8),
            vocos_dim=sc.get("vocos_dim", 384),
            vocos_intermediate_dim=sc.get("vocos_intermediate_dim", 2048),
            vocos_num_layers=sc.get("vocos_num_layers", 12),
        )

        s2 = raw.get("s2mel", {}) or {}
        pre = (s2.get("preprocess_params", {}) or {}).get("spect_params", {}) or {}
        fmax_raw = pre.get("fmax", None)
        fmax = None if (fmax_raw is None or fmax_raw == "None") else float(fmax_raw)
        lr_raw = s2.get("length_regulator", {}) or {}
        length_reg = S2MelLengthRegConfig(
            channels=lr_raw.get("channels", 512),
            is_discrete=lr_raw.get("is_discrete", False),
            in_channels=lr_raw.get("in_channels", 1024),
            content_codebook_size=lr_raw.get("content_codebook_size", 2048),
            sampling_ratios=tuple(lr_raw.get("sampling_ratios", [1, 1, 1, 1])),
            vector_quantize=lr_raw.get("vector_quantize", False),
            n_codebooks=lr_raw.get("n_codebooks", 1),
            quantizer_dropout=lr_raw.get("quantizer_dropout", 0.0),
            f0_condition=lr_raw.get("f0_condition", False),
            n_f0_bins=lr_raw.get("n_f0_bins", 512),
        )
        dit_raw = s2.get("DiT", {}) or {}
        dit = S2MelDiTConfig(
            hidden_dim=dit_raw.get("hidden_dim", 512),
            num_heads=dit_raw.get("num_heads", 8),
            depth=dit_raw.get("depth", 13),
            class_dropout_prob=dit_raw.get("class_dropout_prob", 0.1),
            block_size=dit_raw.get("block_size", 8192),
            in_channels=dit_raw.get("in_channels", 80),
            style_condition=dit_raw.get("style_condition", True),
            final_layer_type=dit_raw.get("final_layer_type", "wavenet"),
            target=dit_raw.get("target", "mel"),
            content_dim=dit_raw.get("content_dim", 512),
            content_codebook_size=dit_raw.get("content_codebook_size", 1024),
            content_type=dit_raw.get("content_type", "discrete"),
            f0_condition=dit_raw.get("f0_condition", False),
            n_f0_bins=dit_raw.get("n_f0_bins", 512),
            content_codebooks=dit_raw.get("content_codebooks", 1),
            is_causal=dit_raw.get("is_causal", False),
            long_skip_connection=dit_raw.get("long_skip_connection", True),
            zero_prompt_speech_token=dit_raw.get("zero_prompt_speech_token", False),
            time_as_token=dit_raw.get("time_as_token", False),
            style_as_token=dit_raw.get("style_as_token", False),
            uvit_skip_connection=dit_raw.get("uvit_skip_connection", True),
            add_resblock_in_transformer=dit_raw.get("add_resblock_in_transformer", False),
        )
        wn = s2.get("wavenet", {}) or {}
        s2mel = S2MelConfig(
            sr=pre.get("sr", 22050),
            n_fft=pre.get("n_fft", 1024),
            win_length=pre.get("win_length", 1024),
            hop_length=pre.get("hop_length", 256),
            n_mels=pre.get("n_mels", 80),
            fmin=pre.get("fmin", 0.0),
            fmax=fmax,
            dit_type=s2.get("dit_type", "DiT"),
            reg_loss_type=s2.get("reg_loss_type", "l1"),
            style_encoder_dim=(s2.get("style_encoder", {}) or {}).get("dim", 192),
            length_reg=length_reg,
            dit=dit,
            wavenet_hidden_dim=wn.get("hidden_dim", 512),
            wavenet_num_layers=wn.get("num_layers", 8),
            wavenet_kernel_size=wn.get("kernel_size", 5),
            wavenet_dilation_rate=wn.get("dilation_rate", 1),
            wavenet_p_dropout=wn.get("p_dropout", 0.2),
            wavenet_style_condition=wn.get("style_condition", True),
        )

        voc = raw.get("vocoder", {}) or {}
        return cls(
            mel=mel,
            gpt=gpt,
            semantic_codec=semantic_codec,
            s2mel=s2mel,
            vocoder_type=voc.get("type", "bigvgan"),
            vocoder_name=voc.get("name", "bigvgan_generator.pt"),
            version=raw.get("version", "2.5"),
            _raw=raw,
        )


DEFAULT_CONFIG_PATH = os.environ.get("WINDEXTTS_WEIGHTS_DIR", "/root/IndexTTS-2.5") + "/config.yaml"


def load_default_config() -> Config:
    """Load config from the canonical IndexTTS-2.5 weights dir."""
    return Config.from_yaml(DEFAULT_CONFIG_PATH)
