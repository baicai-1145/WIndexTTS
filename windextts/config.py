"""Typed access to IndexTTS-2.5 config.yaml — the single source of truth for
hyperparameters. Built once from the weights dir and passed to every module;
no module hardcodes magic numbers that exist in config.yaml. Magic numbers that
live in infer_v2_5.py source instead are centralized in RuntimeConstants.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


# infer_v2_5.py source constants (NOT config.yaml); each cites its line.
@dataclass(frozen=True)
class RuntimeConstants:
    w2v_layer: int = 17            # hidden_states[17] as ref audio feature (:288)
    s2mel_duration_scale: float = 1.72          # 1.72 * duration_factor (:855)
    s2mel_n_quantizers: int = 3                 # (:655,859)
    cfm_diffusion_steps: int = 25               # (:849-850)
    cfm_inference_cfg_rate: float = 0.7         # (:849-850)
    output_sr: int = 22050                      # (:514)
    ref_sr_mel: int = 22000                     # resample for mel path (:628-629)
    ref_sr_w2v: int = 16000                     # for w2v-bert & campplus (:628-629)
    ref_max_seconds: float = 15.0               # max ref audio length taken


# config.yaml sections, loosely typed. Field defaults == from_yaml fallbacks,
# so missing yaml keys resolve identically through the dataclass defaults.
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
    condition_module: dict = field(default_factory=dict)       # nested dict, as-is
    emo_condition_module: dict = field(default_factory=dict)   # nested dict, as-is


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
    style_encoder_dim: int = 192           # s2mel.style_encoder.dim
    length_reg: S2MelLengthRegConfig = field(default_factory=S2MelLengthRegConfig)
    dit: S2MelDiTConfig = field(default_factory=S2MelDiTConfig)
    wavenet_hidden_dim: int = 512          # s2mel.wavenet.* (flattened)
    wavenet_num_layers: int = 8
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_p_dropout: float = 0.2
    wavenet_style_condition: bool = True


def _sec(cls, d, **ov):
    """Construct a section dataclass from a raw yaml dict, filtering to its
    declared fields (unknown keys dropped) and applying explicit overrides
    (renames / type coercions). Missing keys fall through to dataclass defaults,
    which equal the former per-key fallbacks."""
    ks = {f.name for f in fields(cls)}
    kw = {k: v for k, v in d.items() if k in ks and k not in ov}   # ov wins
    return cls(**kw, **ov)


@dataclass
class Config:
    """Top-level config mirroring IndexTTS-2.5 config.yaml."""

    mel: MelConfig = field(default_factory=MelConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    semantic_codec: SemanticCodecConfig = field(default_factory=SemanticCodecConfig)
    s2mel: S2MelConfig = field(default_factory=S2MelConfig)
    vocoder_type: str = "bigvgan"
    vocoder_name: str = "bigvgan_generator.pt"
    version: str = "2.5"
    _raw: dict = field(default_factory=dict, repr=False)     # full yaml fallback
    rt: RuntimeConstants = field(default_factory=RuntimeConstants)

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(open(path, encoding="utf-8"))
        ds, g = raw.get("dataset", {}) or {}, raw.get("gpt", {}) or {}
        sc, s2 = raw.get("semantic_codec", {}) or {}, raw.get("s2mel", {}) or {}
        pre = (s2.get("preprocess_params", {}) or {}).get("spect_params", {}) or {}
        lr, dit, wn = (s2.get("length_regulator", {}) or {}, s2.get("DiT", {}) or {},
                       s2.get("wavenet", {}) or {})
        voc = raw.get("vocoder", {}) or {}
        fmax = pre.get("fmax")
        fmax = None if (fmax is None or fmax == "None") else float(fmax)
        return cls(
            mel=_sec(MelConfig, ds.get("mel", {}) or {}),
            gpt=_sec(GPTConfig, g,
                     condition_module=g.get("condition_module", {}) or {},
                     emo_condition_module=g.get("emo_condition_module", {}) or {}),
            semantic_codec=_sec(SemanticCodecConfig, sc),
            s2mel=S2MelConfig(
                **{k: pre[k] for k in ("sr", "n_fft", "win_length", "hop_length", "n_mels", "fmin") if k in pre},
                fmax=fmax,
                dit_type=s2.get("dit_type", "DiT"),
                reg_loss_type=s2.get("reg_loss_type", "l1"),
                style_encoder_dim=(s2.get("style_encoder", {}) or {}).get("dim", 192),
                length_reg=_sec(S2MelLengthRegConfig, lr,
                                sampling_ratios=tuple(lr.get("sampling_ratios", [1, 1, 1, 1]))),
                dit=_sec(S2MelDiTConfig, dit),
                wavenet_hidden_dim=wn.get("hidden_dim", 512),
                wavenet_num_layers=wn.get("num_layers", 8),
                wavenet_kernel_size=wn.get("kernel_size", 5),
                wavenet_dilation_rate=wn.get("dilation_rate", 1),
                wavenet_p_dropout=wn.get("p_dropout", 0.2),
                wavenet_style_condition=wn.get("style_condition", True),
            ),
            vocoder_type=voc.get("type", "bigvgan"),
            vocoder_name=voc.get("name", "bigvgan_generator.pt"),
            version=raw.get("version", "2.5"),
            _raw=raw,
        )


DEFAULT_CONFIG_PATH = os.environ.get("WINDEXTTS_WEIGHTS_DIR", "/root/IndexTTS-2.5") + "/config.yaml"


def load_default_config() -> Config:
    """Load config from the canonical IndexTTS-2.5 weights dir."""
    return Config.from_yaml(DEFAULT_CONFIG_PATH)
