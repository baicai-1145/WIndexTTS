"""Weight loading utilities for IndexTTS-2.5 checkpoints.

IndexTTS-2.5 ships several weight files with different on-disk conventions:

============================  =======================  ===================================
file                          load convention          top-level container
============================  =======================  ===================================
gpt.pth                       torch.load (state_dict)  **none** — flat 456-key state_dict
codec.pth                     torch.load               ckpt['model']  (drop iter/opt/lr)
s2mel.pth                     torch.load               ckpt['net'] = {cfm, length_reg,
                                                        gpt_layer}  (each a submodule dict)
hf_cache/w2v-bert-2.0/
    conformer_shaw.pt         torch.load               flat state_dict (HF naming)
bigvgan/bigvgan_generator.pt  torch.load               flat state_dict
campplus_cn_common.bin        torch.load               flat state_dict (ECAPA-TDNN)
feat1.pt / feat2.pt           torch.load               tensors ([73,192] / [73,1280])
wav2vec2bert_stats.pt         torch.load               {mean:[1024], var:[1024]}
============================  =======================  ===================================

We never import ``transformers`` / ``modelscope``. Everything is plain
``torch.load`` (``weights_only=False`` for checkpoints with optimizer state).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# Default weights root (data disk, per AGENTS.md).
DEFAULT_WEIGHTS_DIR = Path("/root/IndexTTS-2.5")


class WeightLoader:
    """Loads the raw IndexTTS-2.5 weight files into torch tensors/state_dicts.

    Centralizes the on-disk quirks (container keys, file paths) so every
    ``nn.Module.load_state_dict`` call site reads from one place. Each loader
    returns tensors on CPU (fp32 unless noted); the model decides where/how to
    cast (e.g. selective bf16 for S2Mel).
    """

    def __init__(self, weights_dir: str | Path = DEFAULT_WEIGHTS_DIR) -> None:
        self.dir = Path(weights_dir)

    # ----- paths ------------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.dir / name

    def hf_path(self, *parts: str) -> Path:
        return self.dir / "hf_cache" / Path(*parts)

    # ----- raw torch.load helpers ------------------------------------------

    @staticmethod
    def _load_torch(path: Path) -> Any:
        # codec.pth / s2mel.pth carry optimizer state and datetime objects →
        # weights_only=False. gpt.pth is a plain OrderedDict (either works).
        return torch.load(path, map_location="cpu", weights_only=False)

    # ----- model state_dicts ------------------------------------------------

    def load_gpt(self) -> dict[str, torch.Tensor]:
        """GPT-AR (UnifiedVoice). Flat 456-key state_dict, no container."""
        return self._load_torch(self.path("gpt.pth"))

    def load_codec(self) -> dict[str, torch.Tensor]:
        """EnhancedCodec. Training checkpoint → take ckpt['model']."""
        ckpt = self._load_torch(self.path("codec.pth"))
        assert isinstance(ckpt, dict) and "model" in ckpt, (
            f"codec.pth missing 'model' key; got keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
        return ckpt["model"]

    def load_s2mel(self) -> dict[str, dict[str, torch.Tensor]]:
        """S2Mel-CFM. Returns {cfm, length_regulator, gpt_layer}.

        - cfm:        {'estimator': <256-key DiT state_dict>}
        - length_reg: {mask_token, model, embedding, content_in_proj}
        - gpt_layer:  Sequential of 3 × (norm + linear)
        """
        ckpt = self._load_torch(self.path("s2mel.pth"))
        assert isinstance(ckpt, dict) and "net" in ckpt, (
            f"s2mel.pth missing 'net' key; got keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
        net = ckpt["net"]
        for sub in ("cfm", "length_regulator", "gpt_layer"):
            assert sub in net, f"s2mel net missing '{sub}': {list(net.keys())}"
        return net

    def load_w2v_bert(self) -> dict[str, torch.Tensor]:
        """w2v-bert-2.0 conformer. File conformer_shaw.pt holds ``{'model': sd}``
        with 781 keys using modelscope/Seamless-style naming
        (``encoder_frontend.model_dim_proj``, ``encoder.layers.{i}.*``).

        We strip the outer ``model`` container; any HF-key remapping needed to
        feed our hand-written conformer happens in the model adapter, not here.
        """
        d = self._load_torch(self.hf_path("w2v-bert-2.0", "conformer_shaw.pt"))
        assert isinstance(d, dict) and "model" in d, (
            f"conformer_shaw.pt missing 'model' key; got keys={list(d.keys()) if isinstance(d, dict) else type(d)}"
        )
        inner = d["model"]
        assert isinstance(inner, dict), "conformer_shaw.pt['model'] is not a state_dict"
        return inner

    def load_bigvgan(self) -> dict[str, torch.Tensor]:
        """BigVGAN generator. File holds ``{'generator': {'model': sd}}``;
        we unwrap both layers and return the flat generator state_dict."""
        d = self._load_torch(self.hf_path("bigvgan", "bigvgan_generator.pt"))
        cur = d
        # unwrap {'generator': ...} and/or {'model': ...} containers
        while isinstance(cur, dict) and len(cur) == 1 and isinstance(next(iter(cur.values())), dict):
            inner = next(iter(cur.values()))
            # only unwrap if inner values are tensors (true state_dict), else stop
            if all(isinstance(t, torch.Tensor) for t in inner.values()):
                cur = inner
                break
            cur = inner
        assert isinstance(cur, dict), "bigvgan state_dict unwrap failed"
        return cur

    def load_campplus(self) -> dict[str, torch.Tensor]:
        """CAMPPlus ECAPA-TDNN speaker encoder (feat_dim=80, emb=192)."""
        return self._load_torch(self.hf_path("campplus_cn_common.bin"))

    # ----- small auxiliary tensors -----------------------------------------

    def load_spk_matrix(self) -> torch.Tensor:
        """feat1.pt — spk_matrix, shape [73, 192]."""
        return self._load_torch(self.path("feat1.pt"))

    def load_emo_matrix(self) -> torch.Tensor:
        """feat2.pt — emo_matrix, shape [73, 1280]."""
        return self._load_torch(self.path("feat2.pt"))

    def load_w2v_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """wav2vec2bert_stats.pt → (mean, var), each [1024].

        Normalization convention (infer_v2_5.py:177-179,289):
            feat = (feat - mean) / sqrt(var)
        """
        d = self._load_torch(self.path("wav2vec2bert_stats.pt"))
        assert "mean" in d and "var" in d, f"stats missing mean/var: {list(d.keys())}"
        return d["mean"], d["var"]


# ---------------------------------------------------------------------------
# State-dict key-prefix helpers (used by each module's load_state_dict adapter).
# ---------------------------------------------------------------------------


def slice_prefix(
    sd: dict[str, torch.Tensor], prefix: str, *, strip: bool = True
) -> dict[str, torch.Tensor]:
    """Return only keys starting with ``prefix``.

    If ``strip`` (default), the prefix is removed from returned keys. Useful when
    one checkpoint holds multiple submodules and a model wants only its slice,
    e.g. ``slice_prefix(gpt_sd, 'gpt.')`` for the transformer body.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :] if strip else k] = v
    return out


def filter_present(
    sd: dict[str, torch.Tensor], model_keys: list[str]
) -> dict[str, torch.Tensor]:
    """Drop checkpoint keys not present in ``model_keys`` (for partial loads).

    Used during incremental bring-up when a module is only partially implemented
    but we still want to validate the implemented sub-blocks against real weights.
    """
    present = set(model_keys)
    return {k: v for k, v in sd.items() if k in present}
