"""Weight loading for IndexTTS-2.5 checkpoints. On-disk conventions:

  gpt.pth        torch.load              flat 456-key state_dict (no container)
  codec.pth      torch.load              ckpt['model']
  s2mel.pth      torch.load              ckpt['net'] = {cfm, length_regulator, gpt_layer}
  w2v-bert       safetensors             flat HF Wav2Vec2BertModel naming
  bigvgan_generator.pt                   flat state_dict (unwrap {'generator':...})
  campplus_cn_common.bin                 flat state_dict (ECAPA-TDNN)
  feat1.pt/feat2.pt                      tensors [73,192]/[73,1280]
  wav2vec2bert_stats.pt                  {mean:[1024], var:[1024]}

No transformers/modelscope imports; plain torch.load (weights_only=False —
codec/s2mel carry optimizer state).
"""
import os
from pathlib import Path

import torch

# WINDEXTTS_WEIGHTS_DIR env wins, else the canonical data-disk path (pip users
# pass weights_dir= explicitly).
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("WINDEXTTS_WEIGHTS_DIR", "/root/IndexTTS-2.5"))


class WeightLoader:
    """Centralizes on-disk quirks (containers, paths) for every load_state_dict
    call site. Tensors on CPU (fp32 unless noted); the model decides casting.
    """

    def __init__(self, weights_dir: str | Path = DEFAULT_WEIGHTS_DIR) -> None:
        self.dir = Path(weights_dir)

    def path(self, name: str) -> Path:
        return self.dir / name

    def hf_path(self, *parts: str) -> Path:
        return self.dir / "hf_cache" / Path(*parts)

    @staticmethod
    def _load_torch(path: Path):
        # codec/s2mel carry optimizer state + datetime objects -> weights_only=False
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_gpt(self) -> dict[str, torch.Tensor]:
        # flat 456-key state_dict, no container
        return self._load_torch(self.path("gpt.pth"))

    def load_codec(self) -> dict[str, torch.Tensor]:
        # training checkpoint -> ckpt['model']
        ckpt = self._load_torch(self.path("codec.pth"))
        assert isinstance(ckpt, dict) and "model" in ckpt, (
            f"codec.pth missing 'model' key; got keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
        return ckpt["model"]

    def load_s2mel(self) -> dict[str, dict[str, torch.Tensor]]:
        # returns {cfm: {'estimator': <256-key DiT sd>}, length_regulator: {...},
        # gpt_layer: Sequential of 3 x (norm+linear)} from ckpt['net']
        ckpt = self._load_torch(self.path("s2mel.pth"))
        assert isinstance(ckpt, dict) and "net" in ckpt, (
            f"s2mel.pth missing 'net' key; got keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
        net = ckpt["net"]
        for sub in ("cfm", "length_regulator", "gpt_layer"):
            assert sub in net, f"s2mel net missing '{sub}': {list(net.keys())}"
        return net

    def load_w2v_bert(self) -> dict[str, torch.Tensor]:
        # model.safetensors, HF Wav2Vec2BertModel naming (index-tts reads this
        # at infer_v2_5.py:174). conformer_shaw.pt (modelscope naming) also sits
        # in the dir but is NOT used by inference — do not read it.
        from safetensors.torch import load_file
        return load_file(str(self.hf_path("w2v-bert-2.0", "model.safetensors")))

    def load_bigvgan(self) -> dict[str, torch.Tensor]:
        # file holds {'generator': {'model': sd}}; unwrap both levels
        cur = self._load_torch(self.hf_path("bigvgan", "bigvgan_generator.pt"))
        while isinstance(cur, dict) and len(cur) == 1 and isinstance(next(iter(cur.values())), dict):
            inner = next(iter(cur.values()))
            if all(isinstance(t, torch.Tensor) for t in inner.values()):
                cur = inner
                break
            cur = inner
        assert isinstance(cur, dict), "bigvgan state_dict unwrap failed"
        return cur

    def load_campplus(self) -> dict[str, torch.Tensor]:
        # CAMPPlus ECAPA-TDNN (feat_dim=80, emb=192)
        return self._load_torch(self.hf_path("campplus_cn_common.bin"))

    def load_spk_matrix(self) -> torch.Tensor:
        # feat1.pt — spk_matrix [73, 192]
        return self._load_torch(self.path("feat1.pt"))

    def load_emo_matrix(self) -> torch.Tensor:
        # feat2.pt — emo_matrix [73, 1280]
        return self._load_torch(self.path("feat2.pt"))

    def load_w2v_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        # (mean, var) each [1024]; norm is (feat - mean) / sqrt(var)
        # (infer_v2_5.py:177-179,289)
        d = self._load_torch(self.path("wav2vec2bert_stats.pt"))
        assert "mean" in d and "var" in d, f"stats missing mean/var: {list(d.keys())}"
        return d["mean"], d["var"]


# State-dict key-prefix helpers (used by modules' load_state_dict adapters).


def slice_prefix(
    sd: dict[str, torch.Tensor], prefix: str, *, strip: bool = True
) -> dict[str, torch.Tensor]:
    # keys starting with ``prefix``; stripped if ``strip`` (e.g. slice_prefix(sd, 'gpt.'))
    return {k[len(prefix):] if strip else k: v for k, v in sd.items() if k.startswith(prefix)}


def filter_present(
    sd: dict[str, torch.Tensor], model_keys: list[str]
) -> dict[str, torch.Tensor]:
    # drop checkpoint keys absent from model_keys (partial loads during bring-up)
    present = set(model_keys)
    return {k: v for k, v in sd.items() if k in present}
