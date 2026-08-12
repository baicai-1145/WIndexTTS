"""InterpolateRegulator (S2Mel length regulator) — pure-torch re-implementation.

Re-implements ``indextts/s2mel/modules/length_regulator.py`` (InterpolateRegulator)
with zero indextts/transformers dependencies. This is the S2Mel-CFM first stage:
it length-regulates the codec latent ``[B, T, 1024]`` (continuous, is_discrete=False)
to the target duration ``[B, ylens, 512]`` which becomes the DiT condition.

Forward (continuous branch, eval mode — the only path used in IndexTTS-2.5 inference):
    x = content_in_proj(x)                      # [B, T, 1024] -> [B, T, 512]
    mask = sequence_mask(ylens)[..., None]      # [B, ylens.max, 1]
    x = F.interpolate(x.T, size=ylens.max(), mode='nearest')   # -> [B, 512, ylens.max]
    out = model(x).transpose(1, 2)              # 4x(Conv1d3+GroupNorm1+Mish) + Conv1d1
    return out * mask, ylens, None, None, None  # (inference takes [0])

Key fidelity points vs the official code:
  - ``nn.GroupNorm(1, channels)`` (NOT LayerNorm) — groups=1 normalizes over all
    512 channels per time step; state_dict keys/shapes match either way, but the
    numerics are GroupNorm's (eps=1e-5).
  - ``nn.Mish`` between the conv+norm triplets (no parameters).
  - ``F.interpolate(mode='nearest')`` exactly — never 'linear'.
  - ``embedding`` / ``mask_token`` are registered for strict state_dict load but
    unused in the continuous path (is_discrete=False).
  - vector_quantize / f0_condition are False in IndexTTS-2.5 config; kept as
    constructor options (with official behavior) but unused by default.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InterpolateRegulator"]


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Boolean mask [B, max_length] where mask[b, i] = i < length[b].

    Mirrors indextts/s2mel/modules/commons.py:sequence_mask.
    """
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        channels: int,
        sampling_ratios: Sequence[int],
        is_discrete: bool = False,
        in_channels: Optional[int] = None,  # only applies to continuous input
        vector_quantize: bool = False,  # only applies to continuous input
        codebook_size: int = 1024,  # for discrete only
        out_channels: Optional[int] = None,
        groups: int = 1,
        n_codebooks: int = 1,
        quantizer_dropout: float = 0.0,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ):
        super().__init__()
        self.sampling_ratios = list(sampling_ratios)
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(self.sampling_ratios) > 0:
            self.interpolate = True
            for _ in self.sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        else:
            self.interpolate = False
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        self.n_codebooks = n_codebooks
        if n_codebooks > 1:
            self.extra_codebooks = nn.ModuleList(
                [nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1)]
            )
            self.extra_codebook_mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, channels)) for _ in range(n_codebooks - 1)]
            )
        self.quantizer_dropout = quantizer_dropout

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.n_f0_bins = n_f0_bins
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        else:
            self.f0_condition = False

        if not is_discrete:
            assert in_channels is not None, "in_channels required for continuous input"
            self.content_in_proj = nn.Linear(in_channels, channels)
            if vector_quantize:
                # IndexTTS-2.5 does not use this; raise instead of importing
                # the DAC VectorQuantize dependency we don't ship.
                raise NotImplementedError(
                    "vector_quantize=True is not supported by WIndexTTS"
                )

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    def load_official(self, sd: dict[str, torch.Tensor]) -> None:
        """Load the official s2mel length_regulator state_dict (22 keys, strict).

        Args:
            sd: ``ckpt['net']['length_regulator']`` from s2mel.pth.
        """
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"missing keys loading InterpolateRegulator: {missing}")
        if unexpected:
            raise RuntimeError(f"unexpected keys loading InterpolateRegulator: {unexpected}")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        ylens: Optional[torch.Tensor] = None,
        n_quantizers: Optional[int] = None,
        f0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        """Length-regulate ``x`` to ``ylens``.

        Continuous input path (is_discrete=False) — as used in IndexTTS-2.5.

        Args:
            x: [B, T, in_channels] continuous latent (e.g. S_infer or w2v feat).
            ylens: [B] long tensor of target lengths (inference: single element).
            n_quantizers: unused in the continuous branch (kept for API parity).
            f0: unused (f0_condition=False).

        Returns:
            (out * mask [B, ylens.max(), channels], ylens, None, None, None)
        """
        # apply token drop (training-only; eval keeps all)
        if self.training:
            n_quantizers_t = torch.ones((x.shape[0],), device=x.device) * self.n_codebooks
            dropout = torch.randint(1, self.n_codebooks + 1, (x.shape[0],))
            n_dropout = int(x.shape[0] * self.quantizer_dropout)
            n_quantizers_t[:n_dropout] = dropout[:n_dropout]
        else:
            n_quantizers_t = torch.ones((x.shape[0],), device=x.device) * (
                self.n_codebooks if n_quantizers is None else n_quantizers
            )

        if self.is_discrete:
            if self.n_codebooks > 1:
                assert x.ndim == 3
                x_emb = self.embedding(x[:, 0])
                for i, emb in enumerate(self.extra_codebooks):
                    x_emb = x_emb + (n_quantizers_t > i + 1)[..., None, None] * emb(x[:, i + 1])
                x = x_emb
            elif self.n_codebooks == 1:
                x = self.embedding(x[:, 0] if x.ndim == 3 else x)
        else:
            x = self.content_in_proj(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        if self.interpolate:
            x = F.interpolate(x.transpose(1, 2).contiguous(), size=int(ylens.max()), mode="nearest")
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, : x.size(2), :]
            ylens = ylens.clamp(max=x.size(2)).long()

        if self.f0_condition:
            # IndexTTS-2.5 config sets f0_condition=False; the official branch is
            # implemented for completeness but never exercised here.
            raise NotImplementedError("f0_condition is not exercised by IndexTTS-2.5")

        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens, None, None, None


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")
    from windextts.weights import WeightLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = WeightLoader()
    sd = w.load_s2mel()["length_regulator"]
    m = InterpolateRegulator(
        channels=512,
        sampling_ratios=[1, 1, 1, 1],
        is_discrete=False,
        in_channels=1024,
        codebook_size=2048,
    ).to(dev)
    m.load_official(sd)
    m.eval()
    print(f"loaded {sum(p.numel() for p in m.parameters())/1e6:.3f}M params, strict OK")

    S_infer = torch.load("/root/windextts_dumps/s2mel.S_infer.pt", weights_only=False).to(dev)
    ylens = torch.LongTensor([int(S_infer.shape[1] * 1.72)]).to(dev)
    with torch.no_grad():
        cond = m(S_infer, ylens=ylens, n_quantizers=3)[0]
    print(f"cond: {tuple(cond.shape)}")

    ref_cond = torch.load("/root/windextts_dumps/s2mel.cond.pt", weights_only=False).to(dev)
    d = (cond.float() - ref_cond.float()).abs().max().item()
    print(f"max_abs_diff vs official cond = {d:.3e}")
    print(f"allclose(atol=1e-4, rtol=1e-3) = {torch.allclose(cond.float(), ref_cond.float(), atol=1e-4, rtol=1e-3)}")
    print("SMOKE", "OK" if d < 1e-4 else "FAIL")
