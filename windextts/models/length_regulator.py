"""InterpolateRegulator — S2Mel length regulator, pure-torch (continuous path).

forward(x, ylens, n_quantizers, f0) -> (out*mask, ylens, None, None, None):
  content_in_proj(1024→512) → F.interpolate(nearest) to ylens.max() →
  4×(Conv1d3+GroupNorm1+Mish) + Conv1d1 → mask*transpose. Numerics vs official:
  GroupNorm(1,512) NOT LayerNorm (same keys, different math), nn.Mish,
  nearest-interpolate never 'linear'. embedding/mask_token/extra_codebooks stay
  for strict ckpt load (discrete/multi-codebook paths deleted — train-only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(length, max_length=None):  # commons.py:155 — True where i < length
    if max_length is None:
        max_length = int(length.max())
    return torch.arange(max_length, dtype=length.dtype, device=length.device)[None] < length[:, None]


class InterpolateRegulator(nn.Module):
    def __init__(self, channels=512, sampling_ratios=(1, 1, 1, 1), is_discrete=False, in_channels=None,
                 vector_quantize=False, codebook_size=1024, out_channels=None, groups=1,
                 n_codebooks=1, quantizer_dropout=0.0, f0_condition=False, n_f0_bins=512):
        super().__init__()
        self.sampling_ratios = list(sampling_ratios)
        self.interpolate = len(self.sampling_ratios) > 0
        out_channels = out_channels or channels
        # ckpt keys model.{0,1,3,4,6,7,9,10,12} — Mish at 2,5,8,11 (no params)
        model = [m for _ in self.sampling_ratios
                 for m in (nn.Conv1d(channels, channels, 3, 1, 1), nn.GroupNorm(groups, channels), nn.Mish())]
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)  # discrete path (ckpt compat)
        self.mask_token = nn.Parameter(torch.zeros(1, channels))
        self.is_discrete = is_discrete
        self.n_codebooks = n_codebooks
        if n_codebooks > 1:
            self.extra_codebooks = nn.ModuleList(nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1))
            self.extra_codebook_mask_tokens = nn.ParameterList(nn.Parameter(torch.zeros(1, channels)) for _ in range(n_codebooks - 1))
        self.quantizer_dropout = quantizer_dropout
        self.f0_condition = f0_condition
        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.n_f0_bins = n_f0_bins
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        if not is_discrete:
            assert in_channels is not None, "in_channels required for continuous input"
            self.content_in_proj = nn.Linear(in_channels, channels)
            if vector_quantize:
                raise NotImplementedError("vector_quantize=True is not supported by WIndexTTS")

    def load_official(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"loading InterpolateRegulator: missing={missing} unexpected={unexpected}")

    def forward(self, x, ylens=None, n_quantizers=None, f0=None):
        x = self.content_in_proj(x)
        mask = sequence_mask(ylens).unsqueeze(-1)  # [B, ylens.max, 1]
        if self.interpolate:
            x = F.interpolate(x.transpose(1, 2).contiguous(), size=int(ylens.max()), mode="nearest")
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, :x.size(2), :]
            ylens = ylens.clamp(max=x.size(2)).long()
        if self.f0_condition:
            raise NotImplementedError("f0_condition not exercised by IndexTTS-2.5")
        return self.model(x).transpose(1, 2).contiguous() * mask, ylens, None, None, None
