"""EnhancedCodec (Amphion semantic codec) — pure-torch re-implementation.

Replaces ``indextts.codec.models.EnhancedCodec`` for the IndexTTS-2.5 semantic
codec path, with zero dependency on ``indextts`` / ``transformers`` / ``vocos`` /
``einops``. Only ``torch.nn`` ops — zero JIT-compile dependency (Windows-friendly).

Numerical behavior is a faithful copy of the official Amphion code (MIT license):
- ``models.py`` (EnhancedCodec: down/up Conv1d + VocosBackbone encoder/decoder + ResidualVQ)
- ``amphion_codec/quantize/factorized_vector_quantize.py`` (FactorizedVectorQuantize: fvq)
- ``amphion_codec/quantize/residual_vq.py`` (ResidualVQ)
- ``kmeans/vocos.py`` (VocosBackbone + ConvNeXtBlock, LayerNorm eps=1e-6, GELU)

Seams (verified shapes, ref /root/windextts_dumps):
  quantize(x[1,T,1024]) -> (codes[1,T//2] int64 in 0..8191, feat[1,T//2,1024])
  decode(codes[1,T//2]) -> latent[1,T,1024]

Key details that must match exactly (any mismatch flips the VQ codebook index
selection and produces completely wrong codes):
- down: Conv1d(1024,1024,k=3,stride=2,pad=1) + GELU; up: Conv1d(1024,1024,k=3,stride=1,pad=1)
- encoder/decoder: VocosBackbone(input_channels=1024, dim=384, intermediate_dim=2048,
  num_layers=12, adanorm_num_embeddings=None) followed by Linear(384,1024)
- ConvNeXtBlock: dwconv k=7 groups=dim pad=3, LayerNorm(eps=1e-6), GELU (exact erf),
  pwconv1/pwconv2 Linear, gamma = layer_scale_init_value = 1/num_layers
- FactorizedVectorQuantize: in_project/out_project are WEIGHT-NORMED Conv1d(1x1)
  (classic torch.nn.utils.weight_norm → state_dict keys weight_g/weight_v);
  codebook Embedding(8192,8); decode_latents L2-normalizes both encodings and
  codebook then argmax(-dist) for indices; z_q = raw codebook emb (un-normalized)
  after straight-through (z_e + (z_q-z_e).detach()); out_project(z_q).
- quantize() does NOT clip odd-length inputs (unlike forward() which does for the
  reconstruction path). down with stride=2 maps T=303 -> 152.
- decode(): unsqueeze codes [B,T]->[1,B,T], vq2emb (embedding lookup -> out_project),
  decoder, then F.interpolate(scale_factor=2, mode="nearest") + up conv.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

__all__ = ["EnhancedCodec"]


# ---------------------------------------------------------------------------
# VocosBackbone (ConvNeXt blocks) — from kmeans/vocos.py
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted to 1D audio (kmeans/vocos.py).

    dwconv(depthwise k=7, groups=dim) -> LayerNorm(eps=1e-6) -> pwconv1 Linear
    -> GELU -> pwconv2 Linear -> *gamma -> + residual.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        assert adanorm_num_embeddings is None, "AdaLayerNorm not used by IndexTTS-2.5 codec"
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return residual + x


class VocosBackbone(nn.Module):
    """ConvNeXt backbone (kmeans/vocos.py) preserving temporal resolution.

    Input [B, C, T], output [B, T, dim].
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        assert adanorm_num_embeddings is None, "AdaLayerNorm not used by IndexTTS-2.5 codec"
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)  # [B, C, T] -> [B, dim, T]
        x = self.norm(x.transpose(1, 2))  # [B, T, dim]
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))  # [B, T, dim]
        return x


# ---------------------------------------------------------------------------
# FactorizedVectorQuantize — from amphion_codec/quantize/factorized_vector_quantize.py
# ---------------------------------------------------------------------------


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float = 0.005,
        codebook_loss_weight: float = 1.0,
        use_l2_normlize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize

        if self.input_dim != self.codebook_dim:
            # WNConv1d: classic weight_norm so state_dict keys are weight_g/weight_v.
            self.in_project = weight_norm(nn.Conv1d(self.input_dim, self.codebook_dim, kernel_size=1))
            self.out_project = weight_norm(nn.Conv1d(self.codebook_dim, self.input_dim, kernel_size=1))
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z: torch.Tensor):
        # z: [B, D, T]
        z_e = self.in_project(z)  # [B, codebook_dim, T]
        z_q, indices = self.decode_latents(z_e)

        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()  # straight-through
        z_q = self.out_project(z_q)  # [B, input_dim, T]
        return z_q, commit_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        # einops rearrange(latents, "b d t -> (b t) d")
        encodings = latents.transpose(1, 2).reshape(-1, self.codebook_dim)
        codebook = self.codebook.weight

        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )  # [(b*t), codebook_size]
        indices = (-dist).max(1)[1].reshape(latents.size(0), latents.size(2))
        z_q = self.decode_code(indices)
        return z_q, indices

    def vq2emb(self, vq: torch.Tensor, out_proj: bool = True) -> torch.Tensor:
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb


# ---------------------------------------------------------------------------
# ResidualVQ — from amphion_codec/quantize/residual_vq.py (fvq path only)
# ---------------------------------------------------------------------------


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "fvq",
        quantizer_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        assert quantizer_type == "fvq", f"IndexTTS-2.5 codec uses fvq, got {quantizer_type}"
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type
        self.quantizer_dropout = quantizer_dropout
        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantize(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None):
        quantized_out = 0.0
        residual = z
        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            z_q_i, commit_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            quantized_out = quantized_out + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commit_loss_i = (commit_loss_i * mask).mean()
            codebook_loss_i = (codebook_loss_i * mask).mean()
            all_commit_losses.append(commit_loss_i)
            all_codebook_losses.append(codebook_loss_i)
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        all_commit_losses, all_codebook_losses, all_indices, all_quantized = map(
            torch.stack, (all_commit_losses, all_codebook_losses, all_indices, all_quantized)
        )
        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )

    def vq2emb(self, vq: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        quantized_out = 0.0
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        for idx, quantizer in enumerate(self.quantizers):
            if idx >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[idx])
        return quantized_out


# ---------------------------------------------------------------------------
# EnhancedCodec — from models.py
# ---------------------------------------------------------------------------


class EnhancedCodec(nn.Module):
    def __init__(
        self,
        codebook_size: int = 8192,
        hidden_size: int = 1024,
        codebook_dim: int = 8,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        vocos_num_layers: int = 12,
        num_quantizers: int = 1,
        downsample_scale: int = 2,
        cfg=None,
    ):
        super().__init__()

        def pick(name: str, default):
            return getattr(cfg, name) if cfg is not None and hasattr(cfg, name) else default

        codebook_size = pick("codebook_size", codebook_size)
        codebook_dim = pick("codebook_dim", codebook_dim)
        hidden_size = pick("hidden_size", hidden_size)
        vocos_dim = pick("vocos_dim", vocos_dim)
        vocos_intermediate_dim = pick("vocos_intermediate_dim", vocos_intermediate_dim)
        vocos_num_layers = pick("vocos_num_layers", vocos_num_layers)
        num_quantizers = pick("num_quantizers", num_quantizers)
        downsample_scale = pick("downsample_scale", downsample_scale)

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale

        if self.downsample_scale is not None and self.downsample_scale > 1:
            self.down = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1)
            self.up = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1)

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

    # ----- official forward (training/reconstruction) ----------------------

    def forward(self, x: torch.Tensor):
        feat = x
        length = x.size(1)
        if length % 2 != 0:
            # 去掉最后一帧 (clip odd trailing frame for the reconstruction path)
            x = x[:, :-1, :]
            feat = feat[:, :-1, :]
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized_out, all_indices, all_commit_losses, all_codebook_losses, _ = self.quantizer(x)
        x = self.decoder(quantized_out)
        x_rec = x
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x_rec = self.up(x).transpose(1, 2)
        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        reconstruction_loss = F.mse_loss(x_rec, feat)
        return x_rec, codebook_loss, all_indices, reconstruction_loss

    # ----- inference seams -------------------------------------------------

    def quantize(self, x: torch.Tensor):
        """x [B, T, D] -> (codes [B, T//2] int64 (squeezed if B==1), feat [B, T//2, D]).

        NOTE: does NOT clip odd-length inputs (unlike forward). down(stride=2) maps
        303 -> 152 directly.
        """
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized_out, all_indices, all_commit_losses, all_codebook_losses, _ = self.quantizer(x)
        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, T] (or [N, B, T]) -> latent [B, T*2, D]."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # [B, T] -> [1, B, T]
        quantized_out = self.quantizer.vq2emb(codes)  # [B, D, T]
        x = self.decoder(quantized_out)  # [B, T, D]
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x_rec = self.up(x).transpose(1, 2)
        return x_rec

    # ----- weight loading --------------------------------------------------

    def load_official(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Strict load of the official codec.pth['model'] state dict (243 keys)."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"codec strict load failed: missing={missing[:8]}... unexpected={unexpected[:8]}..."
            )
        self.eval()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")
    from windextts.weights import WeightLoader

    model = EnhancedCodec()
    model.load_official(WeightLoader().load_codec())
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {n/1e6:.1f}M params, strict OK (243 keys)")

    x = torch.randn(1, 303, 1024)
    with torch.no_grad():
        codes, feat = model.quantize(x)
        rec = model.decode(codes)
    print(f"quantize: codes {tuple(codes.shape)} {codes.dtype} range {codes.min().item()}-{codes.max().item()}")
    print(f"quantize: feat {tuple(feat.shape)}")
    print(f"decode:   rec  {tuple(rec.shape)}")
    print("SMOKE OK")
