# EnhancedCodec (Amphion semantic codec) — pure-torch port of indextts.codec.models.
# Only torch.nn, zero JIT-compile dep (Windows). Seams (verified fp32 CUDA vs official):
#   quantize(x[1,T,1024]) -> (codes[1,T//2] i64 0..8191, feat[1,T//2,1024])
#   decode(codes[1,T//2]) -> latent[1,T,1024]
# Numerics that must stay exact (any flip breaks VQ codebook selection):
#   - down Conv1d(k3,s2,p1)+GELU / up Conv1d(k3,s1,p1); VocosBackbone(1024→384, 12
#     ConvNeXt, gamma=1/12, LayerNorm eps=1e-6, exact-erf GELU) + Linear(384,1024)
#   - FVQ in/out_project are classic weight_norm Conv1d(1x1) -> keys weight_g/weight_v
#   - decode_latents L2-normalizes encodings+codebook, argmax(-dist); z_q raw emb,
#     straight-through z_e+(z_q-z_e).detach()
#   - quantize() does NOT clip odd T; down s2 maps 303->152
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value, adanorm_num_embeddings=None):
        super().__init__()
        assert adanorm_num_embeddings is None, "AdaLayerNorm not used by IndexTTS-2.5 codec"
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1, self.act, self.pwconv2 = nn.Linear(dim, intermediate_dim), nn.GELU(), nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        residual = x
        x = self.norm(self.dwconv(x).transpose(1, 2))  # [B,C,T] -> [B,T,C]
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x if self.gamma is not None else x
        return residual + x.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels, dim, intermediate_dim, num_layers, layer_scale_init_value=None, adanorm_num_embeddings=None):
        super().__init__()
        assert adanorm_num_embeddings is None, "AdaLayerNorm not used by IndexTTS-2.5 codec"
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value, None) for _ in range(num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.embed(x)  # [B,C,T] -> [B,dim,T]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # [B,T,dim] -> back
        for b in self.convnext:
            x = b(x)
        return self.final_layer_norm(x.transpose(1, 2))  # [B,T,dim]


class FactorizedVectorQuantize(nn.Module):
    def __init__(self, input_dim, codebook_size, codebook_dim, commitment=0.005, codebook_loss_weight=1.0, use_l2_normlize=True):
        super().__init__()
        # classic weight_norm -> checkpoint keys weight_g/weight_v
        self.in_project, self.out_project = (
            (weight_norm(nn.Conv1d(input_dim, codebook_dim, 1)), weight_norm(nn.Conv1d(codebook_dim, input_dim, 1)))
            if input_dim != codebook_dim else (nn.Identity(), nn.Identity())
        )
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.commitment, self.codebook_loss_weight, self.use_l2_normlize = commitment, codebook_loss_weight, use_l2_normlize
        self.codebook_dim = codebook_dim

    def forward(self, z):
        z_e = self.in_project(z)  # [B, codebook_dim, T]
        z_q, indices = self.decode_latents(z_e)
        commit_loss = codebook_loss = torch.zeros(z.shape[0], device=z.device)
        z_q = z_e + (z_q - z_e).detach()  # straight-through
        return self.out_project(z_q), commit_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = latents.transpose(1, 2).reshape(-1, self.codebook_dim)  # einops "b d t -> (b t) d"
        codebook = self.codebook.weight
        if self.use_l2_normlize:
            encodings, codebook = F.normalize(encodings), F.normalize(codebook)
        dist = encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t()  # [(b*t), cbs]
        indices = (-dist).max(1)[1].reshape(latents.size(0), latents.size(2))  # argmax cosine
        return self.decode_code(indices), indices

    def vq2emb(self, vq, out_proj=True):
        emb = self.decode_code(vq)
        return self.out_project(emb) if out_proj else emb


class ResidualVQ(nn.Module):
    def __init__(self, input_dim=256, num_quantizers=8, codebook_size=1024, codebook_dim=256, quantizer_type="fvq", quantizer_dropout=0.5, **kwargs):
        super().__init__()
        assert quantizer_type == "fvq", f"IndexTTS-2.5 codec uses fvq, got {quantizer_type}"
        self.num_quantizers, self.quantizer_dropout = num_quantizers, quantizer_dropout
        self.quantizers = nn.ModuleList(
            FactorizedVectorQuantize(input_dim, codebook_size, codebook_dim, **kwargs) for _ in range(num_quantizers)
        )

    def forward(self, z, n_quantizers=None):
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        quantized_out, residual = 0.0, z
        all_commit, all_codebook, all_indices, all_quantized = [], [], [], []
        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            z_q_i, commit_loss_i, codebook_loss_i, indices_i, _ = quantizer(residual)
            mask = torch.full((z.shape[0],), i, device=z.device) < n_quantizers
            quantized_out += z_q_i * mask[:, None, None]
            residual -= z_q_i
            all_commit.append((commit_loss_i * mask).mean())
            all_codebook.append((codebook_loss_i * mask).mean())
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)
        return (quantized_out, *map(torch.stack, (all_indices, all_commit, all_codebook, all_quantized)))

    def vq2emb(self, vq, n_quantizers=None):
        n = self.num_quantizers if n_quantizers is None else n_quantizers
        out = 0.0
        for i, q in enumerate(self.quantizers[:n]):
            out += q.vq2emb(vq[i])
        return out


class EnhancedCodec(nn.Module):
    def __init__(self, codebook_size=8192, hidden_size=1024, codebook_dim=8, vocos_dim=384,
                 vocos_intermediate_dim=2048, vocos_num_layers=12, num_quantizers=1, downsample_scale=2):
        super().__init__()
        if downsample_scale is not None and downsample_scale > 1:
            self.down = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
            self.up = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        vb = lambda: nn.Sequential(
            VocosBackbone(hidden_size, vocos_dim, vocos_intermediate_dim, vocos_num_layers),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.encoder, self.decoder = vb(), vb()
        self.quantizer = ResidualVQ(
            input_dim=hidden_size, num_quantizers=num_quantizers, codebook_size=codebook_size,
            codebook_dim=codebook_dim, quantizer_type="fvq", quantizer_dropout=0.0,
            commitment=0.15, codebook_loss_weight=1.0, use_l2_normlize=True,
        )

    # quantize()/decode() are the only paths used at inference (reconstruction
    # forward is train-only and deleted). quantize() does NOT clip odd T; down
    # s2 maps 303->152.
    def quantize(self, x):
        """x [B,T,D] -> (codes [B,T//2] i64 (squeezed if B==1), feat [B,T//2,D]); no odd clip (unlike forward)."""
        if hasattr(self, "down"):
            x = F.gelu(self.down(x.transpose(1, 2))).transpose(1, 2)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized_out, all_indices, _, _, _ = self.quantizer(x)
        return (all_indices.squeeze(0) if all_indices.shape[0] == 1 else all_indices), quantized_out.transpose(1, 2)

    def decode(self, codes):
        """codes [B,T] (or [N,B,T]) -> latent [B,T*2,D]."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # [B,T] -> [1,B,T]
        x = self.decoder(self.quantizer.vq2emb(codes))  # [B,D,T] -> [B,T,D]
        if hasattr(self, "down"):
            x = self.up(F.interpolate(x.transpose(1, 2), scale_factor=2, mode="nearest")).transpose(1, 2)
        return x

    def load_official(self, state_dict):
        """Strict load of codec.pth['model'] (243 keys)."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"codec strict load failed: missing={missing[:8]}... unexpected={unexpected[:8]}..."
            )
        self.eval()
