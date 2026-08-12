"""BigVGAN vocoder (IndexTTS-2.5) — pure-torch re-implementation.

Re-implements ``indextts/s2mel/modules/bigvgan/bigvgan.py`` (the simplified
NVIDIA BigVGAN used by IndexTTS-2.5: no speaker-encoder conditioning, mel in /
audio out) without any dependency on ``indextts`` / ``huggingface_hub`` /
``alias_free_activation`` external libs. Only ``torch.nn`` — zero JIT-compile
dependency (Windows-friendly).

Input:  mel ``[B, 80, T]`` float32
Output: audio ``[B, 1, T * 256]`` float32   (hop_size = 256, upsample 4*4*2*2*2*2)

Numerical behavior is a faithful copy of the official code:
- weight_norm on every conv (default dim=0); the checkpoint stores
  ``weight_g``/``weight_v`` which we load directly. The official inference path
  calls ``remove_weight_norm()`` after loading, but the math is identical to
  keeping the parametrization (both compute ``weight = weight_g * normalize(weight_v)``).
- AMPBlock1 (resblock='1'): 3× (conv1(dilated) + conv2(dilation=1)) with
  SnakeBeta activations interleaved (6 activations per block).
- SnakeBeta with ``alpha_logscale=True``: alpha/beta are stored in log space and
  exponentiated in forward: ``x + (1/beta) * sin(x*alpha)^2``.
- Anti-aliased activation (Activation1d): upsample (conv_transpose1d, ratio 2,
  kernel 12) -> SnakeBeta -> downsample (conv1d stride 2). The kaiser-sinc
  filters are loaded from the checkpoint as buffers (``upsample.filter`` /
  ``downsample.lowpass.filter``) — no runtime filter synthesis needed.
- Final: activation_post -> conv_post(no bias, use_bias_at_final=false) ->
  clamp(-1, 1) (use_tanh_at_final=false).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

__all__ = ["BigVGAN", "BigVGANConfig"]

# ---------------------------------------------------------------------------
# Config (mirrors hf_cache/bigvgan/config.json, only generator-relevant fields)
# ---------------------------------------------------------------------------


class BigVGANConfig(dict):
    """Generator hyperparameters for the IndexTTS-2.5 BigVGAN.

    Defaults match /root/IndexTTS-2.5/hf_cache/bigvgan/config.json exactly.
    """

    def __init__(self, **kwargs):
        defaults = dict(
            resblock="1",
            upsample_rates=[4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes=[8, 8, 4, 4, 4, 4],
            upsample_initial_channel=1536,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            use_tanh_at_final=False,
            use_bias_at_final=False,
            activation="snakebeta",
            snake_logscale=True,
            num_mels=80,
            sampling_rate=22050,
        )
        defaults.update(kwargs)
        super().__init__(defaults)

    @classmethod
    def from_json(cls, path: str) -> "BigVGANConfig":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls().__dict__ or True})


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """HiFi-GAN / BigVGAN padding: int((k*d - d) / 2) ('same' for odd k)."""
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# SnakeBeta activation
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """Snake with separate frequency (alpha) and magnitude (beta) parameters.

    Official math (alpha_logscale=True):
        alpha = exp(alpha_param), beta = exp(beta_param)
        y = x + 1/(beta + 1e-9) * sin(x * alpha)^2
    """

    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


# ---------------------------------------------------------------------------
# Anti-aliased activation (alias-free-torch, Apache-2.0)
# ---------------------------------------------------------------------------


class LowPassFilter1d(nn.Module):
    """Per-channel 1D low-pass (conv1d with groups=C). Filter buffer loaded
    from checkpoint."""

    def __init__(self, kernel_size: int = 12, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", torch.zeros(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return out


class UpSample1d(nn.Module):
    """Transposed-conv upsample with the kaiser-sinc filter (ratio 2)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", torch.zeros(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )
        x = x[..., self.pad_left:-self.pad_right]
        return x


class DownSample1d(nn.Module):
    """Strided low-pass downsample (ratio 2)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter1d(kernel_size=kernel_size, stride=ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample -> act -> downsample. (x: [B,C,T])"""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# AMP (anti-aliased multi-periodicity) residual block
# ---------------------------------------------------------------------------


class AMPBlock1(nn.Module):
    """BigVGAN default residual block (resblock='1').

    Each of the 3 dilation stages: act1 -> conv1(dilated d) -> act2 ->
    conv2(dilation=1), with a residual add per stage.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)  # 6

        act_cls = SnakeBeta if activation == "snakebeta" else None
        if act_cls is None:
            raise NotImplementedError(
                f"activation incorrectly specified: {activation!r}"
            )
        self.activations = nn.ModuleList(
            [
                Activation1d(activation=act_cls(channels, alpha_logscale=snake_logscale))
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


# ---------------------------------------------------------------------------
# BigVGAN generator
# ---------------------------------------------------------------------------


class BigVGAN(nn.Module):
    """BigVGAN vocoder: mel [B,80,T] -> audio [B,1,T*256]."""

    def __init__(self, cfg: BigVGANConfig):
        super().__init__()
        self.cfg = cfg
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        ch0 = cfg["upsample_initial_channel"]

        # Pre-conv
        self.conv_pre = weight_norm(
            nn.Conv1d(cfg["num_mels"], ch0, 7, 1, padding=3)
        )

        if cfg["resblock"] == "1":
            resblock_class = AMPBlock1
        elif cfg["resblock"] == "2":
            raise NotImplementedError("AMPBlock2 not needed for IndexTTS-2.5 (resblock='1')")
        else:
            raise ValueError(f"Incorrect resblock: {cfg['resblock']}")

        # Transposed-conv upsamplers (no anti-aliasing)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.ConvTranspose1d(
                                ch0 // (2**i),
                                ch0 // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = ch0 // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"])
            ):
                self.resblocks.append(
                    resblock_class(ch, k, d, activation=cfg["activation"],
                                   snake_logscale=cfg["snake_logscale"])
                )

        # Post-conv
        ch = ch0 // (2 ** self.num_upsamples)
        act_cls = SnakeBeta if cfg["activation"] == "snakebeta" else None
        if act_cls is None:
            raise NotImplementedError(f"activation incorrectly specified: {cfg['activation']!r}")
        self.activation_post = Activation1d(
            activation=act_cls(ch, alpha_logscale=cfg["snake_logscale"])
        )

        self.use_bias_at_final = cfg.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            nn.Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )
        self.use_tanh_at_final = cfg.get("use_tanh_at_final", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks (average of num_kernels resblocks)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    # ----- weight loading --------------------------------------------------

    def load_official(self, sd: dict[str, torch.Tensor]) -> None:
        """Load the official bigvgan_generator.pt state_dict (783 keys, strict)."""
        missing, unexpected = self.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"BigVGAN load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/root/WIndexTTS")
    from windextts.weights import WeightLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BigVGANConfig()
    m = BigVGAN(cfg)
    m.load_official(WeightLoader().load_bigvgan())
    m = m.to(dev).eval()
    print(f"[BigVGAN] params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M, strict load OK")

    ref_in = torch.load("/root/windextts_dumps/bigvgan.input_mel.pt", weights_only=False).to(dev)
    ref_out = torch.load("/root/windextts_dumps/bigvgan.output_wav.pt", weights_only=False).to(dev)
    with torch.no_grad():
        out = m(ref_in)
    assert out.shape == ref_out.shape, f"shape {out.shape} != ref {ref_out.shape}"
    diff = (out.float() - ref_out.float()).abs().max().item()
    print(f"out {tuple(out.shape)} vs ref {tuple(ref_out.shape)}")
    print(f"max_abs_diff = {diff:.3e}")
    print(f"allclose(atol=1e-3, rtol=1e-3) = {torch.allclose(out.float(), ref_out.float(), atol=1e-3, rtol=1e-3)}")
    print("SMOKE", "OK" if diff < 1e-3 else "FAIL")
