"""BigVGAN vocoder (IndexTTS-2.5) — pure torch, mel [B,80,T] -> audio [B,1,T*256].

Faithful copy of indextts/s2mel/modules/bigvgan/bigvgan.py (hop 256, ups 4*4*2*2*2*2).
weight_norm on every conv (checkpoint stores weight_g/weight_v; official calls
remove_weight_norm() after load — math identical since both compute g*v/||v||).
SnakeBeta alpha_logscale=True: alpha/beta stored in log space, exp'd in forward.
Anti-alias act = convT upsample(ratio 2, k12) -> SnakeBeta -> stride-2 conv; the
kaiser-sinc filters load from the checkpoint as buffers (no runtime synthesis).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class BigVGANConfig(dict):
    """Generator hyperparams; defaults = hf_cache/bigvgan/config.json (generator fields)."""

    def __init__(self, **kwargs):
        super().__init__(dict(
            resblock="1", upsample_rates=[4, 4, 2, 2, 2, 2], upsample_kernel_sizes=[8, 8, 4, 4, 4, 4],
            upsample_initial_channel=1536, resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5] for _ in range(3)],
            use_tanh_at_final=False, use_bias_at_final=False, activation="snakebeta",
            snake_logscale=True, num_mels=80, sampling_rate=22050) | kwargs)

    @classmethod
    def from_json(cls, path):
        import json
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def get_padding(k, d=1):  # HiFi-GAN/BigVGAN 'same'-pad for odd k
    return int((k * d - d) / 2)


class SnakeBeta(nn.Module):
    def __init__(self, C, alpha=1.0, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        # log scale: zero-init (alpha=exp(0)=1); linear scale: one-init
        z = torch.zeros(C) if alpha_logscale else torch.ones(C)
        self.alpha, self.beta = nn.Parameter(z * alpha), nn.Parameter(z * alpha)

    def forward(self, x):
        a, b = self.alpha.unsqueeze(0).unsqueeze(-1), self.beta.unsqueeze(0).unsqueeze(-1)  # [1,C,1]
        if self.alpha_logscale:
            a, b = torch.exp(a), torch.exp(b)
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a).pow(2)


class LowPassFilter1d(nn.Module):  # per-channel low-pass (groups=C); filter buffer from checkpoint
    def __init__(self, k=12, stride=1):
        super().__init__()
        even = k % 2 == 0
        self.pad_left, self.pad_right, self.stride = k // 2 - int(even), k // 2, stride
        self.register_buffer("filter", torch.zeros(1, 1, k))

    def forward(self, x):
        C = x.shape[1]
        return F.conv1d(F.pad(x, (self.pad_left, self.pad_right), mode="replicate"),
                        self.filter.expand(C, -1, -1), stride=self.stride, groups=C)


class UpSample1d(nn.Module):  # convT upsample with kaiser-sinc filter (ratio 2)
    def __init__(self, ratio=2, k=12):
        super().__init__()
        pad = k // ratio - 1
        # crop: left = pad*ratio + (k-ratio)//2, right = pad*ratio + (k-ratio+1)//2
        self.ratio, self.stride, self.pad = ratio, ratio, pad
        self.pad_left = pad * ratio + (k - ratio) // 2
        self.pad_right = pad * ratio + (k - ratio + 1) // 2
        self.register_buffer("filter", torch.zeros(1, 1, k))

    def forward(self, x):
        C = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):  # strided low-pass downsample (ratio 2)
    def __init__(self, ratio=2, k=12):
        super().__init__()
        self.lowpass = LowPassFilter1d(k, stride=ratio)

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):  # anti-aliased act: upsample -> act -> downsample
    def __init__(self, activation, up_ratio=2, down_ratio=2, up_k=12, down_k=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_k)
        self.downsample = DownSample1d(down_ratio, down_k)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock1(nn.Module):  # resblock='1': 3 stages (act1 conv1(d) act2 conv2(1)) + residual add
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation="snakebeta", snake_logscale=True):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                                          padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                                          padding=get_padding(kernel_size, 1))) for _ in dilation])
        self.activations = nn.ModuleList([Activation1d(SnakeBeta(channels, alpha_logscale=snake_logscale))
                                          for _ in range(2 * len(dilation))])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]):
            x = c2(a2(c1(a1(x)))) + x
        return x


class BigVGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_kernels, self.num_upsamples = len(cfg["resblock_kernel_sizes"]), len(cfg["upsample_rates"])
        ch0 = cfg["upsample_initial_channel"]
        if cfg["resblock"] == "2":
            raise NotImplementedError("AMPBlock2 not needed for IndexTTS-2.5 (resblock='1')")
        if cfg["resblock"] != "1":
            raise ValueError(f"Incorrect resblock: {cfg['resblock']}")
        if cfg["activation"] != "snakebeta":
            raise NotImplementedError(f"activation incorrectly specified: {cfg['activation']!r}")
        self.conv_pre = weight_norm(nn.Conv1d(cfg["num_mels"], ch0, 7, 1, padding=3))
        self.ups = nn.ModuleList([nn.ModuleList([weight_norm(nn.ConvTranspose1d(
            ch0 // (2 ** i), ch0 // (2 ** (i + 1)), k, u, padding=(k - u) // 2))])
            for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"]))])
        self.resblocks = nn.ModuleList([AMPBlock1(ch0 // (2 ** (i + 1)), k, d,
                                                  cfg["activation"], cfg["snake_logscale"])
                                        for i in range(len(self.ups))
                                        for k, d in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"])])
        ch = ch0 // (2 ** self.num_upsamples)
        self.activation_post = Activation1d(SnakeBeta(ch, alpha_logscale=cfg["snake_logscale"]))
        self.use_bias_at_final = cfg.get("use_bias_at_final", True)
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))
        self.use_tanh_at_final = cfg.get("use_tanh_at_final", True)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for u in self.ups[i]:
                x = u(x)
            x = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)) / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1.0, 1.0)

    def remove_weight_norm(self) -> int:
        """Flatten weight_norm into plain convs (official inference path).

        The weight_norm _forward_pre_hook recomputes w = g*v/||v|| on EVERY
        forward — pure overhead at inference and ~330 conv-dispatch host bubbles
        (profiler: 149ms, 91% conv1d dispatch). Removing pre-flattens the weight
        so convs run the fast unhooked path; math identical.
        """
        from torch.nn.utils import remove_weight_norm as _rwn
        n = 0
        for m in self.modules():
            try:
                _rwn(m)
                n += 1
            except (ValueError, KeyError):
                pass
        return n

    def load_official(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"BigVGAN load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/root/WIndexTTS")
    from windextts.weights import WeightLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = BigVGAN(BigVGANConfig())
    m.load_official(WeightLoader().load_bigvgan())
    m = m.to(dev).eval()
    ref_in = torch.load("/root/windextts_dumps/bigvgan.input_mel.pt", weights_only=False).to(dev)
    ref_out = torch.load("/root/windextts_dumps/bigvgan.output_wav.pt", weights_only=False).to(dev)
    with torch.no_grad():
        out = m(ref_in)
    diff = (out.float() - ref_out.float()).abs().max().item()
    print(f"[BigVGAN] params={sum(p.numel() for p in m.parameters())/1e6:.1f}M shape={tuple(out.shape)} "
          f"max_abs_diff={diff:.3e} allclose={torch.allclose(out.float(), ref_out.float(), atol=1e-3, rtol=1e-3)}")
