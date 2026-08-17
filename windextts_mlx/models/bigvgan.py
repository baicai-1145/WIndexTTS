# BigVGAN vocoder — MLX port of windextts/models/bigvgan.py (mel [B,80,T] -> audio).
# weight_norm already flattened at conversion; kaiser-sinc filters come from the
# checkpoint buffers ([1,1,k] torch layout, transposed here). All convs NLC-layout.
import mlx.core as mx
import mlx.nn as nn

from windextts_mlx import ops
from windextts_mlx.ops import Seq


def _pad_same(k, d=1):  # HiFi-GAN/BigVGAN 'same'-pad for odd k
    return int((k * d - d) / 2)


class SnakeBeta(nn.Module):
    def __init__(self, C, alpha=1.0, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        z = mx.zeros(C) if alpha_logscale else mx.ones(C)
        self.alpha = z * alpha
        self.beta = z * alpha

    def __call__(self, x):  # [B,L,C] NLC
        a, b = self.alpha[None, None, :], self.beta[None, None, :]
        if self.alpha_logscale:
            a, b = mx.exp(a), mx.exp(b)
        return x + (1.0 / (b + 1e-9)) * mx.sin(x * a) ** 2


class _LPF1d(nn.Module):  # per-channel low-pass (groups=C); filter buffer from ckpt
    def __init__(self, k=12, stride=1):
        super().__init__()
        even = k % 2 == 0
        self.pad_left, self.pad_right, self.stride = k // 2 - int(even), k // 2, stride
        self.filter = mx.zeros((1, 1, k))  # torch buffer layout [1,1,k]

    def __call__(self, x):  # [B,L,C]
        C = x.shape[-1]
        f = mx.broadcast_to(self.filter.transpose(0, 2, 1), (C, self.filter.shape[-1], 1))  # [C,k,1] mlx [o,k,i/g]
        x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode="edge")
        return mx.conv1d(x, f, stride=self.stride, groups=C)


class UpSample1d(nn.Module):  # convT upsample with kaiser-sinc filter (ratio 2)
    def __init__(self, ratio=2, k=12):
        super().__init__()
        pad = k // ratio - 1
        self.ratio, self.stride, self.pad = ratio, ratio, pad
        self.pad_left = pad * ratio + (k - ratio) // 2
        self.pad_right = pad * ratio + (k - ratio + 1) // 2
        self.filter = mx.zeros((1, 1, k))

    def __call__(self, x):  # [B,L,C]
        C = x.shape[-1]
        f = mx.broadcast_to(self.filter.transpose(0, 2, 1), (C, self.filter.shape[-1], 1))  # mlx convT [o,k,i/g]
        x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        y = self.ratio * mx.conv_transpose1d(x, f, stride=self.stride, groups=C)
        return y[:, self.pad_left:-self.pad_right, :]


class DownSample1d(nn.Module):  # keeps the torch 'lowpass' attr name for ckpt key parity
    def __init__(self, ratio=2, k=12):
        super().__init__()
        self.lowpass = _LPF1d(k, stride=ratio)

    def __call__(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):  # anti-aliased act: upsample -> act -> downsample
    def __init__(self, activation, up_ratio=2, down_ratio=2, up_k=12, down_k=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_k)
        self.downsample = DownSample1d(down_ratio, down_k)

    def __call__(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock1(nn.Module):  # 3 stages (act1 conv1(d) act2 conv2(1)) + residual add
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation="snakebeta", snake_logscale=True):
        super().__init__()
        self.convs1 = Seq({str(i): nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_pad_same(kernel_size, d))
                           for i, d in enumerate(dilation)})
        self.convs2 = Seq({str(i): nn.Conv1d(channels, channels, kernel_size, padding=_pad_same(kernel_size, 1))
                           for i in range(len(dilation))})
        self.activations = Seq({str(i): Activation1d(SnakeBeta(channels, alpha_logscale=snake_logscale))
                                for i in range(2 * len(dilation))})

    def __call__(self, x):
        a = self.activations._order
        for i, (c1, c2) in enumerate(zip(self.convs1._order, self.convs2._order)):
            x = getattr(self.convs2, c2)(getattr(self.activations, a[2 * i + 1])(
                getattr(self.convs1, c1)(getattr(self.activations, a[2 * i])(x)))) + x
        return x


class BigVGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_kernels, self.num_upsamples = len(cfg["resblock_kernel_sizes"]), len(cfg["upsample_rates"])
        ch0 = cfg["upsample_initial_channel"]
        self.conv_pre = nn.Conv1d(cfg["num_mels"], ch0, 7, padding=3)
        self.ups = Seq({str(i): Seq({"0": nn.ConvTranspose1d(ch0 // (2 ** i), ch0 // (2 ** (i + 1)), k,
                                                              stride=u, padding=(k - u) // 2)})
                        for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"]))})
        self.resblocks = Seq({str(i * self.num_kernels + j): AMPBlock1(
            ch0 // (2 ** (i + 1)), k, d, cfg["activation"], cfg["snake_logscale"])
            for i in range(len(self.ups))
            for j, (k, d) in enumerate(zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]))})
        ch = ch0 // (2 ** self.num_upsamples)
        self.activation_post = Activation1d(SnakeBeta(ch, alpha_logscale=cfg["snake_logscale"]))
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=cfg.get("use_bias_at_final", True))
        self.use_tanh_at_final = cfg.get("use_tanh_at_final", True)

    def __call__(self, x):  # [B,80,T] -> [B,1,T*256]
        x = self.conv_pre(x.transpose(0, 2, 1))  # [B,T,80] NLC internally
        mx.eval(x)
        for i in range(self.num_upsamples):
            x = getattr(self.ups, str(i))(x)
            x = sum(getattr(self.resblocks, str(i * self.num_kernels + j))(x)
                    for j in range(self.num_kernels)) / self.num_kernels
            mx.eval(x)  # per-stage eval: first-time kernel compile < watchdog
        x = self.conv_post(self.activation_post(x))
        mx.eval(x)
        x = mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1.0, 1.0)
        return x.transpose(0, 2, 1)  # [B,1,T']
