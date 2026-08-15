"""S2Mel-CFM DiT velocity estimator — pure-torch. Replaces
indextts.s2mel.modules.diffusion_transformer.DiT (+ gpt_fast Transformer + WN).

forward(x, prompt_x, x_lens, t, style, cond) -> dphi_dt [B, 80, T]:
  x_in = cat(x, prompt_x, cond_projection(cond), style) (80+80+512+192=864)
  -> cond_x_merge_linear -> Transformer (13 blocks, RoPE, uvit skip)
  -> skip_linear -> conv1 -> WN(t2-cond) + res_projection -> FinalLayer -> conv2.
Official DiT defaults are the only configuration shipped (config-class removed);
all flags that were permanently False/True in the checkpoint path are folded.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

__all__ = ["DiT", "sequence_mask"]


def sequence_mask(length, max_length=None):  # commons.py:155 — True where arange < length
    if max_length is None:
        max_length = int(length.max())
    return torch.arange(max_length, dtype=length.dtype, device=length.device)[None] < length[:, None]


def find_multiple(n, k):
    return n if n % k == 0 else n + k - (n % k)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------- gpt_fast-style Transformer (RoPE, adaLN, swiGLU, uvit) ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # fp16-native: no .float() guard — fp16's 10-bit mantissa is sufficient
        # (verified cosine 0.9997 vs fp32); removes the cast-kernel tax.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model, norm):
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm

    def forward(self, input, embedding=None):
        if embedding is None:
            return self.norm(input)
        w, b = self.project_layer(embedding).chunk(2, dim=-1)
        return w * self.norm(input) + b


def apply_rotary_emb(x, freqs_cis):
    # fp16-native: no .float() guard — mixed fp16×fp32 promote-then-flow halved
    # the cast-kernel count in the DiT forward (verified cosine 0.9997 vs fp32).
    x = x.reshape(*x.shape[:-1], -1, 2)
    f = freqs_cis.view(1, x.size(1), 1, x.size(3), 2)
    return torch.stack([x[..., 0] * f[..., 0] - x[..., 1] * f[..., 1],
                        x[..., 1] * f[..., 0] + x[..., 0] * f[..., 1]], -1).flatten(3)


class Attention(nn.Module):
    def __init__(self, dim, n_head, n_local_heads, head_dim):
        super().__init__()
        self.wqkv = nn.Linear(dim, (n_head + 2 * n_local_heads) * head_dim, bias=False)
        self.wo = nn.Linear(head_dim * n_head, dim, bias=False)
        self.n_head, self.head_dim, self.n_local_heads, self.dim = n_head, head_dim, n_local_heads, dim

    def forward(self, x, freqs_cis, mask, input_pos=None):
        bsz, seqlen, _ = x.shape
        kv = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv, kv, kv], dim=-1)
        q = apply_rotary_emb(q.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        k = apply_rotary_emb(k.view(bsz, seqlen, self.n_local_heads, self.head_dim), freqs_cis)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        r = self.n_head // self.n_local_heads  # MHA: n_local == n_head → r=1
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.wo(y.transpose(1, 2).contiguous().view(bsz, seqlen, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, intermediate_size):
        super().__init__()
        self.w1, self.w3, self.w2 = [nn.Linear(i, o, bias=False) for i, o in
                                     [(dim, intermediate_size), (dim, intermediate_size), (intermediate_size, dim)]]

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head, n_local_heads, head_dim, intermediate_size, norm_eps, uvit):
        super().__init__()
        self.attention = Attention(dim, n_head, n_local_heads, head_dim)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.ffn_norm = AdaptiveLayerNorm(dim, RMSNorm(dim, eps=norm_eps))
        self.attention_norm = AdaptiveLayerNorm(dim, RMSNorm(dim, eps=norm_eps))
        self.uvit_skip_connection = uvit
        if uvit:
            self.skip_in_linear = nn.Linear(dim * 2, dim)

    def forward(self, x, c, input_pos, freqs_cis, mask, skip_in_x=None):
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
        return h + self.feed_forward(self.ffn_norm(h, c))


class Transformer(nn.Module):
    def __init__(self, n_layer, dim, n_head, head_dim, intermediate_size,
                 block_size=16384, rope_base=10000.0, norm_eps=1e-5, uvit_skip_connection=True):
        super().__init__()
        self.n_layer, self.dim, self.n_head, self.head_dim = n_layer, dim, n_head, head_dim
        self.block_size, self.rope_base = block_size, rope_base
        self.layers = nn.ModuleList(
            TransformerBlock(dim, n_head, n_head, head_dim, intermediate_size, norm_eps, uvit_skip_connection)
            for _ in range(n_layer)
        )
        self.norm = AdaptiveLayerNorm(dim, RMSNorm(dim, eps=norm_eps))
        self.freqs_cis = None
        self.causal_mask = None
        self.max_seq_length = self.max_batch_size = -1
        half = n_layer // 2
        # uvit: layers [0,half) emit skips, (half, n) receive them (LIFO)
        self.layers_emit_skip = list(range(half)) if uvit_skip_connection else []
        self.layers_receive_skip = list(range(half + 1, n_layer)) if uvit_skip_connection else []

    def setup_caches(self, max_batch_size, max_seq_length, use_kv_cache=True):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        self.max_seq_length = max_seq_length = find_multiple(max_seq_length, 8)
        self.max_batch_size = max_batch_size
        p = self.norm.project_layer.weight
        # freqs_cis is sized by block_size (16384) — independent of max_seq_length.
        # Build ONCE: rebuilding on max_seq_length growth changes the tensor
        # address and invalidates captured CUDA Graphs (garbage RoPE → brick
        # audio). Root cause of cross-bucket graph corruption.
        if self.freqs_cis is None:
            freqs = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2)[:(self.head_dim // 2)].float() / self.head_dim))
            fc = torch.polar(torch.ones_like(torch.outer(torch.arange(self.block_size).float(), freqs)),
                             torch.outer(torch.arange(self.block_size).float(), freqs))
            self.freqs_cis = torch.stack([fc.real, fc.imag], -1).to(p.dtype).to(p.device)
        self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=p.device))

    def forward(self, x, c, input_pos=None, mask=None):
        assert self.freqs_cis is not None, "call setup_caches first"
        if mask is None:
            mask = self.causal_mask[None, None, input_pos][..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        skips = []
        for i, layer in enumerate(self.layers):
            s = skips.pop() if i in self.layers_receive_skip else None
            x = layer(x, c, input_pos, freqs_cis, mask, s)
            if i in self.layers_emit_skip:
                skips.append(x)
        return self.norm(x, c)


# ---------------- Wavenet (asymmetric reflect padding + weight_norm) ----------------

def pad1d(x, paddings, mode="constant", value=0.0):
    # encodec pad1d: reflect needs length > max(pad) — top up first, trim after
    if mode == "reflect":
        pl, pr = paddings
        extra = max(0, max(pl, pr) - x.shape[-1] + 1)
        if extra:
            x = F.pad(x, (0, extra))
        return F.pad(x, paddings, mode, value)[..., : x.shape[-1] + pl + pr]
    return F.pad(x, paddings, mode, value)


class SConv1d(nn.Module):
    # non-causal asymmetric reflect pad: left = ((k-1)d+1-s) - right; right = pad_total//2
    # NormConv1d nesting keeps the ckpt key layout sconv.conv.conv.weight_g
    def __init__(self, i, o, k, stride=1, dilation=1, groups=1, bias=True, causal=False, norm="none"):
        super().__init__()
        c = nn.Conv1d(i, o, k, stride=stride, dilation=dilation, groups=groups, bias=bias)
        if norm == "weight_norm":
            weight_norm(c)  # in-place: registers weight_g/weight_v on the Conv1d
        self.conv = nn.Module(); self.conv.conv = c
        self.causal = causal

    def forward(self, x):
        c = self.conv.conv
        k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
        ek = (k - 1) * d + 1
        pad_total = ek - s
        # extra pad rounds output frames up: ideal = (ceil(n)-1)*s + (k - pad_total)
        n = (x.shape[-1] - k + pad_total) / s + 1
        extra = int((math.ceil(n) - 1) * s + k - pad_total - x.shape[-1])
        if self.causal:
            x = pad1d(x, (pad_total, extra), mode="reflect")
        else:
            r = pad_total // 2
            x = pad1d(x, (pad_total - r, r + extra), mode="reflect")
        return self.conv.conv(x)


class WN(nn.Module):
    # gated residual stack: in_layers SConv1d(512->1024,k5) + tanh/sigmoid gate
    # conditioned on t2; res_skip 1x1 convs; last layer drops the skip half.
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, causal=False):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels, self.n_layers = hidden_channels, n_layers
        self.in_layers = nn.ModuleList(
            SConv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation_rate ** i, norm="weight_norm", causal=causal)
            for i in range(n_layers))
        self.res_skip_layers = nn.ModuleList(
            SConv1d(hidden_channels, 2 * hidden_channels if i < n_layers - 1 else hidden_channels, 1, norm="weight_norm", causal=causal)
            for i in range(n_layers))
        self.cond_layer = SConv1d(gin_channels, 2 * hidden_channels * n_layers, 1, norm="weight_norm") if gin_channels else None
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask, g=None, **kw):
        out = torch.zeros_like(x)
        hc = self.hidden_channels
        g = self.cond_layer(g) if g is not None else None
        for i in range(self.n_layers):
            a = self.in_layers[i](x)
            if g is None:
                a = a + 0
            else:
                a = a + g[:, i * 2 * hc:(i + 1) * 2 * hc, :]
            a = torch.tanh(a[:, :hc]) * torch.sigmoid(a[:, hc:])  # gated activation
            a = self.drop(a)
            rs = self.res_skip_layers[i](a)
            if i < self.n_layers - 1:
                x = (x + rs[:, :hc]) * x_mask
                out = out + rs[:, hc:]
            else:
                out = out + rs
        return out * x_mask


# ---------------- DiT ----------------

class TimestepEmbedder(nn.Module):
    # sinusoidal (scale 1000, max_period 10000) + 2-layer MLP
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
        self.register_buffer("freqs", torch.exp(-math.log(10000) *
                                                torch.arange(0, frequency_embedding_size // 2, dtype=torch.float32)
                                                / (frequency_embedding_size // 2)))

    def forward(self, t):
        args = 1000 * t[:, None].float() * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
        # cast back to t's dtype: downstream Linear may be fp16 (estimator_fp16_weights)
        return self.mlp(emb.to(t.dtype))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, patch_size * patch_size * out_channels))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Velocity estimator. IndexTTS-2.5 defaults only (research tier)."""

    def __init__(self):
        super().__init__()
        D, A = 512, 8  # hidden_dim, num_heads (config.yaml s2mel.DiT)
        self.transformer = Transformer(
            n_layer=13, dim=D, n_head=A, head_dim=D // A,
            intermediate_size=find_multiple(int(2 * (4 * D) / 3), 256), block_size=16384)
        self.x_embedder = weight_norm(nn.Linear(80, D))
        self.cond_embedder = nn.Embedding(1024, D)  # discrete content path (unused, in ckpt)
        self.cond_projection = nn.Linear(512, D)
        self.t_embedder = TimestepEmbedder(D)
        self.register_buffer("input_pos", torch.arange(16384))
        # wavenet final path (config.yaml s2mel.wavenet)
        self.t_embedder2 = TimestepEmbedder(512)
        self.conv1 = nn.Linear(D, 512)
        self.conv2 = nn.Conv1d(512, 80, 1)
        self.wavenet = WN(512, 5, 1, 8, gin_channels=512, p_dropout=0.2)
        self.final_layer = FinalLayer(512, 1, 512)
        self.res_projection = nn.Linear(D, 512)
        self.content_mask_embedder = nn.Embedding(1, D)
        self.skip_linear = nn.Linear(D + 80, D)
        self.cond_x_merge_linear = nn.Linear(80 + 80 + 512 + 192, D)  # x|prompt_x|cond|style = 864
        self.class_dropout_prob = 0.1

    def load_official(self, sd):
        remapped = {k[len("estimator."):]: v for k, v in sd.items() if k.startswith("estimator.")}
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"loading DiT: missing={missing[:5]}... unexpected={unexpected[:5]}...")

    def setup_caches(self, max_batch_size, max_seq_length):
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)

    def precast_linear_bf16(self):
        # vLLM-Omni strategy: Linear/Conv1d weights -> bf16 (flash-attn eligible),
        # LayerNorm/embeddings stay fp32. Returns param count cast.
        n = 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                for p in m.parameters(recurse=False):
                    p.data = p.data.to(torch.bfloat16)
                    n += p.numel()
        return n

    def forward(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):
        # x/prompt_x [B,80,T] -> [B,T,80]; cond [B,T,512] -> [B,T,D]
        B, _, T = x.size()
        t1 = self.t_embedder(t)                       # [B, D]
        x, prompt_x, cond = x.transpose(1, 2), prompt_x.transpose(1, 2), self.cond_projection(cond)
        x_in = torch.cat([x, prompt_x, cond, style[:, None].expand(B, T, 192)], -1)  # [B,T,864]
        if (self.training and torch.rand(1) < self.class_dropout_prob) or (not self.training and mask_content):
            x_in[..., 80:] = x_in[..., 80:] * 0       # classifier-free guidance dropout of cond+style
        x_in = self.cond_x_merge_linear(x_in)         # [B, T, D]
        x_mask = sequence_mask(x_lens, x_in.size(1)).to(x.device).unsqueeze(1)  # [B,1,T]
        x_mask_expanded = x_mask[:, None].expand(-1, 1, x_in.size(1), -1)  # [B,1,T,T] key-mask broadcast
        x_res = self.transformer(x_in, t1.unsqueeze(1), self.input_pos[:x_in.size(1)], x_mask_expanded)
        x_res = self.skip_linear(torch.cat([x_res, x], -1))  # long skip
        x = self.conv1(x_res).transpose(1, 2)         # [B, 512, T]
        t2 = self.t_embedder2(t)
        x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(x_res)
        return self.conv2(self.final_layer(x, t1).transpose(1, 2))  # [B, 80, T]
