"""Emotion conditioning for the GPT-AR emo-reference path (conformer → perceiver → emovec).

Replaces indextts/gpt/conformer_encoder.py + perceiver.py; aligned fp32 to official
(see tests/align/test_emo_conditioning_align.py). """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_mask(lengths, max_len):
    return torch.arange(max_len, dtype=torch.int64, device=lengths.device)[None].expand(
        lengths.size(0), max_len) >= lengths.unsqueeze(-1)  # [B, max_len] True=pad


class _FF(nn.Module):  # per-position 2-layer MLP (SiLU)
    def __init__(self, idim, hidden, activation=nn.SiLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden)
        self.activation = activation
        self.w_2 = nn.Linear(hidden, idim)

    def forward(self, xs):
        return self.w_2(self.activation(self.w_1(xs)))


class _ConvMod(nn.Module):  # conformer conv block: pointwise→GLU→depthwise→norm→pointwise
    def __init__(self, channels, kernel_size=15, activation=nn.SiLU()):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size,
                                        padding=(kernel_size - 1) // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1)
        self.activation = activation

    def forward(self, x, mask_pad):  # x [B,T,C]; mask_pad [B,1,T] True=valid
        x = x.transpose(1, 2).masked_fill(~mask_pad, 0.0)          # [B,C,T]
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.activation(self.norm(self.depthwise_conv(x).transpose(1, 2))).transpose(1, 2)
        return self.pointwise_conv2(x).masked_fill(~mask_pad, 0.0).transpose(1, 2)


class _RelPosEnc(nn.Module):  # sinusoidal pos encoding, returned separately (rel-pos attn)
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.xscale = math.sqrt(d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[None])  # [1, max_len, d_model]

    def forward(self, x):
        return x * self.xscale, self.pe[:, : x.size(1)]


class _RelPosMHA(nn.Module):
    # Transformer-XL rel-pos attention: score = ((q+bias_u)·kᵀ + (q+bias_v)·pᵀ)/√d_k.
    # Folded into ONE SDPA via head-dim concat q'=cat([q+bu,q+bv],-1), k'=cat([k,p],-1):
    # q'·k'ᵀ = (q+bu)·kᵀ + (q+bv)·pᵀ (exact — verified against official).
    def __init__(self, n_head, n_feat, dropout_rate=0.0):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q, self.linear_k, self.linear_v = [nn.Linear(n_feat, n_feat) for _ in range(3)]
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)  # sane defaults; overwritten by checkpoint
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(self, query, key, value, mask, pos_emb):  # mask [B,1,T] True=valid
        b, t = query.size(0), query.size(1)
        q = self.linear_q(query).view(b, t, self.h, self.d_k)
        k = self.linear_k(key).view(b, t, self.h, self.d_k)
        v = self.linear_v(value).view(b, t, self.h, self.d_k)
        p = self.linear_pos(pos_emb).view(1, t, self.h, self.d_k)  # [1,T,h,dk]
        q = torch.cat([q + self.pos_bias_u, q + self.pos_bias_v], -1).transpose(1, 2)  # [B,h,T,2dk]
        k = torch.cat([k, p], -1).transpose(1, 2)
        v = v.transpose(1, 2)
        m = torch.zeros(b, 1, 1, t, device=query.device, dtype=query.dtype).masked_fill(
            ~mask.unsqueeze(1), float("-inf"))  # [B,1,1,T] additive
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=1.0 / math.sqrt(self.d_k))
        return self.linear_out(out.transpose(1, 2).contiguous().view(b, t, self.h * self.d_k))


class _ConfLayer(nn.Module):
    # pre-norm conformer block (normalize_before=True is the checkpoint's only path;
    # macaron_style=False → ff_scale=1.0, single feed_forward per layer)
    def __init__(self, size, attn, ff, conv):
        super().__init__()
        self.self_attn, self.feed_forward, self.conv_module = attn, ff, conv
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        self.norm_conv = nn.LayerNorm(size, eps=1e-5)
        self.norm_final = nn.LayerNorm(size, eps=1e-5)

    def forward(self, x, mask, pos_emb, mask_pad):
        xn = self.norm_mha(x)
        x = x + self.self_attn(xn, xn, xn, mask, pos_emb)
        x = x + self.conv_module(self.norm_conv(x), mask_pad)
        x = x + self.feed_forward(self.norm_ff(x))
        return self.norm_final(x)


class EmoConformerEncoder(nn.Module):
    # conv2d2 subsample: Conv2d(1,512,3,3,s=2) halves T (133→66) and feat dim to 511
    # → Linear(512*511=261632→512); rel_pos; 4 macaron-free blocks
    def __init__(self, input_size=1024, output_size=512, attention_heads=4, linear_units=1024,
                 num_blocks=4, cnn_module_kernel=15):
        super().__init__()
        self._output_size = output_size
        conv_inner = output_size * ((input_size - 1) // 2)  # 261632
        self.embed = nn.Module()
        self.embed.conv = nn.Sequential(nn.Conv2d(1, output_size, 3, stride=2), nn.ReLU())
        self.embed.out = nn.Sequential(nn.Linear(conv_inner, output_size))
        self.embed.pos_enc = _RelPosEnc(output_size)
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        act = nn.SiLU()
        self.encoders = nn.ModuleList(
            _ConfLayer(output_size, _RelPosMHA(attention_heads, output_size, 0.0),
                       _FF(output_size, linear_units, act),
                       _ConvMod(output_size, cnn_module_kernel, act))
            for _ in range(num_blocks))

    def forward(self, xs, xs_lens):  # xs [B,T,1024]; xs_lens [B] → (out [B,T',512], mask [B,1,T'])
        T = xs.size(1)
        masks = ~_pad_mask(xs_lens, T).unsqueeze(1)  # [B,1,T] True=valid
        x = self.embed.conv(xs.unsqueeze(1))          # [B,512,T',511]
        b, c, t, f = x.size()
        x = self.embed.out(x.transpose(1, 2).contiguous().view(b, t, c * f))  # [B,T',512]
        x, pos_emb = self.embed.pos_enc(x)
        masks = masks[:, :, 2::2]  # conv stride-2 sample grid (positions 2,4,..)
        for layer in self.encoders:
            x = layer(x, masks, pos_emb, masks)
        return self.after_norm(x), masks


class _RMSNorm(nn.Module):  # official: F.normalize * dim^0.5 * gamma
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class _FF2(nn.Module):
    # GEGLU FFN; Linear named "0"/"2" so ckpt keys layers.{i}.1.0 / layers.{i}.1.2 match
    def __init__(self, dim, mult=2):
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)  # 1024*2*2/3 = 1365
        self.add_module("0", nn.Linear(dim, dim_inner * 2))
        self.add_module("2", nn.Linear(dim_inner, dim))

    def forward(self, x):
        x, gate = self._modules["0"](x).chunk(2, dim=-1)
        return self._modules["2"](F.gelu(gate) * x)


class _Attn(nn.Module):  # cross-attention, optionally prepending the latent query
    def __init__(self, dim, dim_context, dim_head=64, heads=8, cross_attn_include_queries=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):  # x [B,N,D]; mask [B,J'] True=keep
        h = self.heads
        has_context = context is not None
        context = x if context is None else context
        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)  # [B, 1+J, Dc]
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = q.view(*q.shape[:2], h, -1).transpose(1, 2)  # [B,h,N,dh]
        k = k.view(*k.shape[:2], h, -1).transpose(1, 2)
        v = v.view(*v.shape[:2], h, -1).transpose(1, 2)
        j = k.size(-2)
        m = None
        if mask is not None:
            mb = mask.bool() if mask.dtype == torch.bool else mask > 0
            m = torch.zeros(*mb.shape[:1], 1, 1, j, device=mb.device, dtype=x.dtype).masked_fill(
                ~mb.unsqueeze(1).unsqueeze(1), float("-inf"))  # [B,1,1,J'] additive
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=self.scale)
        out = out.transpose(1, 2).contiguous()
        return self.to_out(out.view(*out.shape[:2], -1))


class EmoPerceiverEncoder(nn.Module):
    # PerceiverResampler: dim=1024, dim_context=512, num_latents=1, heads=4, ff_mult=2, depth=2
    def __init__(self, dim=1024, depth=2, dim_context=512, num_latents=1, dim_head=64, heads=4, ff_mult=2):
        super().__init__()
        self.proj_context = nn.Linear(dim_context, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList(
            nn.ModuleList([
                _Attn(dim=dim, dim_context=dim, dim_head=dim_head, heads=heads,
                      cross_attn_include_queries=True),
                _FF2(dim=dim, mult=ff_mult),
            ]) for _ in range(depth))
        self.norm = _RMSNorm(dim)

    def forward(self, x, mask=None):  # x [B,S,512]; mask [B,1+S] True=keep (latent slot padded)
        batch = x.shape[0]
        x = self.proj_context(x)  # [B,S,1024]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)  # [B,1,1024]
        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)  # [B,1,1024]


def get_emovec(conformer, perceiver, emovec_layer, emo_layer, speech_conditioning_latent, cond_mel_lengths):
    # conformer → perceiver → emovec_layer → emo_layer (GPT.get_emovec, model_v2.py:827)
    seq, mask = conformer(speech_conditioning_latent, cond_mel_lengths)  # [B,T',512], [B,1,T']
    conds = perceiver(seq, F.pad(mask.squeeze(1), (1, 0), value=True))  # [B,1,1024]
    return emo_layer(emovec_layer(conds.squeeze(1)))  # [B,1280]
