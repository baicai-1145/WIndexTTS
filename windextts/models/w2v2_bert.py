# w2v-bert-2.0 conformer (pure torch, replaces transformers.Wav2Vec2BertModel for the
# ref-audio feature path): 24-layer macaron conformer, hidden 1024, 16 heads (head 64),
# intermediate 4096, depthwise conv k31, relative_key pos emb (NO explicit pos embedding —
# embed_positions=None; position info lives only in attention's distance_embedding).
# Seam: hidden_states[17] (index 0 = feature_projection out) — infer_v2_5.py:288.
import torch, torch.nn as nn, torch.nn.functional as F


class Wav2Vec2BertFeedForward(nn.Module):
    def __init__(self, hs=1024, is_=4096):
        super().__init__()
        self.intermediate_dense = nn.Linear(hs, is_)
        self.output_dense = nn.Linear(is_, hs)
    def forward(self, x):
        return self.output_dense(F.silu(self.intermediate_dense(x)))  # hidden_act="swish"


class Wav2Vec2BertSelfAttention(nn.Module):  # relative_key: scores = Q@K^T/sqrt(d) + einsum(Q, dist_emb)/sqrt(d)
    def __init__(self, hs=1024, heads=16, left=64, right=8):
        super().__init__()
        self.heads, self.left, self.right = heads, left, right
        self.distance_embedding = nn.Embedding(left + right + 1, hs // heads)  # 73 pos x head 64
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = [nn.Linear(hs, hs) for _ in range(4)]
    def forward(self, h, attention_mask=None):
        B, T, _ = h.shape
        H, d = self.heads, self.linear_q.out_features // self.heads
        q, k, v = [l(h).view(B, T, H, d).transpose(1, 2) for l in (self.linear_q, self.linear_k, self.linear_v)]  # B,H,T,d
        s = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        dist = torch.clamp(torch.arange(T, device=h.device)[None, :] - torch.arange(T, device=h.device)[:, None],
                           -self.left, self.right)  # distance j(key)-i(query), [Tq,Tk]
        s = s + torch.einsum("bhld,lrd->bhlr", q, self.distance_embedding(dist + self.left).to(q.dtype)) / (d ** 0.5)
        if attention_mask is not None: s = s + attention_mask
        return self.linear_out(torch.matmul(s.softmax(-1), v).transpose(1, 2).reshape(B, T, -1))


class Wav2Vec2BertConvolutionModule(nn.Module):
    def __init__(self, hs=1024, k=31, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hs, eps=eps)
        self.pointwise_conv1 = nn.Conv1d(hs, 2 * hs, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(hs, hs, k, groups=hs, bias=False)
        self.depthwise_layer_norm = nn.LayerNorm(hs, eps=eps)
        self.pointwise_conv2 = nn.Conv1d(hs, hs, 1, bias=False)
    def forward(self, x, conv_attention_mask=None):
        if conv_attention_mask is not None:
            x = x.masked_fill(~conv_attention_mask.bool().unsqueeze(-1), 0.0)
        x = self.layer_norm(x).transpose(1, 2)  # B,T,hs -> B,hs,T
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(F.pad(x, (self.depthwise_conv.kernel_size[0] - 1, 0)))  # causal left-pad k-1 (HF:213-215)
        return self.pointwise_conv2(F.silu(self.depthwise_layer_norm(x.transpose(1, 2)).transpose(1, 2))).transpose(1, 2)


class Wav2Vec2BertEncoderLayer(nn.Module):
    def __init__(self, hs=1024, is_=4096, heads=16, k=31, left=64, right=8, eps=1e-5):
        super().__init__()
        self.ffn1, self.ffn2 = [Wav2Vec2BertFeedForward(hs, is_) for _ in range(2)]
        self.ffn1_layer_norm, self.self_attn_layer_norm, self.ffn2_layer_norm, self.final_layer_norm = [nn.LayerNorm(hs, eps=eps) for _ in range(4)]
        self.self_attn = Wav2Vec2BertSelfAttention(hs, heads, left, right)
        self.conv_module = Wav2Vec2BertConvolutionModule(hs, k, eps)
    def forward(self, x, attention_mask=None, conv_attention_mask=None):
        # macaron: half-residual FFN -> attn -> conv -> half-residual FFN + final LN
        x = x + self.ffn1(self.ffn1_layer_norm(x)) * 0.5
        x = x + self.self_attn(self.self_attn_layer_norm(x), attention_mask)
        x = x + self.conv_module(x, conv_attention_mask)
        return self.final_layer_norm(x + self.ffn2(self.ffn2_layer_norm(x)) * 0.5)


class Wav2Vec2BertFeatureProjection(nn.Module):
    def __init__(self, inp=160, hs=1024, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(inp, eps=eps)
        self.projection = nn.Linear(inp, hs)
    def forward(self, x):
        return self.projection(self.layer_norm(x))


class Wav2Vec2BertConformer(nn.Module):
    def __init__(self, hs=1024, layers=24, heads=16, is_=4096, k=31, inp=160, left=64, right=8, eps=1e-5):
        super().__init__()
        self.feature_projection = Wav2Vec2BertFeatureProjection(inp, hs, eps)
        self.encoder_layers = nn.ModuleList(Wav2Vec2BertEncoderLayer(hs, is_, heads, k, left, right, eps) for _ in range(layers))
    def load_official(self, sd):
        # official keys: encoder.layers.{i}.* -> encoder_layers.{i}.*; masked_spec_embed unused (apply_spec_augment=False)
        remapped = {k.replace("encoder.layers.", "encoder_layers.", 1): v for k, v in sd.items() if k != "masked_spec_embed"}
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected: raise RuntimeError(f"w2v-bert state_dict mismatch: missing={missing} unexpected={unexpected}")
    def forward(self, input_features, attention_mask=None, *, return_layer=17):
        h = self.feature_projection(input_features)  # hidden_states[0]
        all_hs = [h] if return_layer is None else None
        if return_layer == 0: return h
        # zero padded-frame features BEFORE the layers — omitting leaks unmasked
        # padded-frame features into attention (~11.0 maxdiff on hidden_states[17]
        # when the featurizer pads an odd tail frame)
        attn_mask = conv_mask = None
        if attention_mask is not None:
            h = h.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            attn_mask = ((1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min).expand(-1, 1, attention_mask.shape[1], attention_mask.shape[1])
            conv_mask = attention_mask
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, attention_mask=attn_mask, conv_attention_mask=conv_mask)
            if all_hs is not None: all_hs.append(h)
            elif return_layer == i + 1: return h
        return all_hs if all_hs is not None else h


