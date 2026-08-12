"""Numerical alignment test: our pure-torch EnhancedCodec vs the official
IndexTTS-2.5 codec outputs.

Reference tensors (dumped from the official IndexTTS2 pipeline on CUDA, fp32 —
the semantic_codec stays fp32 even with use_bf16=True since only self.gpt is
cast to bf16):
  input:  /root/windextts_dumps/w2v.cond_emb_normalized.pt   [1, 303, 1024]
  refs:   codec.quantize_code.pt  [1, 152] int64    (semantic codes 0..8191)
          codec.quantize_feat.pt  [1, 152, 1024]    (quantized embeddings)
          codec.decode_latent.pt  [1, 304, 1024]    (decoded latent, S_infer)

The codec path is fully deterministic fp32 (Conv1d/GELU/LayerNorm/normalize/argmax/
nearest-interpolate), so feeding the exact dumped input must reproduce the dumped
outputs bit-for-bit.
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch
import pytest

DUMPS = "/root/windextts_dumps"
INPUT_DUMP = f"{DUMPS}/w2v.cond_emb_normalized.pt"


def _load_dump(name: str) -> torch.Tensor:
    path = name if name.startswith(DUMPS) else f"{DUMPS}/{name}.pt"
    return torch.load(path, weights_only=False)


def _build_our_codec(device):
    from windextts.models.codec import EnhancedCodec
    from windextts.weights import WeightLoader

    model = EnhancedCodec().to(device)
    model.load_official(WeightLoader().load_codec())
    model.eval()
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (dumps are CUDA)")
def test_codec_quantize_alignment():
    device = "cuda"
    model = _build_our_codec(device)

    x = _load_dump(INPUT_DUMP).to(device)  # [1, 303, 1024] fp32
    ref_codes = _load_dump("codec.quantize_code").to(device)  # [1, 152] int64
    ref_feat = _load_dump("codec.quantize_feat").to(device)  # [1, 152, 1024]

    with torch.no_grad():
        codes, feat = model.quantize(x)

    assert codes.shape == ref_codes.shape, f"codes shape {codes.shape} != ref {ref_codes.shape}"
    assert feat.shape == ref_feat.shape, f"feat shape {feat.shape} != ref {ref_feat.shape}"

    # codes must match EXACTLY (argmax of cosine distance — any index flip is a bug)
    codes_match = torch.equal(codes, ref_codes)
    n_flip = int((codes != ref_codes).sum().item()) if not codes_match else 0
    feat_diff = (feat.float() - ref_feat.float()).abs().max().item()
    feat_ok = torch.allclose(feat.float(), ref_feat.float(), atol=1e-4, rtol=1e-3)

    print(f"[align] quantize codes: exact={codes_match} flips={n_flip}/{codes.numel()}")
    print(f"[align] quantize feat:  max_abs_diff={feat_diff:.3e} allclose={feat_ok}")

    assert codes_match, f"quantize codes mismatch at {n_flip} positions (of {codes.numel()})"
    assert feat_ok, f"quantize feat mismatch: max_abs_diff {feat_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (dumps are CUDA)")
def test_codec_decode_alignment():
    device = "cuda"
    model = _build_our_codec(device)

    ref_codes = _load_dump("codec.quantize_code").to(device)
    ref_latent = _load_dump("codec.decode_latent").to(device)  # [1, 304, 1024]

    with torch.no_grad():
        rec = model.decode(ref_codes)

    assert rec.shape == ref_latent.shape, f"decode shape {rec.shape} != ref {ref_latent.shape}"
    diff = (rec.float() - ref_latent.float()).abs().max().item()
    ok = torch.allclose(rec.float(), ref_latent.float(), atol=1e-4, rtol=1e-3)
    print(f"[align] decode latent: max_abs_diff={diff:.3e} allclose={ok}")
    assert ok, f"decode latent mismatch: max_abs_diff {diff}"


if __name__ == "__main__":
    test_codec_quantize_alignment()
    test_codec_decode_alignment()
    print("CODEC ALIGN OK")
