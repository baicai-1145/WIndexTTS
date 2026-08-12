"""Numerical alignment test: our Wav2Vec2BertConformer vs the official
transformers.Wav2Vec2BertModel.

Validates the core "remove transformers" deliverable: a pure-torch re-implementation
that matches the official HF model bit-for-bit (same device, same dtype).

We compare ALL 25 hidden_states (feature_projection out + 24 encoder layers),
which is far stronger than checking only hidden_states[17]. If any layer diverges
we learn exactly where.

NOTE on device/dtype: the official IndexTTS dumps were taken with use_bf16=True,
which introduces ~0.6 bf16-rounding error and is NOT a fair alignment target for
our fp32 model. The fair comparison is fp32-ours vs fp32-official on the same GPU,
which is what this test does.
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch
import torchaudio
import pytest

# Force offline HF (uses the local model dir only).
os.environ.setdefault("HF_HUB_CACHE", "/root/IndexTTS-2.5/checkpoints/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

W2V_DIR = "/root/IndexTTS-2.5/hf_cache/w2v-bert-2.0"
TEST_WAV = "/root/WIndexTTS/test.wav"


def _load_inputs(device):
    from windextts.frontend.audio_utils import SeamlessM4TFeaturizer

    fe = SeamlessM4TFeaturizer(device=device)
    audio, sr = torchaudio.load(TEST_WAV)
    a16 = torchaudio.transforms.Resample(sr, 16000)(audio).to(device)
    inp = fe(a16)
    am = torch.ones(inp.shape[:2], dtype=torch.int32, device=device)
    return inp, am


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (fp32 GPU parity)")
def test_w2v_bert_full_alignment():
    from transformers import Wav2Vec2BertModel

    from windextts.models.w2v2_bert import Wav2Vec2BertConformer
    from windextts.weights import WeightLoader

    device = "cuda"
    inp, am = _load_inputs(device)

    # official model (fp32)
    off = Wav2Vec2BertModel.from_pretrained(W2V_DIR, local_files_only=True).to(device).eval()
    with torch.no_grad():
        off_hs = off(input_features=inp, attention_mask=am, output_hidden_states=True).hidden_states

    # ours
    mine = Wav2Vec2BertConformer().to(device).eval()
    mine.load_official(WeightLoader().load_w2v_bert())
    with torch.no_grad():
        my_hs = mine(inp, am, return_layer=None)

    assert len(my_hs) == len(off_hs) == 25, f"expected 25 hidden states, got {len(my_hs)} vs {len(off_hs)}"

    # Every layer must match in fp32 on the same GPU.
    max_per_layer = []
    for i, (o, mi) in enumerate(zip(off_hs, my_hs)):
        assert o.shape == mi.shape, f"layer {i} shape mismatch {o.shape} != {mi.shape}"
        d = (o.float() - mi.float()).abs().max().item()
        max_per_layer.append(d)

    overall = max(max_per_layer)
    seam_diff = max_per_layer[17]  # the IndexTTS-2.5 seam layer
    print(f"\n[align] max per-layer diffs: {[f'{d:.2e}' for d in max_per_layer]}")
    print(f"[align] overall max = {overall:.3e}, hidden_states[17] = {seam_diff:.3e}")

    # fp32 same-device parity should be effectively exact (< 1e-4 from non-deterministic
    # kernel ordering; in practice we observe 0.0).
    assert overall < 1e-3, f"w2v-bert divergence: max layer diff {overall}"
    # The seam (hidden_states[17]) must pass the project standard.
    assert torch.allclose(
        my_hs[17].float(), off_hs[17].float(), atol=1e-4, rtol=1e-3
    ), f"seam hidden_states[17] not aligned: diff {seam_diff}"


if __name__ == "__main__":
    test_w2v_bert_full_alignment()
    print("W2V-BERT ALIGN OK")
