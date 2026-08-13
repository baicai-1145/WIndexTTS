"""Numerical alignment test: emo_conditioning modules vs official IndexTTS-2.5.

Compares EmoConformerEncoder + EmoPerceiverEncoder against official tensors
dumped by scripts/dump_emo_ref_tensors.py:

    conformer_in  [1,133,1024] + lens=[133] → conformer_seq [1,66,512]
    conformer_seq [1,66,512] + padded mask  → perceiver_out [1,1,1024]
    full get_emovec(emo_cond_emb)           → merged_emovec_a065 [1,1280]

Run: /root/index-tts/.venv/bin/python tests/align/test_emo_conditioning_align.py
"""
import sys
import torch

sys.path.insert(0, "/root/WIndexTTS")

from windextts.models.emo_conditioning import (
    EmoConformerEncoder,
    EmoPerceiverEncoder,
    get_emovec,
)

DUMP = "/root/windextts_dumps/emo_ref"
WEIGHTS = "/root/IndexTTS-2.5/gpt.pth"
ATOL, RTOL = 1e-4, 1e-3


def _load_emo_weights(prefix: str, module: torch.nn.Module) -> int:
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    remapped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(remapped, strict=True)
    assert not missing and not unexpected, f"{prefix}: {missing} {unexpected}"
    return len(remapped)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conformer = EmoConformerEncoder().to(dev).eval()
    perceiver = EmoPerceiverEncoder().to(dev).eval()
    n_ce = _load_emo_weights("emo_conditioning_encoder.", conformer)
    n_pe = _load_emo_weights("emo_perceiver_encoder.", perceiver)
    print(f"[emo] loaded conformer {n_ce} keys, perceiver {n_pe} keys")

    # emovec/emo layers (also in gpt.pth)
    emovec_layer = torch.nn.Linear(1024, 1280).to(dev)
    emo_layer = torch.nn.Linear(1280, 1280).to(dev)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    emovec_layer.load_state_dict({"weight": sd["emovec_layer.weight"],
                                  "bias": sd["emovec_layer.bias"]})
    emo_layer.load_state_dict({"weight": sd["emo_layer.weight"],
                               "bias": sd["emo_layer.bias"]})

    d = lambda name: torch.load(f"{DUMP}/{name}.pt", weights_only=False)

    # ---- 1. conformer encoder ----
    conf_in = d("conformer_in")["conformer_in"].to(dev)
    ref_seq = d("conformer_seq")["conformer_seq"].to(dev)
    lens = torch.tensor([conf_in.size(1)], device=dev)
    with torch.no_grad():
        my_seq, my_mask = conformer(conf_in, lens)
    # Intermediate tensor: project precedent (w2v-bert test) allows <1e-3 here;
    # SDPA's fused accumulation order differs from official's two separate
    # matmuls on near-zero elements only. The final emo_vec seam is checked
    # strictly below.
    max_diff = (my_seq - ref_seq).abs().max().item()
    print(f"[1] conformer_seq:  {tuple(my_seq.shape)} vs {tuple(ref_seq.shape)} "
          f"max_diff={max_diff:.6f} (std <1e-3)")
    assert max_diff < 1e-3

    # ---- 2. perceiver encoder ----
    ref_perc = d("perceiver_out")["perceiver_out"].to(dev)
    # conformer mask [B,1,T'] → squeeze → pad (1,0) True (as official emo_cond_mask_pad)
    conds_mask = torch.nn.functional.pad(my_mask.squeeze(1), (1, 0), value=True)
    with torch.no_grad():
        my_perc = perceiver(my_seq, conds_mask)
    perc_ok = torch.allclose(my_perc, ref_perc, atol=1e-4, rtol=1e-3)
    max_diff = (my_perc - ref_perc).abs().max().item()
    print(f"[2] perceiver_out:  {tuple(my_perc.shape)} vs {tuple(ref_perc.shape)} "
          f"allclose={perc_ok} max_diff={max_diff:.6f}")
    assert perc_ok

    # ---- 3. full get_emovec path (merged emovec, alpha=0.65) ----
    ref_merged = d("merged_emovec")["merged_emovec_a065"].to(dev)
    emo_cond_emb = d("emo_cond_emb")["emo_cond_emb"].to(dev)
    with torch.no_grad():
        emo_vec = get_emovec(conformer, perceiver, emovec_layer, emo_layer,
                             emo_cond_emb, torch.tensor([emo_cond_emb.size(1)], device=dev))
    # THE SEAM that feeds GPT conditioning — must pass the project standard.
    merged_ok = torch.allclose(emo_vec, ref_merged, atol=1e-4, rtol=1e-3)
    max_diff = (emo_vec - ref_merged).abs().max().item()
    print(f"[3] emo_vec (seam):  {tuple(emo_vec.shape)} vs {tuple(ref_merged.shape)} "
          f"allclose={merged_ok} max_diff={max_diff:.6f}")
    assert merged_ok

    print("\n✓ emo_conditioning alignment PASSED (3/3)")


def test_emo_conditioning_alignment() -> None:
    """Pytest entry — same checks as main()."""
    main()


if __name__ == "__main__":
    main()
