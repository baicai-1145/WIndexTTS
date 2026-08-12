"""Alignment test: our GPT-AR decode loop (generate) vs official greedy codes.

Official reference (deterministic): the greedy AR decode of IndexTTS-2.5 for
text '<|zh|> 大家好。' on test.wav, dumped to gpt.greedy_codes.pt ([1,49]
codes ending at stop_mel_token 8193). Greedy argmax is deterministic, so our
decode loop must reproduce the exact token-id sequence.

Inputs used by the greedy dump (rebuilt here to be bit-identical):
  - style        = gpt.style.pt      [1,192]  CAMPPlus embedding (test.wav)
  - emo_vec_final= gpt.emo_vec_final.pt [1,1280]  full emo path output
  - conds_latent = build_conds_latent(style, emo_vec_final)  (NOTE: NOT the
    gpt.conds_latent.pt dump — that one came from a different emo path and
    diverges by ~0.16; the greedy dump used style+emo_vec_final directly.)
  - text_tokens  = gpt.text_tokens_greedy.pt [1,7]  ('<|zh|> 大家好。' + stop 1)
  - lang         = gpt.lang.pt [1] (=1, ZH)

Also verifies the prefill path (CORE alignment) is unregressed by the KV-cache
refactor: prefill logits must still match gpt.prefill_logits.pt (atol=1e-4).
"""
import os
import sys

sys.path.insert(0, "/root/WIndexTTS")

import torch

DUMPS = "/root/windextts_dumps"


def _load_model(dev):
    from windextts.models.gpt import UnifiedVoice
    from windextts.weights import WeightLoader

    m = UnifiedVoice()
    m.load_official(WeightLoader().load_gpt())
    return m.to(dev).eval()


def _greedy_inputs(dev):
    """Rebuild the exact inputs the greedy dump run used."""
    style = torch.load(f"{DUMPS}/gpt.style.pt", weights_only=False).to(dev)
    emo_vec_final = torch.load(f"{DUMPS}/gpt.emo_vec_final.pt", weights_only=False).to(dev)
    text_tokens = torch.load(f"{DUMPS}/gpt.text_tokens_greedy.pt", weights_only=False).to(dev)
    lang = torch.load(f"{DUMPS}/gpt.lang.pt", weights_only=False).to(dev)
    return style, emo_vec_final, text_tokens, lang


def test_gpt_ar_greedy_alignment():
    """AR decode loop must reproduce the official greedy codes exactly."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load_model(dev)

    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)  # [1,3,1280]
    ref = torch.load(f"{DUMPS}/gpt.greedy_codes.pt", weights_only=False).to(dev)

    with torch.no_grad():
        codes = m.generate(
            conds, text_tokens, lang,
            max_new_tokens=100, do_sample=False,
        )

    codes = codes.cpu()
    ref = ref.cpu()
    print(f"\n[align] greedy codes: mine={tuple(codes.shape)} ref={tuple(ref.shape)}")
    print(f"[align] mine first 15: {codes[0, :15].tolist()}")
    print(f"[align] ref  first 15: {ref[0, :15].tolist()}")

    # Exact match (greedy is deterministic).
    assert codes.shape == ref.shape, f"shape {codes.shape} != {ref.shape}"
    assert torch.equal(codes, ref), (
        "greedy codes mismatch; "
        f"prefix_len={int((codes[0] == ref[0]).sum())} "
        f"first_divergence={int((codes[0] != ref[0]).nonzero()[0].item()) if (codes[0] != ref[0]).any() else 'none'}"
    )
    assert codes[0, -1].item() == 8193, "last code should be stop_mel_token(8193)"
    print("[align] greedy codes EXACT match")


def test_gpt_prefill_still_aligned():
    """KV-cache refactor must not regress the CORE prefill alignment."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load_model(dev)

    conds = torch.load(f"{DUMPS}/gpt.conds_latent.pt", weights_only=False).to(dev)
    text_tokens = torch.load(f"{DUMPS}/gpt.text_tokens_short.pt", weights_only=False).to(dev)
    lang = torch.load(f"{DUMPS}/gpt.lang.pt", weights_only=False).to(dev)
    ref_logits = torch.load(f"{DUMPS}/gpt.prefill_logits.pt", weights_only=False).to(dev)

    with torch.no_grad():
        logits = m.prefill_logits_from_inputs(conds, text_tokens, lang)

    diff = (logits.float() - ref_logits.float()).abs().max().item()
    print(f"[align] prefill_logits max_diff={diff:.3e}")
    assert torch.allclose(logits.float(), ref_logits.float(), atol=1e-4, rtol=1e-3), f"diff {diff}"
    assert logits[0, -1].argmax() == ref_logits[0, -1].argmax(), "greedy first-mel mismatch"


if __name__ == "__main__":
    test_gpt_prefill_still_aligned()
    test_gpt_ar_greedy_alignment()
    print("\nGPT-AR GREEDY ALIGN OK")
