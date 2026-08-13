"""Alignment test: beam-search AR decode (num_beams>1) semantics.

Stage 3c verification of `_generate_beam_search`:
  - num_beams=1 (default) must produce the EXACT same greedy codes as before
    (regression guard; do_sample=False, repetition_penalty default 1.0).
  - num_beams=3 + do_sample=False must be deterministic and equal the
    num_beams=1 greedy codes up to the trailing stop token (all K beams are
    identical under greedy argmax, so the best finished hypothesis == greedy).
  - num_beams=3 + do_sample=True must be deterministic under a fixed RNG seed.
  - repetition_penalty must actually change the sampled sequence (quality fix
    active; official IndexTTS uses rep=10.0).

Greedy input contract is the same as test_gpt_ar_align.py (style + emo_vec_final
+ text_tokens + lang from the dump directory).
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
    style = torch.load(f"{DUMPS}/gpt.style.pt", weights_only=False).to(dev)
    emo_vec_final = torch.load(f"{DUMPS}/gpt.emo_vec_final.pt", weights_only=False).to(dev)
    text_tokens = torch.load(f"{DUMPS}/gpt.text_tokens_greedy.pt", weights_only=False).to(dev)
    lang = torch.load(f"{DUMPS}/gpt.lang.pt", weights_only=False).to(dev)
    return style, emo_vec_final, text_tokens, lang


def _greedy_codes(dev):
    return torch.load(f"{DUMPS}/gpt.greedy_codes.pt", weights_only=False).to(dev)


def test_beam1_greedy_no_regression():
    dev = "cuda"
    m = _load_model(dev)
    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)
    ref = _greedy_codes(dev)
    with torch.no_grad():
        codes = m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False)
    n = min(codes.shape[1], ref.shape[1])
    assert torch.equal(codes[:, :n], ref[:, :n]), (
        f"beam1 greedy diverged from official greedy: got {codes[:, :n].tolist()[:10]}"
    )
    print(f"[align] beam=1 greedy still matches official ({codes.shape[1]} tokens)")


def test_beam3_greedy_deterministic_equals_beam1():
    dev = "cuda"
    m = _load_model(dev)
    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)
    with torch.no_grad():
        c1 = m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False)
        c3 = m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False, num_beams=3)
        c3b = m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False, num_beams=3)
    assert torch.equal(c3, c3b), "beam=3 greedy must be deterministic"
    # beam=1 includes the trailing stop token in `codes`; beam search strips it.
    assert torch.equal(c1[:, :-1], c3), (
        f"beam=3 greedy should equal beam=1 codes minus stop: {c1.shape} vs {c3.shape}"
    )
    print(f"[align] beam=3 greedy deterministic == beam=1 greedy ({c3.shape[1]} codes)")


def test_beam3_sample_deterministic_and_penalty_active():
    dev = "cuda"
    m = _load_model(dev)
    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)
    kw = dict(max_new_tokens=100, do_sample=True, top_k=30, top_p=0.8, temperature=0.8)
    torch.manual_seed(11)
    torch.cuda.manual_seed(11)
    with torch.no_grad():
        cs = m.generate(conds, text_tokens, lang, num_beams=3, repetition_penalty=10.0, **kw)
    torch.manual_seed(11)
    torch.cuda.manual_seed(11)
    with torch.no_grad():
        cs2 = m.generate(conds, text_tokens, lang, num_beams=3, repetition_penalty=10.0, **kw)
    torch.manual_seed(11)
    torch.cuda.manual_seed(11)
    with torch.no_grad():
        cs1 = m.generate(conds, text_tokens, lang, num_beams=3, repetition_penalty=1.0, **kw)
    assert torch.equal(cs, cs2), "beam=3 seeded sample must be deterministic"
    assert not torch.equal(cs, cs1), "repetition_penalty must change the sequence"
    print(f"[align] beam=3 seeded sample deterministic; rep penalty active ({cs.shape[1]} codes)")


if __name__ == "__main__":
    test_beam1_greedy_no_regression()
    test_beam3_greedy_deterministic_equals_beam1()
    test_beam3_sample_deterministic_and_penalty_active()
    print("BEAM OK")
