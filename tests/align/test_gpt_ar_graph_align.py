"""Alignment test: CUDA-Graph AR decode must be bit-identical to eager AR decode.

Stage 3b verification:
  - generate(use_cuda_graph=False) vs generate(use_cuda_graph=True) with
    do_sample=False must produce the EXACT same code sequence.
  - Both must still match the official greedy dump gpt.greedy_codes.pt (49/49).
  - Bonus: seeded do_sample=True must also match (logits are bit-identical, so
    with a shared RNG seed multinomial draws identically).
  - Timing: warmup once, then compare eager vs graph decode wall time.

Greedy input contract is the same as test_gpt_ar_align.py (style + emo_vec_final
+ text_tokens + lang from the dump directory).
"""
import os
import sys
import time

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


def test_graph_greedy_identical_to_eager_and_official():
    dev = "cuda"
    m = _load_model(dev)
    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)  # [1,3,1280]
    ref = torch.load(f"{DUMPS}/gpt.greedy_codes.pt", weights_only=False).to(dev)

    with torch.no_grad():
        codes_eager = m.generate(
            conds, text_tokens, lang, max_new_tokens=100, do_sample=False,
        )
        codes_graph = m.generate(
            conds, text_tokens, lang, max_new_tokens=100, do_sample=False,
            use_cuda_graph=True,
        )

    codes_eager = codes_eager.cpu()
    codes_graph = codes_graph.cpu()
    ref = ref.cpu()
    print(f"\n[align] eager : {codes_eager[0, :15].tolist()} ...")
    print(f"[align] graph : {codes_graph[0, :15].tolist()} ...")
    print(f"[align] ref   : {ref[0, :15].tolist()} ...")

    assert torch.equal(codes_eager, codes_graph), (
        "CUDA Graph path diverged from eager path; "
        f"prefix={int((codes_eager[0] == codes_graph[0]).sum())}"
    )
    assert torch.equal(codes_graph, ref), (
        f"graph codes != official greedy; shape {codes_graph.shape} vs {ref.shape}"
    )
    assert codes_graph[0, -1].item() == 8193
    print(f"[align] graph == eager == official ({ref.shape[1]}/{ref.shape[1]} exact)")


def test_graph_sampling_matches_seeded():
    """Seeded do_sample must draw the same sequence (bit-identical logits)."""
    dev = "cuda"
    m = _load_model(dev)
    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)

    with torch.no_grad():
        torch.manual_seed(1234)
        codes_eager = m.generate(
            conds, text_tokens, lang, max_new_tokens=100, do_sample=True,
            top_k=30, top_p=0.8, temperature=0.8,
        )
        torch.manual_seed(1234)
        codes_graph = m.generate(
            conds, text_tokens, lang, max_new_tokens=100, do_sample=True,
            top_k=30, top_p=0.8, temperature=0.8, use_cuda_graph=True,
        )

    codes_eager = codes_eager.cpu()
    codes_graph = codes_graph.cpu()
    print(f"[align] seeded sample eager len={codes_eager.shape[1]} graph len={codes_graph.shape[1]}")
    assert torch.equal(codes_eager, codes_graph), (
        f"seeded sampling diverged; prefix={int((codes_eager[0] == codes_graph[0]).sum())}"
    )
    print(f"[align] seeded do_sample match ({codes_graph.shape[1]} tokens)")


def test_graph_speedup():
    """Warmup once, then time eager vs graph decode (excludes load)."""
    dev = "cuda"
    m = _load_model(dev)
    style, emo_vec_final, text_tokens, lang = _greedy_inputs(dev)
    conds = m.build_conds_latent(style, emo_vec_final)

    with torch.no_grad():
        # warmup both paths (first graph call captures)
        m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False)
        m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False, use_cuda_graph=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        ce = m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        t_eager = time.perf_counter() - t0

        t0 = time.perf_counter()
        cg = m.generate(conds, text_tokens, lang, max_new_tokens=100, do_sample=False, use_cuda_graph=True)
        torch.cuda.synchronize()
        t_graph = time.perf_counter() - t0

    n = ce.shape[1]
    print(f"\n[perf] eager={t_eager*1000:.1f}ms ({n} codes, {t_eager/n*1000:.2f}ms/tok)  "
          f"graph={t_graph*1000:.1f}ms ({t_graph/n*1000:.2f}ms/tok)  "
          f"speedup={t_eager/t_graph:.2f}x")
    assert torch.equal(ce, cg), "speedup run diverged (should be impossible)"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for CUDA Graph test")
    test_graph_greedy_identical_to_eager_and_official()
    test_graph_sampling_matches_seeded()
    test_graph_speedup()
    print("\nGPT-AR CUDA GRAPH ALIGN OK")
