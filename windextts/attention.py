"""Adaptive SDPA (Scaled Dot-Product Attention) backend selection.

Design goal: **zero external attention dependency** in the core inference path.
We use only ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) with
explicit backend pinning via ``torch.nn.attention.sdpa_kernel``. No ``flash_attn``
import, no triton — works on Windows with stock torch wheels.

Backend policy (verified on torch 2.8.0+cu128, A10G):
  - prefill (seq > 1, e.g. GPT conditioning / DiT): ``CUDNN_ATTENTION``.
    Ships inside the torch CUDA wheel (cross-platform, no extra install).
    Measured ~= flash_attn speed, slightly faster.
  - decode (q seq == 1, GPT autoregressive step): cuDNN is *rejected* by
    PyTorch's sdp_utils guard ("cudnn SDPA does not support sequence length 1"
    — conservative perf guard in sdp_utils.cpp:458, not a library limitation).
    Use ``FLASH_ATTENTION`` on Linux, ``EFFICIENT_ATTENTION`` on Windows
    (mem_eff is the Windows-safe fallback; official flash backend needs a
    wheel match).

The kernel choice is made per-call (cheap) so a single module can mix prefill
and decode without two code paths. All public helpers accept the standard SDPA
tensors and forward ``is_causal`` / ``scale``.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ---------------------------------------------------------------------------
# Platform detection — Windows gets mem_eff for decode; Linux gets flash.
# cuDNN is always preferred for prefill where available.
# ---------------------------------------------------------------------------

_IS_WINDOWS = os.name == "nt" or os.environ.get("WINDEXTTS_FORCE_WIN") == "1"
# Allow override for benchmarking / debugging.
_FORCE_PREFILL = os.environ.get("WINDEXTTS_ATTN_PREFILL", "")  # cudnn|flash|eff|math
_FORCE_DECODE = os.environ.get("WINDEXTTS_ATTN_DECODE", "")


def _backend_for_prefill() -> SDPBackend:
    if _FORCE_PREFILL == "flash":
        return SDPBackend.FLASH_ATTENTION
    if _FORCE_PREFILL == "eff":
        return SDPBackend.EFFICIENT_ATTENTION
    if _FORCE_PREFILL == "math":
        return SDPBackend.MATH
    # Default: cuDNN (fastest, bundled in torch CUDA wheel).
    return SDPBackend.CUDNN_ATTENTION


def _backend_for_decode() -> SDPBackend:
    if _FORCE_DECODE == "flash":
        return SDPBackend.FLASH_ATTENTION
    if _FORCE_DECODE == "eff":
        return SDPBackend.EFFICIENT_ATTENTION
    if _FORCE_DECODE == "math":
        return SDPBackend.MATH
    # cuDNN cannot run with query seq==1 (sdp_utils.cpp:458 guard). Pick the
    # best available: flash on Linux, mem_eff on Windows (no wheel dependency).
    return SDPBackend.EFFICIENT_ATTENTION if _IS_WINDOWS else SDPBackend.FLASH_ATTENTION


def _is_decode(query: torch.Tensor) -> bool:
    """A decode step has query sequence length == 1 (single new token)."""
    return query.size(-2) == 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Adaptive-backend scaled dot-product attention.

    Picks cuDNN for prefill (seq>1) and flash/mem_eff for decode (q seq==1),
    matching the measured-optimal backend per shape.

    Args:
        query/key/value: [..., seq, head_dim] tensors (leading dims broadcast).
        attn_mask, dropout_p, is_causal, scale: forwarded to SDPA.

    Returns:
        Attention output, same shape as ``query``.
    """
    backend = _backend_for_decode() if _is_decode(query) else _backend_for_prefill()
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )


def attn_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Prefill-path attention — pinned to cuDNN (seq>1, e.g. DiT / GPT cond)."""
    with sdpa_kernel([_backend_for_prefill()]):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


def attn_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Decode-path attention — flash(Linux)/mem_eff(Win), for q seq==1."""
    with sdpa_kernel([_backend_for_decode()]):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


if __name__ == "__main__":
    # Smoke test: verify backend selection and that decode falls back correctly.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"platform: {'windows' if _IS_WINDOWS else 'linux'}, torch {torch.__version__}")
    print(
        f"prefill backend: {_backend_for_prefill().name} | "
        f"decode backend: {_backend_for_decode().name}"
    )

    # prefill
    q = k = v = torch.randn(1, 16, 512, 64, device=dev, dtype=dt)
    o1 = attn(q, k, v, is_causal=True)
    assert o1.shape == q.shape
    print(f"prefill ok, out {tuple(o1.shape)}")

    # decode (q seq=1)
    qd = torch.randn(1, 16, 1, 64, device=dev, dtype=dt)
    kd = vd = torch.randn(1, 16, 512, 64, device=dev, dtype=dt)
    o2 = attn(qd, kd, vd)
    assert o2.shape == qd.shape
    print(f"decode ok, out {tuple(o2.shape)}")
    print("SMOKE OK")
