"""WIndexTTS HTTP API — OpenAI-compatible /v1/audio/speech.

POST /v1/audio/speech   {"model","input","response_format":"wav","voice",
                         "speed", "extra_params":{reference_audio(b64),
                         language, duration_factor, emo_vector[8]|emo_text|
                         emo_ref_audio(b64), do_sample, top_p, top_k,
                         temperature, cfm_steps, num_beams,
                         max_text_tokens_per_segment, text_normalization}}
GET  /v1/models / /health
Run: python -m windextts.server --model-dir <dir> [--w4a16] [--low-vram] [--port]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
import threading
import time
from pathlib import Path

# keep VRAM low before torch import (same as webui)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "server extras missing: pip install 'windextts[server]' "
        "(fastapi + uvicorn + pydantic)"
    ) from e

from windextts.inference import WIndexTTS

# single engine instance; requests serialised — CUDA Graphs are single-session
_engine: dict = {"tts": None}
_infer_lock = threading.Lock()
_voices: dict[str, str] = {}  # voice name -> ref audio path


class SpeechRequest(BaseModel):
    model: str = "windextts"
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0  # OpenAI `speed`; mapped to 1/duration_factor
    extra_params: dict = Field(default_factory=dict)


def _b64_to_tempfile(b64: str, suffix: str = ".wav") -> str:
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 reference_audio: {e}")
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tempfile.gettempdir())
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="WIndexTTS Server", version="0.1.0")

    dtype = torch.float32 if args.fp32 else torch.float16
    print(f">> Loading WIndexTTS (dtype={dtype}, w4a16={args.w4a16}, low_vram={args.low_vram}) ...")
    t0 = time.perf_counter()
    tts = WIndexTTS(
        weights_dir=args.model_dir, device="cuda", dtype=dtype,
        enable_w4a16=args.w4a16, low_vram=args.low_vram,
    )
    tts.warmup()
    _engine["tts"] = tts
    print(f">> WIndexTTS ready in {time.perf_counter() - t0:.1f}s")

    # --voice name=path[,name2=path2...] registration
    if args.voices:
        for spec in args.voices.split(","):
            name, _, path = spec.partition("=")
            name, path = name.strip(), path.strip()
            if name and path and os.path.exists(path):
                _voices[name] = path
                print(f">> registered voice '{name}' -> {path}")
            else:
                print(f"!! voice spec ignored (bad/missing file): {spec}")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": "windextts", "object": "model", "owned_by": "local",
                 "voices": sorted(_voices)}
            ],
        }

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        tts_ = _engine["tts"]
        ep = req.extra_params or {}
        t0 = time.perf_counter()

        # resolve reference audio: voice name > extra_params b64
        ref_b64 = ep.get("reference_audio")
        ref_path: str | None = _voices.get(req.voice) if req.voice else None
        tmp_ref = None
        if ref_path is None and ref_b64:
            tmp_ref = _b64_to_tempfile(ref_b64)
            ref_path = tmp_ref
        if not ref_path:
            raise HTTPException(400, "no reference audio: pass extra_params.reference_audio "
                                     "(base64) or use a registered voice")
        if req.response_format not in ("wav",):
            raise HTTPException(400, f"unsupported response_format '{req.response_format}' (wav only)")

        emo_ref_b64 = ep.get("emo_ref_audio")
        tmp_emo = _b64_to_tempfile(emo_ref_b64) if emo_ref_b64 else None
        emo_vector = ep.get("emo_vector")
        if emo_vector is not None:
            if not (isinstance(emo_vector, (list, tuple)) and len(emo_vector) == 8):
                raise HTTPException(400, "emo_vector must be a list of 8 floats")

        # OpenAI `speed` > 1 = faster/shorter; our duration_factor scales target
        # length, so duration_factor = 1 / speed (speed 1.2 -> factor 0.833).
        speed = req.speed if req.speed and req.speed > 0 else 1.0
        duration_factor = float(ep.get("duration_factor", 1.0 / speed))

        kwargs = dict(
            spk_audio_prompt=ref_path,
            text=req.input,
            lang=str(ep.get("language", "ZH")),
            emo_vector=[float(x) for x in emo_vector] if emo_vector else None,
            emo_text=ep.get("emo_text"),
            emo_ref_path=tmp_emo,
            emo_alpha=float(ep.get("emo_alpha", 1.0)),
            duration_factor=duration_factor,
            do_sample=bool(ep.get("do_sample", True)),
            top_p=float(ep.get("top_p", 0.8)),
            top_k=int(ep.get("top_k", 30)),
            temperature=float(ep.get("temperature", 0.8)),
            cfm_steps=int(ep.get("cfm_steps", 15)),
            text_normalization=bool(ep.get("text_normalization", True)),
            max_text_tokens_per_segment=int(ep.get("max_text_tokens_per_segment", 120)),
            num_beams=int(ep.get("num_beams", 3)),
        )

        try:
            acquired = _infer_lock.acquire(timeout=600)
            if not acquired:
                raise HTTPException(503, "synthesis queue busy (try again)")
            try:
                sr, audio = await run_in_threadpool(tts_.infer, **kwargs)
            finally:
                _infer_lock.release()
        finally:
            # clean temp files (non-registered refs / emo refs)
            for p in (tmp_ref, tmp_emo):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, audio.float().cpu().numpy().squeeze(), sr, format="WAV")
        dt = time.perf_counter() - t0
        print(f">> /v1/audio/speech: {req.input[:30]!r} -> {audio.numel() / sr:.2f}s "
              f"audio in {dt * 1000:.0f}ms (RTF={dt / (audio.numel() / sr):.3f})")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    return app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="windextts-server",
                                description="WIndexTTS OpenAI-compatible TTS server")
    p.add_argument("--model-dir", default=None, help="weights directory")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--w4a16", action="store_true", help="W4A16 INT4 GPT (needs torchao)")
    p.add_argument("--fp32", action="store_true", help="fp32 weights (reference precision)")
    p.add_argument("--low-vram", action="store_true",
                   help="low-VRAM mode (3-4GB GPUs, keeps beam3)")
    p.add_argument("--voices", default=None,
                   help="pre-registered voices: name=path.wav,name2=path2.wav")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
