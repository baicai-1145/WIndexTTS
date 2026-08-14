"""WIndexTTS WebUI — Gradio interface for zero-shot voice cloning.

A self-contained reimplementation of the official IndexTTS webui interaction,
wired directly to the pure-torch WIndexTTS engine (no indextts/transformers
dependency). Replicates the core UX: reference audio + text → synthesized
speech, with language selection, duration control, emotion (vector / text /
reference-audio), and advanced sampling parameters.

Run:
    python webui.py --model_dir /root/IndexTTS-2.5
    python webui.py --port 7860 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading

# Reduce CUDA caching-allocator reserved memory (fragmentation + preallocation):
# measured 4.28GB reserved -> 3.63GB on A10G for a full pipeline + warmup. Also
# lowers allocation latency on low-VRAM devices. Must be set before torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time

import gradio as gr
import torch
import torchaudio

# make windextts importable when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from windextts.inference import WIndexTTS
from windextts.download import check_install

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="WIndexTTS WebUI")
parser.add_argument("--model_dir", type=str, default="/root/IndexTTS-2.5",
                    help="Model checkpoints directory (gpt.pth, config.yaml, hf_cache/...)")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--fp16", action="store_true", default=True, help="Use fp16 (mixed precision)")
parser.add_argument("--no_fp16", dest="fp16", action="store_false", help="Disable fp16")
parser.add_argument("--ref", type=str, default=None, help="Default reference audio path")
parser.add_argument("--w4a16", action="store_true", default=False, help="Start in W4A16 mode (GPT INT4; requires torchao). The engine can be re-switched at runtime from the UI.")
parser.add_argument("--low_vram", action="store_true", default=False, help="Start in low-VRAM mode (3-4GB GPUs): w2v-bert streamed to CPU, beam3 KV bucket capped. Measured ~2.9GB steady. Switchable at runtime from the UI.")
cmd_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Engine management — runtime-switchable precision (W4A16 / fp16 / fp32)
#
# The three modes have different weight dtypes, so switching means a full
# engine rebuild (weight reload + graph re-capture, ~30-60s). The rebuild and
# synthesize share one lock: an in-flight synthesis blocks the rebuild, and
# synthesis during a rebuild is rejected with a warning.
# ---------------------------------------------------------------------------
_infer_lock = threading.Lock()

_PRECISION_CHOICES = [
    ("W4A16 (最快 · GPT INT4 量化)", "w4a16"),
    ("fp16 (推荐 · 速度/精度平衡)", "fp16"),
    ("fp32 (最高精度 · 基准)", "fp32"),
]
_MODE_NAMES = {"w4a16": "W4A16", "fp16": "fp16", "fp32": "fp32"}

_engine = {"tts": None, "mode": None, "low_vram": None, "building": False}


def _initial_mode() -> str:
    if cmd_args.w4a16:
        return "w4a16"
    return "fp16" if cmd_args.fp16 else "fp32"


def _build_engine(mode: str, low_vram: bool) -> WIndexTTS:
    dtype = torch.float32 if mode == "fp32" else torch.float16
    if mode == "w4a16":
        try:
            import torchao  # noqa: F401
        except ImportError as e:
            raise RuntimeError("W4A16 需要 torchao (pip install torchao)") from e
    print(f">> Loading WIndexTTS (mode={_MODE_NAMES[mode]}, dtype={dtype}, low_vram={low_vram})...")
    t = WIndexTTS(
        weights_dir=cmd_args.model_dir, device="cuda", dtype=dtype,
        enable_w4a16=(mode == "w4a16"), low_vram=low_vram,
    )
    t.warmup()
    return t


def engine_status_text() -> str:
    if _engine["building"]:
        return "⏳ 引擎重建中，请稍候..."
    if _engine["tts"] is None:
        return "❌ 引擎未加载"
    free_b, total_b = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    lv = "开" if _engine["low_vram"] else "关"
    return (
        f"当前引擎: {_MODE_NAMES[_engine['mode']]} · low_vram={lv}\n"
        f"显存: 已分配 {alloc:.2f}GB / 保留 {reserved:.2f}GB · GPU 剩余 {free_b / 1e9:.2f}GB"
    )


def apply_engine_config(precision, low_vram, progress=gr.Progress()):
    """Rebuild the engine with the requested precision / low_vram setting."""
    if _engine["building"]:
        gr.Warning("引擎正在重建中，请等待完成")
        return engine_status_text()
    if precision == _engine["mode"] and bool(low_vram) == bool(_engine["low_vram"]):
        return engine_status_text()  # nothing to change

    _engine["building"] = True
    try:
        progress(0.1, desc=f"等待合成队列并释放旧引擎 ({_MODE_NAMES[precision]})...")
        acquired = _infer_lock.acquire(timeout=300)
        if not acquired:
            gr.Warning("有合成任务长时间未结束，重建取消")
            return engine_status_text()
        try:
            old = _engine["tts"]
            _engine["tts"] = None
            del old
            torch.cuda.empty_cache()
            progress(0.3, desc="加载权重 + 捕获 CUDA Graph (~30-60s)...")
            _engine["tts"] = _build_engine(precision, bool(low_vram))
            _engine["mode"] = precision
            _engine["low_vram"] = bool(low_vram)
            progress(0.95, desc="完成")
        finally:
            _infer_lock.release()
        gr.Info(f"引擎已切换: {_MODE_NAMES[precision]} (low_vram={'开' if low_vram else '关'})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Warning(f"引擎重建失败: {e} —— 回退到启动配置")
        try:
            _engine["tts"] = _build_engine(_initial_mode(), cmd_args.low_vram)
            _engine["mode"] = _initial_mode()
            _engine["low_vram"] = cmd_args.low_vram
        except Exception:
            _engine["tts"] = None
    finally:
        _engine["building"] = False
    return engine_status_text()


_engine["tts"] = _build_engine(_initial_mode(), cmd_args.low_vram)
_engine["mode"] = _initial_mode()
_engine["low_vram"] = cmd_args.low_vram
print(">> WIndexTTS ready.")

# ---------------------------------------------------------------------------
# Inference handler — bridges the Gradio UI to WIndexTTS.infer()
# ---------------------------------------------------------------------------
def synthesize(
    prompt_audio,
    text,
    lang,
    duration_factor,
    text_normalization,
    max_text_tokens_per_segment,
    # emotion controls
    emo_control_method,  # 0=none, 1=ref audio, 2=vector, 3=text
    emo_ref_path,
    emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text,
    # advanced sampling
    do_sample,
    top_p,
    top_k,
    temperature,
    # perf knobs (WIndexTTS extensions vs official)
    cfm_steps,
    teacache_thresh,
    progress=gr.Progress(),
):
    """Run WIndexTTS.infer() from UI inputs; return output audio path."""
    progress(0, desc="准备中...")
    if prompt_audio is None and cmd_args.ref is None:
        gr.Warning("请上传参考音频")
        return None
    if not text.strip():
        gr.Warning("请输入要合成的文本")
        return None
    if emo_control_method == 1 and emo_ref_path is None:
        gr.Warning("情感参考音频模式需要上传情感参考音频")

    ref_path = prompt_audio if prompt_audio is not None else cmd_args.ref

    # resolve emotion source
    emo_vector = None
    emo_text_arg = None
    emo_ref_arg = None
    if emo_control_method == 1:  # emotion reference audio (conformer path)
        emo_ref_arg = emo_ref_path
    elif emo_control_method == 2:  # custom vector
        emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    elif emo_control_method == 3:  # text description
        emo_text_arg = emo_text if emo_text else None

    progress(0.2, desc="合成中...")
    if _engine["building"]:
        gr.Warning("引擎重建中，请稍候再试")
        return None
    acquired = _infer_lock.acquire(timeout=300)
    if not acquired:
        gr.Warning("合成队列繁忙，请稍后重试")
        return None
    try:
        tts = _engine["tts"]
        if tts is None:
            gr.Warning("引擎未加载")
            return None
        t0 = time.perf_counter()
        sr, audio = tts.infer(
            spk_audio_prompt=ref_path,
            text=text,
            lang=lang,
            emo_vector=emo_vector,
            emo_text=emo_text_arg,
            emo_ref_path=emo_ref_arg,
            emo_alpha=float(emo_weight),
            duration_factor=float(duration_factor),
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            cfm_steps=int(cfm_steps),
            teacache_thresh=float(teacache_thresh),
            text_normalization=bool(text_normalization),
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        )
        dt = (time.perf_counter() - t0) * 1000
    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Warning(f"合成失败: {e}")
        return None
    finally:
        _infer_lock.release()

    progress(0.9, desc="保存中...")
    # unique output path (avoid concurrent overwrites)
    out_path = os.path.join(tempfile.gettempdir(), f"windextts_{int(time.time()*1000)}.wav")
    torchaudio.save(out_path, audio.float().unsqueeze(0), sr)
    print(f">> synthesized {audio.numel()/sr:.2f}s audio in {dt:.0f}ms (RTF={dt/(audio.numel()/sr):.3f})")
    return out_path


# ---------------------------------------------------------------------------
# UI layout — mirrors the official IndexTTS webui structure
# ---------------------------------------------------------------------------
# All 99 Whisper languages + extensions. Names from tokenizer.LANGUAGES.
# Ordered by English name for easy lookup; ZH/EN/JA/KO/YUE kept first (common use).
_LANG_ORDER = ["ZH", "EN", "JA", "KO", "YUE", "MINNAN", "WUYU"]
LANGUAGES = _LANG_ORDER + sorted(
    ["AF","AM","AR","AS","AZ","BA","BE","BG","BN","BO","BR","BS","CA","CS",
     "CY","DA","DE","EL","ES","ET","EU","FA","FI","FO","FR","GL","GU","HA",
     "HAW","HE","HI","HR","HT","HU","HY","ID","IS","IT","JW","KA","KK","KM",
     "KN","LA","LB","LN","LO","LT","LV","MG","MI","MK","ML","MN","MR","MS",
     "MT","MY","NE","NL","NN","NO","OC","PA","PL","PS","PT","RO","RU","SA",
     "SD","SI","SK","SL","SN","SO","SQ","SR","SU","SV","SW","TA","TE","TG",
     "TH","TK","TL","TR","TT","UK","UR","UZ","VI","YI","YO"]
)

with gr.Blocks(
    title="WIndexTTS WebUI",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# WIndexTTS — 纯 torch 加速 IndexTTS-2.5\n"
        "Windows 友好 · 零 JIT 编译 · `pip install` 即用"
    )

    # --- model management tab: download / verify weights ---
    with gr.Tab("模型管理"):
        gr.Markdown(
            "### 安装 / 检查模型权重\n"
            "模型来自 IndexTeam 官方发布（HuggingFace 或 ModelScope），共 ~10GB。\n"
            "首次使用前需下载到本地目录，然后在该目录启动 webui（或用 `--model_dir` 指定）。"
        )
        with gr.Row():
            with gr.Column():
                dl_dir = gr.Textbox(
                    label="下载目录", value=cmd_args.model_dir,
                    placeholder="/path/to/IndexTTS-2.5",
                )
                dl_source = gr.Radio(
                    label="下载源", choices=["huggingface", "modelscope"],
                    value="huggingface",
                    info="国内网络选 modelscope 更快",
                )
                dl_skip_qwen = gr.Checkbox(
                    label="跳过 qwen0.6bemo4-merge（省 1.2GB，仅影响“情感文本描述”功能）",
                    value=False,
                )
                dl_btn = gr.Button("⬇ 下载模型 (~10GB)", variant="primary")
                dl_status = gr.Textbox(label="下载状态", lines=3, interactive=False,
                                       value="点击下载开始（可断点续传；完成后用下方按钮验证）")
            with gr.Column():
                verify_dir = gr.Textbox(label="检查目录", value=cmd_args.model_dir)
                verify_btn = gr.Button("✓ 检查完整性")
                verify_status = gr.Textbox(label="检查结果", lines=6, interactive=False)

        import threading as _th
        _dl_state = {"running": False}

        def _do_download(out_dir, source, skip_qwen):
            if _dl_state["running"]:
                return
            if not out_dir or not out_dir.strip():
                return
            _dl_state["running"] = True
            try:
                from windextts.download import download_model
                download_model(out_dir.strip(), source=source, include_qwen=not skip_qwen)
            except SystemExit as e:
                print(f"[download] {e}")
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                _dl_state["running"] = False

        def start_download(out_dir, source, skip_qwen):
            if _dl_state["running"]:
                return "⏳ 已有下载任务进行中..."
            if not out_dir or not out_dir.strip():
                return "❌ 请填写下载目录"
            if not (source == "modelscope" or source == "huggingface"):
                return "❌ 无效下载源"
            _th.Thread(target=_do_download, args=(out_dir, source, skip_qwen), daemon=True).start()
            return (f"⏳ 下载已后台启动 -> {out_dir.strip()}（源: {source}）\n"
                    "进度请看服务器控制台输出；完成后点“检查完整性”验证。")

        def do_verify(d):
            if not d or not d.strip():
                return "❌ 请填写目录"
            ok, missing, missing_opt = check_install(d.strip())
            lines = [f"目录: {d.strip()}"]
            lines.append(f"核心文件: {'✓ 完整' if not missing else '✗ 缺失 ' + str(missing)}")
            opt = "✓ 已安装" if not missing_opt else "- 未安装 (仅影响情感文本描述)"
            lines.append(f"可选 (emo_text): {opt}")
            lines.append("✅ 可以启动推理" if ok else "❌ 需补齐核心文件")
            return "\n".join(lines)

        dl_btn.click(start_download, inputs=[dl_dir, dl_source, dl_skip_qwen], outputs=[dl_status])
        verify_btn.click(do_verify, inputs=[verify_dir], outputs=[verify_status])

    with gr.Tab("音频生成"):
        # --- engine config: runtime precision switch (W4A16 / fp16 / fp32) ---
        with gr.Accordion("引擎配置 (精度切换)", open=True):
            with gr.Row():
                precision_radio = gr.Radio(
                    label="精度模式", choices=_PRECISION_CHOICES, value=_initial_mode(),
                )
                with gr.Column(scale=1):
                    low_vram_cb = gr.Checkbox(
                        label="低显存模式 (3-4GB 显卡; 保持 beam3, 稳态 ~2.9GB)",
                        value=cmd_args.low_vram,
                    )
                    apply_btn = gr.Button("应用配置 (重建引擎)", variant="secondary")
            engine_status = gr.Textbox(
                label="引擎状态", value=engine_status_text(), lines=2, interactive=False,
            )
            apply_btn.click(
                apply_engine_config, inputs=[precision_radio, low_vram_cb],
                outputs=[engine_status], api_name="apply_engine",
            )

        # --- reference audio + text input ---
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                prompt_audio = gr.Audio(
                    label="参考音频 (上传你想克隆的音色)",
                    type="filepath",
                    value=cmd_args.ref,
                )
            with gr.Column(scale=1):
                gr.Markdown("**提示**: 上传 5-15 秒清晰人声效果最佳")

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="要合成的文本",
                    placeholder="输入要合成的文字...",
                    lines=4,
                    value="欢迎使用WIndexTTS语音合成系统。",
                )
                with gr.Row():
                    lang_dropdown = gr.Dropdown(
                        label="语言", choices=LANGUAGES, value="ZH",
                    )
                    duration_factor = gr.Slider(
                        label="语速因子 (duration_factor)", minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    )
                    text_normalization = gr.Checkbox(
                        label="文本归一化 (数字→中文)", value=True,
                    )
                    max_text_tokens_per_segment = gr.Slider(
                        label="分段 token 上限 (单段文本长度, mel生成长度按语言自动匹配: 中文/日语×6, 英文×11)", minimum=20, maximum=600, value=120, step=10,
                    )

            with gr.Column(scale=1):
                gen_button = gr.Button("🎵 生成语音", variant="primary", size="lg")
                output_audio = gr.Audio(label="合成结果", type="filepath")

        with gr.Accordion("情感控制", open=False):
            emo_control = gr.Radio(
                label="情感来源",
                choices=[("无 (平静)", 0), ("情感参考音频", 1), ("自定义情感向量", 2), ("情感文本描述", 3)],
                value=0,
            )
            with gr.Group(visible=False) as emo_ref_group:
                emo_upload = gr.Audio(label="上传情感参考音频", type="filepath")
                emo_weight = gr.Slider(label="情感强度 (emo_alpha)", minimum=0.0, maximum=1.5, value=1.0, step=0.05)
            with gr.Group(visible=False) as emo_vec_group:
                gr.Markdown("8 维情感向量 (喜/怒/哀/惧/厌恶/低落/惊喜/平静)")
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(label="喜", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                        vec2 = gr.Slider(label="怒", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                        vec3 = gr.Slider(label="哀", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                        vec4 = gr.Slider(label="惧", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                    with gr.Column():
                        vec5 = gr.Slider(label="厌恶", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                        vec6 = gr.Slider(label="低落", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                        vec7 = gr.Slider(label="惊喜", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
                        vec8 = gr.Slider(label="平静", minimum=0.0, maximum=1.2, value=0.0, step=0.05)
            with gr.Group(visible=False) as emo_text_group:
                emo_text = gr.Textbox(
                    label="情感描述文本 (用 QwenEmotion 解析)",
                    placeholder="例如: 非常开心激动的语气",
                )

        with gr.Accordion("高级参数 (采样)", open=False):
            do_sample = gr.Checkbox(label="随机采样 (do_sample)", value=True)
            with gr.Row():
                top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.8, step=0.05)
                top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=30, step=1)
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)

        with gr.Accordion("性能调优 (WIndexTTS 专属)", open=False):
            gr.Markdown(
                "以下参数控制推理速度/质量权衡（默认值已调优为最佳速度-质量平衡）：\n"
                "- **CFM 步数**：S2Mel Flow Matching 求解步数，越少越快（官方 25，默认 12，最低 ~8）\n"
                "- **TeaCache 阈值**：DiT 跳步缓存阈值，越高越快（0=禁用，默认 0.25）\n"
            )
            with gr.Row():
                cfm_steps = gr.Slider(label="CFM 步数", minimum=6, maximum=25, value=12, step=1)
                teacache_thresh = gr.Slider(label="TeaCache 阈值", minimum=0.0, maximum=0.5, value=0.25, step=0.05)

        # --- wire up the generate button ---
        gen_button.click(
            synthesize,
            inputs=[
                prompt_audio, text_input, lang_dropdown, duration_factor,
                text_normalization, max_text_tokens_per_segment,
                emo_control, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text,
                do_sample, top_p, top_k, temperature,
                cfm_steps, teacache_thresh,
            ],
            outputs=output_audio,
        )

    # --- toggle emotion control group visibility ---
    def toggle_emo(method):
        return (
            gr.update(visible=(method == 1)),   # ref audio
            gr.update(visible=(method == 2)),   # vector
            gr.update(visible=(method == 3)),   # text
        )

    emo_control.change(
        toggle_emo, inputs=[emo_control],
        outputs=[emo_ref_group, emo_vec_group, emo_text_group],
    )



if __name__ == "__main__":
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, show_error=True)
