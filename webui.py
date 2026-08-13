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

import gradio as gr
import torch
import torchaudio

# make windextts importable when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from windextts.inference import WIndexTTS

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
cmd_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build the TTS engine (fp16 by default; the fast path)
# ---------------------------------------------------------------------------
dtype = torch.float16 if cmd_args.fp16 else torch.float32
print(f">> Loading WIndexTTS (weights_dir={cmd_args.model_dir}, dtype={dtype})...")
tts = WIndexTTS(weights_dir=cmd_args.model_dir, device="cuda", dtype=dtype)
tts.warmup()
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
    max_mel_tokens,
):
    """Run WIndexTTS.infer() from UI inputs; return output audio path."""
    if prompt_audio is None and cmd_args.ref is None:
        gr.Warning("请上传参考音频")
        return None
    if not text.strip():
        gr.Warning("请输入要合成的文本")
        return None

    ref_path = prompt_audio if prompt_audio is not None else cmd_args.ref

    # resolve emotion source
    emo_vector = None
    emo_text_arg = None
    if emo_control_method == 2:  # custom vector
        emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        # apply weight scaling (alpha)
        if emo_weight != 1.0:
            emo_vector = [int(v * emo_weight * 10000) / 10000 for v in emo_vector]
    elif emo_control_method == 3:  # text description
        emo_text_arg = emo_text if emo_text else None
    # method 1 (ref audio) and 0 (none) → emo_vector=None (calm default)

    try:
        sr, audio = tts.infer(
            spk_audio_prompt=ref_path,
            text=text,
            lang=lang,
            emo_vector=emo_vector,
            emo_text=emo_text_arg,
            duration_factor=float(duration_factor),
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            max_mel_tokens=int(max_mel_tokens),
            text_normalization=bool(text_normalization),
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Warning(f"合成失败: {e}")
        return None

    # write to a temp wav and return the path (Gradio Audio wants a path)
    out_path = os.path.join(tempfile.gettempdir(), "windextts_out.wav")
    torchaudio.save(out_path, audio.float().unsqueeze(0), sr)
    return out_path


# ---------------------------------------------------------------------------
# UI layout — mirrors the official IndexTTS webui structure
# ---------------------------------------------------------------------------
LANGUAGES = ["ZH", "EN", "JA", "KO", "YUE"]

with gr.Blocks(
    title="WIndexTTS WebUI",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# WIndexTTS — 纯 torch 加速 IndexTTS-2.5\n"
        "Windows 友好 · 零 JIT 编译 · `pip install` 即用"
    )

    with gr.Tab("音频生成"):
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
                        label="分段 token 上限", minimum=20, maximum=600, value=120, step=10,
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
                max_mel_tokens = gr.Slider(label="max_mel_tokens (生成长度上限)", minimum=50, maximum=1000, value=220, step=10)

        # --- wire up the generate button ---
        gen_button.click(
            synthesize,
            inputs=[
                prompt_audio, text_input, lang_dropdown, duration_factor,
                text_normalization, max_text_tokens_per_segment,
                emo_control, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text,
                do_sample, top_p, top_k, temperature, max_mel_tokens,
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
