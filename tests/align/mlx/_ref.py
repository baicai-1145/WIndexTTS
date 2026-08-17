# Torch-CPU reference pipeline for MLX alignment (tests only). Mirrors
# windextts/inference.py step-for-step with graphs disabled; loads the OFFICIAL
# weights on CPU fp32 and the mlx side from the converted dir.
import os

import numpy as np
import torch

os.environ.setdefault("WINDEXTTS_WEIGHTS_DIR", "/Volumes/2T/IndexTTS-2.5")
torch.set_num_threads(8)
torch.manual_seed(0)

import torchaudio  # noqa: E402
from windextts.models.bigvgan import BigVGAN, BigVGANConfig  # noqa: E402
from windextts.models.campplus import CAMPPlus  # noqa: E402
from windextts.models.codec import EnhancedCodec  # noqa: E402
from windextts.models.gpt import UnifiedVoice  # noqa: E402
from windextts.models.length_regulator import InterpolateRegulator  # noqa: E402
from windextts.models.s2mel_cfm import S2Mel, S2MelCFM  # noqa: E402
from windextts.models.s2mel_dit import DiT  # noqa: E402
from windextts.models.w2v2_bert import Wav2Vec2BertConformer  # noqa: E402
from windextts.weights import WeightLoader  # noqa: E402

SRC = "/Volumes/2T/IndexTTS-2.5"
MLX = "/Volumes/2T/IndexTTS-2.5-mlx"


import contextlib  # noqa: E402

# WIndexTTS pins SDPA to the (CUDA-only) efficient backend; on CPU let torch
# pick the math backend so the torch-CPU reference runs at all.
import windextts.models.gpt as _gpt

gpt_module = _gpt

gpt_module.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()


class RefTTS:
    # torch-CPU fp32 mirror of the WIndexTTS module set (no CUDA graphs)
    def __init__(self, weights_dir=SRC):
        w = WeightLoader(weights_dir)
        self.w2v_bert = Wav2Vec2BertConformer()
        self.w2v_bert.load_official(w.load_w2v_bert())
        self.w2v_bert.eval()
        mean, var = w.load_w2v_stats()
        self.w2v_mean, self.w2v_std = mean, torch.sqrt(var)

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(w.load_campplus())
        self.campplus.eval()

        self.codec = EnhancedCodec(codebook_size=8192, hidden_size=1024, codebook_dim=8,
                                   vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12)
        self.codec.load_official(w.load_codec())
        self.codec.eval()

        self.gpt = UnifiedVoice()
        self.gpt.build_emo_conditioning()
        self.gpt.load_official(w.load_gpt(), load_emo_conditioning=True)
        self.gpt.eval()

        from windextts.config import load_default_config

        self.cfg = load_default_config()
        self.spk_matrix = w.load_spk_matrix()
        self.emo_matrix = w.load_emo_matrix()
        self.emo_num = list(self.cfg._raw.get("emo_num", [3, 17, 2, 8, 4, 5, 10, 24]))

        lr = self.cfg.s2mel.length_reg
        self.length_regulator = InterpolateRegulator(
            channels=lr.channels, sampling_ratios=lr.sampling_ratios, is_discrete=lr.is_discrete,
            in_channels=lr.in_channels, codebook_size=lr.content_codebook_size)
        net = w.load_s2mel()
        self.length_regulator.load_official(net["length_regulator"])
        self.dit = DiT()
        self.dit.load_official(net["cfm"])
        self.dit.eval()
        self.s2mel = S2Mel(self.length_regulator, S2MelCFM(self.dit, in_channels=self.cfg.s2mel.dit.in_channels))
        self.length_regulator.eval()

        bcfg = BigVGANConfig.from_json(f"{weights_dir}/hf_cache/bigvgan/config.json")
        self.bigvgan = BigVGAN(bcfg)
        self.bigvgan.load_official(w.load_bigvgan())
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

        from windextts.frontend.audio_utils import SeamlessM4TFeaturizer
        from windextts.frontend.mel import MelSpectrogram

        self.featurizer = SeamlessM4TFeaturizer()
        self.mel_fn = MelSpectrogram()

    # ---- feature extraction (torch CPU, same math as inference.py) ----
    @torch.no_grad()
    def extract_spk_cond(self, audio_16k):
        inp, am = self.featurizer(audio_16k, return_mask=True)
        out = self.w2v_bert(inp, am, return_layer=17)
        return (out - self.w2v_mean) / self.w2v_std

    @torch.no_grad()
    def extract_style(self, audio_16k):
        f = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        return self.campplus((f - f.mean(0, keepdim=True)).unsqueeze(0))

    @torch.no_grad()
    def build_emo_vec(self, style, spk_cond, emo_vector=None):
        if emo_vector is None:
            emo_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]
        wv = torch.tensor(emo_vector, dtype=torch.float32)
        mat = self.gpt.emo_matrix_lookup(style, wv, torch.split(self.spk_matrix, self.emo_num),
                                         torch.split(self.emo_matrix, self.emo_num))
        dt = self.gpt.emovec_layer.weight.dtype
        return mat.to(dt) + float((1.0 - wv.sum()).clamp(min=0.0)) * self.gpt.get_emovec(spk_cond.to(dt))

    # ---- full single-segment pipeline (mirrors WIndexTTS._infer_single) ----
    @torch.no_grad()
    def infer_single(self, spk_cond, style, ref_mel, text_tokens, lang_id, max_mel_tokens=96,
                     do_sample=False, cfm_steps=8, num_beams=1):
        emo_vec = self.build_emo_vec(style, spk_cond)
        conds_latent = self.gpt.build_conds_latent(style, emo_vec)
        codes = self.gpt.generate(conds_latent, text_tokens, lang_id, max_new_tokens=max_mel_tokens,
                                  do_sample=do_sample, top_k=30, top_p=0.8, temperature=0.8,
                                  stop_token=self.cfg.gpt.stop_mel_token,
                                  repetition_penalty=10.0, num_beams=num_beams, use_cuda_graph=False)
        if codes[0, -1].item() == self.cfg.gpt.stop_mel_token:
            codes = codes[:, :-1]
        s_infer = self.codec.decode(codes)
        mel = self.s2mel.inference(spk_cond, s_infer, ref_mel, style,
                                   n_timesteps=cfm_steps, inference_cfg_rate=0.7, use_graph=False)
        audio = self.bigvgan(mel).squeeze(0).squeeze(0).clamp(-1, 1)
        return codes, mel, audio


# ---------------- comparison helpers ----------------

def cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def to_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
