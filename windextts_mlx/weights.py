# Weight conversion (torch checkpoints -> MLX safetensors, one-time) + MLX loader.
# convert_all runs on any machine with torch (windextts installed); the runtime
# loader is pure mlx. Conventions mirror windextts/weights.py + each model's
# load_official: keys remapped exactly as the torch side does, then tensors
# re-oriented for MLX layouts (Conv1d [out,k,in], Conv2d [out,kh,kw,in],
# ConvTranspose1d [out,k,in]) and weight_norm flattened (w = g*v/||v||).
from pathlib import Path

import mlx.core as mx

DEFAULT_MLX_DIR = Path("/Volumes/2T/IndexTTS-2.5-mlx")


def load_mlx(weights_dir, name):
    st = mx.load(str(Path(weights_dir) / f"{name}.safetensors"), format="safetensors")
    # safetensors is mmap-backed: without this, the first GPU kernel that
    # touches a weight page pulls it from disk mid-kernel (mechanical HDD =
    # 10ms+/page), blowing the Metal 2s watchdog. Force the whole file into
    # memory once at load time.
    mx.eval(*st.values())
    return st


def load_into(model, st, dtype=None):
    # filter to the model's own param names (drops ckpt extras like tied lm_head),
    # then assign leaves by attribute path. Direct leaf assignment (instead of
    # model.update(tree_unflatten(...))) sidesteps list/dict shape mismatch with
    # name-preserving Seq containers.
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))
    sel = {k: v for k, v in st.items() if k in flat}
    if dtype is not None:
        sel = {k: (v.astype(dtype) if v.dtype != mx.int32 else v) for k, v in sel.items()}
    for k, v in sel.items():
        parts = k.split(".")
        target = model
        for p in parts[:-1]:
            target = target[int(p)] if isinstance(target, list) else getattr(target, p)
        setattr(target, parts[-1], v)


# ---------------------------------------------------------------- torch side

def _convert_state_dict(torch_model, sd):
    # generic remapper: classify each key by its leaf module type
    cls = {}
    for name, mod in torch_model.named_modules():
        for pname, _ in mod.named_parameters(recurse=False):
            cls[f"{name}.{pname}" if name else pname] = type(mod).__name__
        for bname, _ in mod.named_buffers(recurse=False):
            cls[f"{name}.{bname}" if name else bname] = type(mod).__name__

    def _orient(k, t):
        c = cls.get(k)
        if c == "Conv1d" and t.dim() == 3:
            return t.permute(0, 2, 1)  # [out,in,k] -> [out,k,in]
        if c == "ConvTranspose1d" and t.dim() == 3:
            return t.permute(1, 2, 0)  # [in,out,k] -> [out,k,in]
        if c == "Conv2d" and t.dim() == 4:
            return t.permute(0, 2, 3, 1)  # [out,in,kh,kw] -> [out,kh,kw,in]
        return t  # biases/norm params stay as-is

    out = {}
    wn = {}
    for k, t in sd.items():
        v = t.detach()  # dtype handled in _torch_to_mlx (int buffers stay integral)
        if k.endswith(".weight_g"):
            wn[k[:-9]] = v
        elif k.endswith(".weight_v"):
            pass  # flattened with its g below
        else:
            out[k] = _orient(k, v)
    for base, g in wn.items():
        v = sd[base + ".weight_v"].detach().float()
        # torch weight_norm(dim=0): each output slice normalized SEPARATELY
        vn = v / (v.norm(dim=tuple(range(1, v.dim())), keepdim=True) + 1e-12)
        out[base + ".weight"] = _orient(base + ".weight_g", g * vn)
    return out


def _torch_to_mlx(t):
    import numpy as np
    import torch

    t = t.detach().contiguous().cpu().numpy()
    if t.dtype == np.int64:  # int buffers (input_pos) must stay integral
        t = t.astype(np.int32)
    else:  # bf16/fp16 ckpts -> fp32
        t = t.astype(np.float32)
    return mx.array(t)


def convert_all(src_dir, out_dir=DEFAULT_MLX_DIR, models=("gpt", "codec", "s2mel", "bigvgan", "campplus", "w2v_bert", "qwen")):
    # one-time torch -> mlx conversion of the official IndexTTS-2.5 checkpoints
    from windextts.weights import WeightLoader

    w = WeightLoader(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(name, st):
        mx.save_safetensors(str(out / f"{name}.safetensors"), st)
        print(f"{name}: {len(st)} keys -> {out / (name + '.safetensors')}")

    if "gpt" in models:
        from windextts.models.gpt import UnifiedVoice

        m = UnifiedVoice()
        m.build_emo_conditioning()
        _C1D = (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")
        sd = {k: (v.t().contiguous() if any(k.endswith(s) for s in _C1D) and v.dim() == 2 else v)
              for k, v in w.load_gpt().items()}
        _save("gpt", {k: _torch_to_mlx(v) for k, v in _convert_state_dict(m, sd).items()})

    if "codec" in models:
        from windextts.models.codec import EnhancedCodec

        m = EnhancedCodec(codebook_size=8192, hidden_size=1024, codebook_dim=8, vocos_dim=384,
                          vocos_intermediate_dim=2048, vocos_num_layers=12)
        _save("codec", {k: _torch_to_mlx(v) for k, v in _convert_state_dict(m, w.load_codec()).items()})

    if "s2mel" in models:
        from windextts.models.length_regulator import InterpolateRegulator
        from windextts.models.s2mel_dit import DiT

        net = w.load_s2mel()
        lr = InterpolateRegulator(channels=512, sampling_ratios=(1, 1, 1, 1), is_discrete=False,
                                  in_channels=1024, codebook_size=2048)
        dit = DiT()
        st = {f"length_regulator.{k}": _torch_to_mlx(v)
              for k, v in _convert_state_dict(lr, net["length_regulator"]).items()}
        st.update({k: _torch_to_mlx(v)
                   for k, v in _convert_state_dict(dit, {k[10:]: v for k, v in net["cfm"].items()}).items()})
        _save("s2mel", st)

    if "bigvgan" in models:
        from windextts.models.bigvgan import BigVGAN, BigVGANConfig

        bcfg = BigVGANConfig.from_json(Path(src_dir) / "hf_cache" / "bigvgan" / "config.json")
        m = BigVGAN(bcfg)
        _save("bigvgan", {k: _torch_to_mlx(v) for k, v in _convert_state_dict(m, w.load_bigvgan()).items()})

    if "campplus" in models:
        from windextts.models.campplus import CAMPPlus

        m = CAMPPlus(feat_dim=80, embedding_size=192)
        _save("campplus", {k: _torch_to_mlx(v) for k, v in _convert_state_dict(m, w.load_campplus()).items()})

    if "w2v_bert" in models:
        from windextts.models.w2v2_bert import Wav2Vec2BertConformer

        m = Wav2Vec2BertConformer()
        sd = {k.replace("encoder.layers.", "encoder_layers.", 1): v
              for k, v in w.load_w2v_bert().items() if k != "masked_spec_embed"}
        _save("w2v_bert", {k: _torch_to_mlx(v) for k, v in _convert_state_dict(m, sd).items()})

    if "qwen" in models:
        import json

        from safetensors.torch import load_file

        qdir = Path(src_dir) / "qwen0.6bemo4-merge"
        cfg = json.load(open(qdir / "config.json"))
        sd = load_file(str(qdir / "model.safetensors"))
        _save("qwen_emotion", {k.removeprefix("model."): _torch_to_mlx(v) for k, v in sd.items()})
        with open(out / "qwen_config.json", "w") as f:
            json.dump(cfg, f)

    # non-neural tables
    import numpy as np

    np.savez(out / "feat.npz", spk=w.load_spk_matrix().detach().numpy(), emo=w.load_emo_matrix().detach().numpy())
    mean, var = w.load_w2v_stats()
    np.savez(out / "stats.npz", mean=mean.numpy(), var=var.numpy())
    print("feat/stats -> feat.npz / stats.npz")

    # runtime text/config files (tiktoken BPE, qwen tokenizer, bigvgan config)
    import shutil

    shutil.copy(Path(src_dir) / "multilingual_zh_ja_yue_char_del.tiktoken", out / "multilingual_zh_ja_yue_char_del.tiktoken")
    qdir = Path(src_dir) / "qwen0.6bemo4-merge"
    if (qdir / "tokenizer.json").exists():
        shutil.copy(qdir / "tokenizer.json", out / "qwen_tokenizer.json")
    shutil.copy(Path(src_dir) / "hf_cache" / "bigvgan" / "config.json", out / "bigvgan_config.json")
    print("tokenizer/config files copied")
