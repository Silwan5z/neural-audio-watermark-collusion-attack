"""多模型统一合谋攻击 — 共享基础设施（自包含）。

统一 6 个已有水印模型的接口（XAttnMark 因无官方实现标记 blocked，见 xattnmark_availability.md）：
  - AudioSeal (16k, 16bit, soft bit posterior + presence)
  - VoiceMark  (16k, 16bit = 4 chunk × 4bit, chunk logits + presence)
  - WavMark    (16k, 16bit + 16 pattern, raw soft, 无 presence)
  - WMCodec    (24k, 16bit = 4 digit × 4bit, digit logits, 无 presence)
  - TimbreWM   (22050, 10bit, 连续值, 无 presence)
  - SilentCipher (44.1k, 40bit = 5×8bit, 硬 message + confidence)

统一抽象：
  - payload: 原生 bit 数 d_m 的 0/1 向量
  - detect -> (soft_posterior_for_codebook, presence, native_hard)
  - 每个模型实现 embed(clean16k, payload) -> wm16k 和 detect(wm16k) -> score

统一 soft score：对码字 c（d_m bit）计算 log-lik 或等价分数，用于 64/16 身份排名。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # opensource/ 目录
AUDIO_DIR = ROOT / "dataset"  # 原始数据集（libritts16k 等）放置处，见 README
# 计算设备：默认 cuda:0，可用环境变量 WATERMARK_DEVICE 覆盖（如 cpu / cuda:1）
DEVICE = os.environ.get("WATERMARK_DEVICE", "cuda:0")
SR16 = 16000

_models = {}


def load_audio(path: Path, sr: int = SR16) -> np.ndarray:
    import librosa
    import soundfile as sf
    w, s = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if s != sr:
        w = librosa.resample(w, orig_sr=s, target_sr=sr).astype(np.float32)
    return w


def resample_to(w, sr_from, sr_to):
    import librosa
    if sr_from == sr_to:
        return w
    return librosa.resample(w, orig_sr=sr_from, target_sr=sr_to).astype(np.float32)


def load_codebook(n_payload=64):
    """返回 (n_payload, 16) {0,1} 码本。AudioSeal 用现有 64 码本；其他模型复用其前 K 行。"""
    cb = json.load(open(AUDIO_DIR / "codebook_k64.json"))
    arr = np.array([[int(c) for c in p] for p in cb], dtype=np.int8)
    return arr[:n_payload]


def pm(codebook_bits):
    return codebook_bits.astype(float) * 2 - 1


def pesq_wb(ref, deg):
    from pesq import pesq as pesq_fn
    n = min(len(ref), len(deg))
    try:
        return float(pesq_fn(16000, ref[:n], deg[:n], "wb"))
    except Exception:
        return float("nan")


def stoi(ref, deg):
    from pystoi import stoi as stoi_fn
    n = min(len(ref), len(deg))
    try:
        return float(stoi_fn(ref[:n], deg[:n], 16000, extended=False))
    except Exception:
        return float("nan")


def si_sdr(ref, deg):
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    s = np.dot(deg, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e = deg - s
    return float(10 * np.log10(np.dot(s, s) / (np.dot(e, e) + 1e-12)))


def k64_path(pi, spk):
    """AudioSeal k64 波形路径（用于反查 clean 文件名）。"""
    import csv
    for row in csv.DictReader(open(AUDIO_DIR / "as_pool_k64_manifest.csv")):
        if row["spk"] == spk and int(row["pi"]) == pi:
            return AUDIO_DIR / "as_pool_k64" / row["file"]
    raise KeyError((spk, pi))


def clean_path(spk):
    """给定 spk，返回 libritts16k 的 clean 波形路径（复用 AudioSeal k64 命名）。"""
    p = k64_path(0, spk)
    stem = p.stem  # {spk}_{spk}_{book}_{spk}_{book}_{utt}_k64_0
    body = stem.split("_k64_")[0]
    return AUDIO_DIR / "libritts16k" / f"{body[len(spk) + 1:]}.wav"


def load_split(split):
    """返回 split16.json 里指定 split 的 speaker 列表。"""
    d = json.load(open(AUDIO_DIR / "split16.json"))
    return d[split]


def make_codebook(n_payload, n_bits, seed=0):
    """按模型原生 bit 长度生成随机码本（固定 seed，攻击前冻结）。
    16-bit 模型复用 AudioSeal 的 codebook_k64.json（已有 min_d=5）。
    40-bit（SilentCipher）：随机生成（空间 2^40 巨大，随机 64 个 min_d 自然足够）。
    其他长度：从 2^n_bits 个整数中选 n_payload 个互异（保证无重复码字）。"""
    if n_bits == 16:
        return load_codebook(n_payload)
    if n_bits == 40:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=(n_payload, n_bits)).astype(np.int8)
    rng = np.random.default_rng(seed)
    assert n_payload <= 2 ** n_bits, f"n_payload={n_payload} > 2^{n_bits}"
    vals = rng.choice(2 ** n_bits, size=n_payload, replace=False)
    cb = np.zeros((n_payload, n_bits), dtype=np.int8)
    for i, v in enumerate(vals):
        for b in range(n_bits):
            cb[i, b] = (v >> b) & 1  # LSB-first（与各模型内部编码无关，仅作身份码）
    return cb


def hamming_stats(codebook_bits):
    """返回码本的 min/mean Hamming distance（任务书 §4.1 要求报告）。"""
    from itertools import combinations
    n = len(codebook_bits)
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            ds.append(int((codebook_bits[i] != codebook_bits[j]).sum()))
    return min(ds), float(np.mean(ds))


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────
def get_audioseal(dev=DEVICE):
    if "audioseal" not in _models:
        import torch
        from audioseal import AudioSeal
        gen = AudioSeal.load_generator("audioseal_wm_16bits").to(dev).eval()
        det = AudioSeal.load_detector("audioseal_detector_16bits").to(dev).eval()
        _models["audioseal"] = {"gen": gen, "det": det, "dev": dev, "nbits": 16, "sr": 16000}
    return _models["audioseal"]


def _clear_conflicting_modules():
    """清理 wmcodec 与 voicemark 之间冲突的模块名（两者都有 models.py 等）。"""
    for mod in list(sys.modules):
        base = mod.split(".")[0]
        if base in ("models", "infer", "watermark", "env", "meldataset",
                    "msstftd", "quantization", "resnet", "position_embedding",
                    "pooling_layers", "mpu", "EMT", "speechtokenizer"):
            del sys.modules[mod]


def get_voicemark(dev=DEVICE):
    if "voicemark" not in _models:
        import torch
        vm_dir = str(ROOT / "third_party/voicemark")
        _clear_conflicting_modules()
        sys.path.insert(0, vm_dir)
        from infer import WatermarkSolver
        solver = WatermarkSolver()
        solver.load_model(checkpoint_dir=vm_dir,
                          checkpoint_name="voicemark.pth", strict=True)
        solver.device = torch.device(dev)
        solver.model = solver.model.to(solver.device).eval()
        _models["voicemark"] = {"solver": solver, "dev": dev, "nbits": 16,
                                "nchunk": 4, "sr": 16000}
    return _models["voicemark"]


def get_wavmark(dev=DEVICE):
    if "wavmark" not in _models:
        import torch
        import wavmark
        m = wavmark.load_model().to(dev).eval()
        _models["wavmark"] = {"model": m, "dev": dev, "nbits": 16, "sr": 16000}
    return _models["wavmark"]


def get_wmcodec(dev=DEVICE):
    if "wmcodec" not in _models:
        import torch
        _clear_conflicting_modules()
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "third_party/wmcodec"))
        from env import AttrDict
        from models import Encoder, Generator, Quantizer
        from watermark import Watermark_Encoder, Watermark_Decoder
        cfg = json.load(open(ROOT / "third_party/wmcodec/save_model/config.json"))
        h = AttrDict(cfg)
        ckpt_path = ROOT / "third_party/wmcodec/save_model/g_00150000"
        gen = Generator(h).to(dev)
        enc = Encoder(h).to(dev)
        quant = Quantizer(h, "Audio").to(dev)
        wm_enc = Watermark_Encoder(h).to(dev)
        wm_dec = Watermark_Decoder(h).to(dev)
        sd = torch.load(ckpt_path, map_location=dev)
        gen.load_state_dict(sd["generator"]); enc.load_state_dict(sd["encoder"])
        quant.load_state_dict(sd["quantizer_Audio"])
        wm_enc.load_state_dict(sd["watermark_encoder"])
        wm_dec.load_state_dict(sd["watermark_decoder"])
        for m in (gen, enc, wm_enc, wm_dec):
            m.eval()
            if hasattr(m, "remove_weight_norm"):
                m.remove_weight_norm()
        _models["wmcodec"] = {"gen": gen, "enc": enc, "quant": quant,
                              "wm_enc": wm_enc, "wm_dec": wm_dec, "h": h,
                              "dev": dev, "nbits": 16, "ndigit": 4, "sr": 24000}
    return _models["wmcodec"]


def get_timbrewm(dev=DEVICE):
    if "timbrewm" not in _models:
        import os
        import torch
        import yaml
        sys.path.insert(0, str(ROOT / "third_party/timbrewm"))
        base = ROOT / "third_party/timbrewm"
        # hifigan vocoder 用相对路径，需切 cwd 到 timbrewm 目录
        cwd = os.getcwd()
        os.chdir(base)
        try:
            from model.conv2_mel_modules import Encoder as TEnc, Decoder as TDec
            process_config = yaml.load(open(base / "config/process.yaml"), Loader=yaml.FullLoader)
            model_config = yaml.load(open(base / "config/model.yaml"), Loader=yaml.FullLoader)
            train_config = yaml.load(open(base / "config/train.yaml"), Loader=yaml.FullLoader)
            win_dim = process_config["audio"]["win_len"]
            embedding_dim = model_config["dim"]["embedding"]
            msg_length = train_config["watermark"]["length"]
            enc = TEnc(process_config, model_config, msg_length, win_dim, embedding_dim,
                       nlayers_encoder=model_config["layer"]["nlayers_encoder"],
                       attention_heads=model_config["layer"]["attention_heads_encoder"]).to(dev)
            dec = TDec(process_config, model_config, msg_length, win_dim, embedding_dim,
                       nlayers_decoder=model_config["layer"]["nlayers_decoder"],
                       attention_heads=model_config["layer"]["attention_heads_decoder"]).to(dev)
            ckpt_path = base / "results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar"
            ckpt = torch.load(ckpt_path, map_location=dev)
            enc.load_state_dict(ckpt["encoder"])
            dec.load_state_dict(ckpt["decoder"], strict=False)  # vocoder 单独加载，ckpt 不含
        finally:
            os.chdir(cwd)
        enc.eval(); dec.eval()
        _models["timbrewm"] = {"enc": enc, "dec": dec, "dev": dev,
                               "nbits": msg_length, "sr": 22050}
    return _models["timbrewm"]


def get_silentcipher(dev=DEVICE):
    if "silentcipher" not in _models:
        import silentcipher
        model_dir = ROOT / "third_party/silentcipher/44_1_khz/73999_iteration"
        model = silentcipher.server.get_model(
            model_type="44.1k", ckpt_path=str(model_dir),
            config_path=str(model_dir / "hparams.yaml"), device=dev)
        _models["silentcipher"] = {"model": model, "dev": dev, "nbits": 40, "sr": 44100}
    return _models["silentcipher"]


# ─────────────────────────────────────────────────────────────────────────────
# 统一检测：detect(wm16k) -> (soft_scores [n_codebook], presence, native)
# soft_scores 是码本每个身份的 log-lik（越大越像）
# ─────────────────────────────────────────────────────────────────────────────
def _loglik_bits(prob, codebook_bits):
    """prob [d] bit 后验，codebook [K,d] {0,1} -> [K] log-lik。"""
    p = np.clip(prob, 1e-9, 1 - 1e-9)
    ll1 = np.log(p); ll0 = np.log1p(-p)
    return np.where(codebook_bits == 1, ll1[None, :], ll0[None, :]).sum(axis=1)


def _loglik_chunks(chunk_probs, codebook_bits, nchunk, bit_order="msb"):
    """chunk_probs [nchunk, 16]（每 chunk 4bit 的 16 类概率），codebook [K, d] -> [K] log-lik。
    每 chunk 4 bit -> 值 v∈[0,15]，loglik = Σ_k log P_k[v_k]。
    bit_order: 'msb'（WMCodec）或 'lsb'（VoiceMark embed 用 LSB-first）。"""
    K = codebook_bits.shape[0]
    d = codebook_bits.shape[1]
    bits_per_chunk = d // nchunk
    if bit_order == "msb":
        weights = 2 ** np.arange(bits_per_chunk)[::-1]
    else:  # lsb
        weights = 2 ** np.arange(bits_per_chunk)
    ll = np.zeros(K)
    for k in range(nchunk):
        chunk_bits = codebook_bits[:, k * bits_per_chunk:(k + 1) * bits_per_chunk]
        vals = chunk_bits @ weights  # [K] 0-15
        probs = chunk_probs[k]  # [16]
        ll += np.log(np.clip(probs[vals], 1e-12, 1.0))
    return ll


def detect_audioseal(m, wm16k, codebook_bits):
    import torch
    t = torch.from_numpy(wm16k).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        r = m["det"].detector(t)
        prob = torch.sigmoid(r[:, 2:].mean(dim=-1))  # [1,16]
        pf = torch.softmax(r[:, :2], dim=1)[:, 1].mean(dim=-1)
    soft = prob[0].cpu().numpy()
    return _loglik_bits(soft, codebook_bits), float(pf[0]), (soft > 0.5).astype(np.int8)


def detect_voicemark(m, wm16k, codebook_bits):
    import torch
    t = torch.from_numpy(wm16k).float().to(m["dev"]).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        logits, chunk_logits = m["solver"].model.detect_watermark(t, return_logits=True)
        chunk_probs = torch.softmax(chunk_logits, dim=-1)[0].cpu().numpy()  # [4,16]
        presence = float(torch.sigmoid(logits).mean().cpu())
    # 硬 bits：每 chunk argmax
    hard = np.concatenate([np.unravel_index(np.argmax(chunk_probs[k]), (16,)) for k in range(4)]) if False else None
    vals = np.argmax(chunk_probs, axis=1)  # [4] 0-15
    bits = []
    for v in vals:
        bits.extend([(v >> i) & 1 for i in range(4)])  # LSB-first（VoiceMark embed 一致）
    hard = np.array(bits, dtype=np.int8)
    return _loglik_chunks(chunk_probs, codebook_bits, m["nchunk"], bit_order="lsb"), presence, hard


def detect_wavmark(m, wm16k, codebook_bits):
    import wavmark
    sig = wm16k.astype(np.float64)
    pay, info = wavmark.decode_watermark(m["model"], sig, decode_batch_size=200,
                                         len_start_bit=16, show_progress=False)
    presence = float(len(info.get("results", [])) > 0)
    hard = np.array(pay[:16], dtype=np.int8) if pay is not None else None
    # WavMark 归 Partial：raw soft 未校准，soft 排名用 hard payload 的 Hamming 距离
    if hard is not None:
        scores = -np.abs(codebook_bits - hard[None, :]).sum(axis=1).astype(float)
    else:
        scores = np.zeros(len(codebook_bits))
    return scores, presence, hard


def detect_wmcodec(m, wm16k, codebook_bits):
    import torch
    from third_party.wmcodec.meldataset import mel_spectrogram
    w24 = resample_to(wm16k, SR16, 24000)
    t = torch.from_numpy(w24).float().to(m["dev"]).unsqueeze(0)
    h = m["h"]
    with torch.no_grad():
        mel = mel_spectrogram(t, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                              h.win_size, h.fmin, h.fmax_for_loss)
        sign_score, sign_g_hat = m["wm_dec"](mel)  # sign_score: tuple of 4 [b,16]
    digit_logits = torch.stack(sign_score, dim=1)  # [1,4,16]
    digit_probs = torch.softmax(digit_logits, dim=-1)[0].cpu().numpy()  # [4,16]
    vals = sign_g_hat[0].cpu().numpy()
    # 硬 bits
    bits = []
    for v in vals:
        bits.extend([(v >> (3 - i)) & 1 for i in range(4)])
    hard = np.array(bits, dtype=np.int8)
    presence = np.nan  # 无 presence score
    return _loglik_chunks(digit_probs, codebook_bits, m["ndigit"], bit_order="msb"), presence, hard


def detect_timbrewm(m, wm16k, codebook_bits):
    import torch
    w22 = resample_to(wm16k, SR16, 22050)
    t = torch.from_numpy(w22).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        msg = m["dec"].test_forward(t)  # [1,1,10]
    soft = msg[0, 0].cpu().numpy()  # [10] 连续值
    # 转概率（sigmoid 近似）
    prob = 1.0 / (1.0 + np.exp(-soft))
    hard = (soft >= 0).astype(np.int8)
    presence = np.nan
    return _loglik_bits(prob, codebook_bits), presence, hard


# ── SilentCipher (44.1k, 40bit = 5×8bit 字符) ──
def _sc_bits_to_msg(bits40):
    """40 bit {0,1} -> 5 个 8-bit 数字（MSB-first 每 8 bit 一组）。"""
    b = [int(x) for x in bits40]
    return [int("".join(map(str, b[i * 8:(i + 1) * 8])), 2) for i in range(5)]


def _sc_msg_to_char_idx(msg, message_len):
    """5 个 8-bit 数字 -> message_len 个字符 index（20 个 2-bit 值 +1，+ 结束符 0）。"""
    bm = "".join(["{0:08b}".format(mi) for mi in msg])  # 40 bit MSB
    idx = [int(bm[j * 2:j * 2 + 2], 2) + 1 for j in range(len(bm) // 2)]  # 20 个 1..4
    idx = idx + [0]  # 结束符
    while len(idx) < message_len:
        idx.append(0)
    return idx[:message_len]


def _sc_logits(m, wm44k):
    """返回逐字符 log-softmax [message_dim, n_rep, message_len]。"""
    import torch
    model = m["model"]
    message_len = model.config.message_len
    message_dim = model.config.message_dim
    y = wm44k.astype(np.float32)
    y = y * np.sqrt(model.average_energy_VCTK / np.mean(y ** 2))
    t = torch.FloatTensor(y).unsqueeze(0).unsqueeze(0).to(m["dev"])
    carrier, _ = model.stft.transform(t.squeeze(1))
    carrier = carrier[:, None]
    mr = model.dec_m[0](carrier)  # [1,1,md,time]
    logits = mr[0, 0]  # [md, time]
    n_rep = logits.shape[1] // message_len
    logits = logits[:, :n_rep * message_len].reshape(message_dim, n_rep, message_len)
    return torch.log_softmax(logits, dim=0)  # [md, n_rep, message_len]


def detect_silentcipher(m, wm44k, codebook_bits):
    """SilentCipher 软排名：逐字符 log-lik 对 64 个 payload 打分。"""
    import torch
    import scipy.stats as st
    model = m["model"]
    message_len = model.config.message_len
    logp = _sc_logits(m, wm44k)  # [md, n_rep, message_len]
    K = codebook_bits.shape[0]
    scores = np.zeros(K, dtype=np.float32)
    for pi in range(K):
        msg = _sc_bits_to_msg(codebook_bits[pi])
        idx = _sc_msg_to_char_idx(msg, message_len)
        ll = 0.0
        for k in range(message_len):
            ll += float(logp[idx[k], :, k].sum())
        scores[pi] = ll
    # hard：mode 投票解码出 40 bit
    pred = torch.argmax(logp, dim=0)  # [n_rep, message_len]
    ord_vals = st.mode(pred.cpu().numpy(), keepdims=False, axis=0).mode  # [message_len]
    hard_bits = []
    for v in ord_vals:
        v = int(v)
        if v == 0:
            break
        hard_bits.extend([(v - 1) >> 1 & 1, (v - 1) & 1])  # 2-bit 值
    hard = np.zeros(40, dtype=np.int8)
    hard[:min(40, len(hard_bits))] = hard_bits[:40]
    presence = float(np.max(scores) > -1e6)
    return scores, presence, hard


# 模型注册表
DETECT_FN = {
    "audioseal": detect_audioseal,
    "voicemark": detect_voicemark,
    "wavmark": detect_wavmark,
    "wmcodec": detect_wmcodec,
    "timbrewm": detect_timbrewm,
    "silentcipher": detect_silentcipher,
}


# ─────────────────────────────────────────────────────────────────────────────
# extract_evidence：返回 bit 级软证据 z ∈ [-1,1]^d_m（解码前连续表示，任务 #35）
# 统一到 d_m 维（与 code_pred = C a 同维度），chunk/digit 模型转 bit 边际证据。
# 不得用 hard bits；这是软证据（softmax/sigmoid 的连续读出）。
# ─────────────────────────────────────────────────────────────────────────────
def evidence_audioseal(m, wm16k):
    import torch
    t = torch.from_numpy(wm16k).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        r = m["det"].detector(t)
    logit = r[:, 2:].mean(dim=-1)[0].cpu().numpy()  # [16] logit
    return 2.0 * (1.0 / (1.0 + np.exp(-logit))) - 1.0  # -> [-1,1]


def _chunk_logits_to_bit_evidence(chunk_probs, bit_order="lsb"):
    """chunk_probs [nchunk,16] -> bit 边际证据 [-1,1] (d_m,)。"""
    nchunk, ncls = chunk_probs.shape
    bits_per_chunk = 4
    out = np.zeros(nchunk * bits_per_chunk)
    for k in range(nchunk):
        p = chunk_probs[k]  # [16]
        for j in range(bits_per_chunk):
            if bit_order == "lsb":
                mask = [(v >> j) & 1 for v in range(ncls)]
            else:  # msb
                mask = [(v >> (bits_per_chunk - 1 - j)) & 1 for v in range(ncls)]
            p1 = float(np.sum(p[np.array(mask, dtype=bool)]))
            out[k * bits_per_chunk + j] = 2.0 * p1 - 1.0
    return out


def evidence_voicemark(m, wm16k):
    import torch
    t = torch.from_numpy(wm16k).float().to(m["dev"]).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        logits, chunk_logits = m["solver"].model.detect_watermark(t, return_logits=True)
    chunk_probs = torch.softmax(chunk_logits, dim=-1)[0].cpu().numpy()  # [4,16]
    return _chunk_logits_to_bit_evidence(chunk_probs, bit_order="lsb")  # [16]


def evidence_wavmark(m, wm16k):
    # WavMark 归 Partial，无可靠 soft evidence
    return None


def evidence_wmcodec(m, wm16k):
    import torch
    from third_party.wmcodec.meldataset import mel_spectrogram
    w24 = resample_to(wm16k, SR16, 24000)
    t = torch.from_numpy(w24).float().to(m["dev"]).unsqueeze(0)
    h = m["h"]
    with torch.no_grad():
        mel = mel_spectrogram(t, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                              h.win_size, h.fmin, h.fmax_for_loss)
        sign_score, _ = m["wm_dec"](mel)
    digit_logits = torch.stack(sign_score, dim=1)  # [1,4,16]
    digit_probs = torch.softmax(digit_logits, dim=-1)[0].cpu().numpy()  # [4,16]
    return _chunk_logits_to_bit_evidence(digit_probs, bit_order="msb")  # [16]


def evidence_timbrewm(m, wm16k):
    import torch
    w22 = resample_to(wm16k, SR16, 22050)
    t = torch.from_numpy(w22).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        msg = m["dec"].test_forward(t)
    soft = msg[0, 0].cpu().numpy()  # [10] 连续值（解码前）
    return 2.0 / (1.0 + np.exp(-soft)) - 1.0  # -> [-1,1]


EXTRACT_FN = {
    "audioseal": evidence_audioseal,
    "voicemark": evidence_voicemark,
    "wavmark": evidence_wavmark,
    "wmcodec": evidence_wmcodec,
    "timbrewm": evidence_timbrewm,
}


def extract_evidence(model_name, wm16k):
    """返回解码前连续证据 z（各模型维度不同）。wavmark 返回 None。"""
    m = GET_MODEL[model_name]()
    return EXTRACT_FN[model_name](m, wm16k)
GET_MODEL = {
    "audioseal": get_audioseal,
    "voicemark": get_voicemark,
    "wavmark": get_wavmark,
    "wmcodec": get_wmcodec,
    "timbrewm": get_timbrewm,
    "silentcipher": get_silentcipher,
}
NATIVE_SR = {"audioseal": 16000, "voicemark": 16000, "wavmark": 16000,
             "wmcodec": 24000, "timbrewm": 22050, "silentcipher": 44100}


def detect(model_name, wm16k, codebook_bits):
    m = GET_MODEL[model_name]()
    return DETECT_FN[model_name](m, wm16k, codebook_bits)


# ─────────────────────────────────────────────────────────────────────────────
# 统一嵌入：embed(model_name, clean16k, payload_bits) -> wm16k
# ─────────────────────────────────────────────────────────────────────────────
def embed_audioseal(m, clean16k, bits):
    import torch
    t = torch.from_numpy(clean16k).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)
    msg = torch.tensor([[int(b) for b in bits]], dtype=torch.long, device=m["dev"])
    with torch.no_grad():
        y = m["gen"](t, sample_rate=16000, message=msg)
    return y[0, 0].cpu().numpy().astype(np.float32)


def embed_voicemark(m, clean16k, bits):
    import torch
    t = torch.from_numpy(clean16k).float().to(m["dev"]).unsqueeze(0).unsqueeze(1)
    msg = torch.tensor([bits], dtype=torch.long, device=m["dev"])
    with torch.no_grad():
        out = m["solver"].model(t, message=msg)
    return out["recon_wm"][0, 0].cpu().numpy().astype(np.float32)


def embed_wavmark(m, clean16k, bits):
    import wavmark
    wm, _ = wavmark.encode_watermark(m["model"], clean16k.astype(np.float64),
                                     [int(b) for b in bits], min_snr=20, max_snr=38,
                                     show_progress=False)
    return wm.astype(np.float32)


def embed_wmcodec(m, clean16k, bits):
    import torch
    from third_party.wmcodec.meldataset import mel_spectrogram
    w24 = resample_to(clean16k, SR16, 24000)
    t = torch.from_numpy(w24).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)
    digits = []
    for k in range(4):
        chunk = bits[k * 4:(k + 1) * 4]
        v = int("".join(map(str, chunk)), 2)
        digits.append(v)
    sign = torch.tensor([digits], dtype=torch.long, device=m["dev"])  # [1,4]
    with torch.no_grad():
        sign_en = m["wm_enc"](sign)
        en_y = m["enc"](t, sign_en)
        q, _, _ = m["quant"](en_y)
        y = m["gen"](q)
    out = y[0, 0].cpu().numpy()
    return resample_to(out, 24000, SR16).astype(np.float32)


def embed_timbrewm(m, clean16k, bits):
    import torch
    w22 = resample_to(clean16k, SR16, 22050)
    t = torch.from_numpy(w22).float().to(m["dev"]).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    msg = torch.tensor([[bits]], dtype=torch.float32, device=m["dev"]) * 2 - 1  # [1,1,10] {-1,1}
    with torch.no_grad():
        y, _ = m["enc"].test_forward(t, msg)
    out = y[0, 0].cpu().numpy()
    return resample_to(out, 22050, SR16).astype(np.float32)


def embed_silentcipher(m, clean16k, bits):
    """SilentCipher 嵌入：clean16k -> 44.1k -> wm44k（返回 44.1k，水印在高频）。"""
    w44 = resample_to(clean16k, SR16, 44100)
    msg = _sc_bits_to_msg(bits)
    enc, _ = m["model"].encode_wav(w44, 44100, msg, calc_sdr=False)
    return enc.astype(np.float32)


EMBED_FN = {
    "audioseal": embed_audioseal,
    "voicemark": embed_voicemark,
    "wavmark": embed_wavmark,
    "wmcodec": embed_wmcodec,
    "timbrewm": embed_timbrewm,
    "silentcipher": embed_silentcipher,
}


def embed(model_name, clean16k, bits):
    m = GET_MODEL[model_name]()
    return EMBED_FN[model_name](m, clean16k, bits)
