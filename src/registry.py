"""全空间注册表、多说话人分配、按需嵌入缓存（公共工具）。

设计要点见 ../README.md。核心区别于早期版本的固定容量码本：
- 排序注册表 = 每个模型原生全 bit 空间（不是人为选的固定容量），虚拟码字只用于打分，不需要真实音频。
- 38 个说话人（libritts16k 现有全部 39 人，排除音频过短的说话人 61），trial 均匀分配。
- 按 (speaker, model, codeword_int) 缓存实际嵌入的音频，避免重复嵌入。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watermarks import load_audio, embed, detect, pesq_wb, stoi, si_sdr  # noqa: E402

REAL_ANALYSIS = Path(__file__).resolve().parent.parent / "dataset"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
NBITS = {"audioseal": 16, "timbrewm": 10, "wavmark": 16, "voicemark": 16, "wmcodec": 16}
CAP = 0.5

MANIFEST = REAL_ANALYSIS / "collusion_300" / "manifest.csv"


def speakers():
    """Return the 100 bilingual speakers from the generated manifest."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing dataset manifest: {MANIFEST}")
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return sorted({f"{r['language']}:{r['speaker_id']}" for r in csv.DictReader(f)})


def trials_per_speaker_plan(n_total=300, n_spk=38):
    """均匀分配 n_total 个 trial 到 n_spk 个说话人：前 r 个分 q+1，其余分 q。
    n_total=300, n_spk=38 时：q=7, r=34 → 34 个说话人分 8 个 + 4 个说话人分 7 个 = 34*8+4*7=300。
    """
    q, r = divmod(n_total, n_spk)
    return [q + 1 if i < r else q for i in range(n_spk)]


def speaker_trial_index(n_total=300, n_spk=None):
    """返回长度 n_total 的列表，每个元素是 (spk, local_t)，local_t 是该说话人内部的 trial 编号。"""
    spks = speakers()
    n_spk = len(spks) if n_spk is None else n_spk
    if n_spk != len(spks):
        raise ValueError(f"requested {n_spk} speakers but dataset has {len(spks)}")
    counts = trials_per_speaker_plan(n_total, n_spk)
    out = []
    for spk, c in zip(spks, counts):
        for local_t in range(c):
            out.append((spk, local_t))
    assert len(out) == n_total
    return out


def coalition_seed(spk, K, local_t):
    """与 v18 的 coalition_seed 保持同一形式，但输入是每说话人内部的 local_t。"""
    h = int.from_bytes(hashlib.sha256(spk.encode("utf-8")).digest()[:4], "little")
    return (h * 100000 + K * 1000 + local_t + 42) % (2 ** 31)


def clean_path_v19(spk):
    """clean 路径：libritts16k 目录里该说话人时长最长的文件（不是排序后第一个）。
    发现说话人 61 唯一的候选文件只有 0.81s，短于 wavmark 嵌入所需的最小 chunk 长度（约2s），
    必须选最长文件而不是任意/首个文件，否则 wavmark embed 会断言失败。"""
    language, speaker_id = spk.split(":", 1)
    cands = sorted((REAL_ANALYSIS / "collusion_300" / language / speaker_id).glob("*.wav"))
    if not cands:
        raise FileNotFoundError(f"未找到说话人 {spk} 的 clean 音频")
    durs = [(sf.info(str(p)).frames / sf.info(str(p)).samplerate, p) for p in cands]
    return max(durs, key=lambda x: x[0])[1]


def load_clean(spk, sr=16000):
    return load_audio(clean_path_v19(spk), sr)


def full_registry_size(model):
    return 2 ** NBITS[model]


def random_codeword_int(rng, model):
    return int(rng.integers(0, full_registry_size(model)))


def int_to_bits(v, d):
    """LSB-first，与 common.py:130 的身份码约定一致（仅作身份码，不影响正确性，
    只要嵌入与检测/排序两端用同一套编码）。"""
    return np.array([(v >> i) & 1 for i in range(d)], dtype=np.int8)


def get_or_embed(model, spk, codeword_int):
    """按需嵌入并缓存，返回 wav（float32, 16k）。"""
    d = NBITS[model]
    cache_path = CACHE_DIR / model / spk / f"{codeword_int}.wav"
    if cache_path.exists():
        # Several GPU workers can request a shared payload simultaneously.
        # Only accept a complete, finite cache file; otherwise regenerate it.
        try:
            cached, sr = sf.read(cache_path, dtype="float32")
            if sr == 16000 and len(cached) >= 16000 and np.isfinite(cached).all():
                return cached
        except RuntimeError:
            pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    clean = load_clean(spk)
    bits = int_to_bits(codeword_int, d).tolist()
    wm = embed(model, clean, bits)
    # Publish atomically: readers observe either an existing valid WAV or the
    # fully written replacement, never a partial header/body.
    tmp_path = cache_path.parent / f".{codeword_int}.{uuid.uuid4().hex}.wav"
    sf.write(tmp_path, wm, 16000, subtype="FLOAT")
    os.replace(tmp_path, cache_path)
    return wm


def sample_coalition(rng, model, K):
    """从全空间随机采样 K 个不重复的码字整数。"""
    n = full_registry_size(model)
    return sorted(rng.choice(n, size=K, replace=False).tolist())


_REGISTRY_CACHE = {}


def full_registry_bits(model):
    """返回 [2^d, d] 的完整虚拟码本（仅用于检测器打分排序，不对应真实音频）。
    d<=16 时 2^16=65536 行，内存/计算开销可忽略（detect 内部对码本的打分是纯 numpy 向量运算，
    见 common.py 的 _loglik_bits/_loglik_chunks，随候选数线性增长，非瓶颈）。"""
    if model in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[model]
    d = NBITS[model]
    n = 2 ** d
    ints = np.arange(n)
    bits = np.zeros((n, d), dtype=np.int8)
    for i in range(d):
        bits[:, i] = (ints >> i) & 1
    _REGISTRY_CACHE[model] = bits
    return bits
