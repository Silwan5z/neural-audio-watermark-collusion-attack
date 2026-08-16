"""
音频质量评估指标模块
包含 PESQ、SI-SNR、STOI、ViSQOL 四个评估维度
"""

import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    读取音频文件，返回 (waveform, sample_rate)
    waveform shape: (samples,)  单声道 float32
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)          # 多声道 → 单声道
    if target_sr is not None and sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def align_length(ref: np.ndarray, deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将两段音频截断/补零到相同长度"""
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]


# ─────────────────────────────────────────────
# PESQ
# ─────────────────────────────────────────────

def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    计算 PESQ (Perceptual Evaluation of Speech Quality)
    支持采样率: 8000 Hz (narrowband) 或 16000 Hz (wideband)
    返回 MOS-LQO 分数，范围约 [-0.5, 4.5]

    Args:
        ref: 参考音频 (干净语音)
        deg: 降质音频 (待评估)
        sr:  采样率，必须为 8000 或 16000

    Returns:
        PESQ 分数 (float)，失败时返回 float('nan')
    """
    try:
        from pesq import pesq, PesqError
    except ImportError:
        raise ImportError("请安装 pesq 库: pip install pesq")

    if sr not in (8000, 16000):
        raise ValueError(f"PESQ 仅支持 8000 或 16000 Hz，当前采样率: {sr}")

    mode = "wb" if sr == 16000 else "nb"
    ref, deg = align_length(ref, deg)

    try:
        score = pesq(sr, ref, deg, mode)
    except PesqError as e:
        print(f"  [PESQ] 计算失败: {e}")
        score = float("nan")
    return float(score)


# ─────────────────────────────────────────────
# SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
# ─────────────────────────────────────────────

def compute_sisnr(ref: np.ndarray, deg: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算 SI-SNR (尺度不变信噪比)
    公式参考: Chen et al., "Deep Attractor Network for Single-microphone Speaker Separation"

    SI-SNR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
    其中:
        s_target = (<s_hat, s> / ||s||^2) * s
        e_noise  = s_hat - s_target

    Args:
        ref: 参考信号 (干净语音)
        deg: 估计信号 (待评估)
        eps: 数值稳定性小量

    Returns:
        SI-SNR 分数 (dB, float)
    """
    ref, deg = align_length(ref, deg)

    # 零均值化
    ref = ref - ref.mean()
    deg = deg - deg.mean()

    # 投影
    dot = np.dot(deg, ref)
    s_target = (dot / (np.dot(ref, ref) + eps)) * ref

    # 噪声分量
    e_noise = deg - s_target

    sisnr = 10.0 * np.log10(
        (np.dot(s_target, s_target) + eps) /
        (np.dot(e_noise, e_noise) + eps)
    )
    return float(sisnr)


# ─────────────────────────────────────────────
# STOI
# ─────────────────────────────────────────────

def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int,
                 extended: bool = False) -> float:
    """
    计算 STOI (Short-Time Objective Intelligibility)
    范围 [0, 1]，越高越好

    Args:
        ref:      参考音频 (干净语音)
        deg:      降质音频 (待评估)
        sr:       采样率
        extended: 是否使用 Extended STOI (ESTOI)，对低 SNR 更鲁棒

    Returns:
        STOI 分数 (float)，失败时返回 float('nan')
    """
    try:
        from pystoi import stoi
    except ImportError:
        raise ImportError("请安装 pystoi 库: pip install pystoi")

    ref, deg = align_length(ref, deg)

    try:
        score = stoi(ref, deg, sr, extended=extended)
    except Exception as e:
        print(f"  [STOI] 计算失败: {e}")
        score = float("nan")
    return float(score)


# ─────────────────────────────────────────────
# ViSQOL
# ─────────────────────────────────────────────

def _visqol_via_pyvisqol(ref_path: str, deg_path: str,
                         use_speech_mode: bool = True) -> float:
    """
    使用 pyvisqol 库计算 ViSQOL
    安装: pip install pyvisqol -i https://pypi.org/simple/

    pyvisqol 是轻量级封装，内部调用预编译的 ViSQOL 二进制，
    无需 Bazel，API 简洁：
        from pyvisqol import visqol
        score = visqol(ref_path, deg_path, mode="speech")  # 返回 MOS-LQO float
    """
    from pyvisqol import visqol as pyvisqol_fn

    mode = "speech" if use_speech_mode else "audio"
    result = pyvisqol_fn(str(ref_path), str(deg_path), mode=mode)

    # pyvisqol 可能返回 float 或带 moslqo 属性的对象
    if isinstance(result, (int, float)):
        return float(result)
    elif hasattr(result, "moslqo"):
        return float(result.moslqo)
    else:
        return float(result)


def _visqol_via_official(ref_path: str, deg_path: str,
                         use_speech_mode: bool = True) -> float:
    """
    使用官方 visqol 库计算 ViSQOL
    安装: pip install visqol  （需要预编译环境或 Bazel）
    """
    import visqol
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2

    config = visqol_config_pb2.VisqolConfig()
    if use_speech_mode:
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model = ("lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048"
                     "_learn.005_ep2400_train1_7_raw.tflite")
    else:
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model = "libsvm_nu_svr_model.txt"

    config.options.svr_model_path = str(
        visqol.resource_path(f"model/{svr_model}")
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    similarity_result = api.Measure(str(ref_path), str(deg_path))
    return float(similarity_result.moslqo)


def compute_visqol(ref_path: str, deg_path: str,
                   use_speech_mode: bool = True) -> float:
    """
    计算 ViSQOL (Virtual Speech Quality Objective Listener)
    MOS-LQO 分数，范围 [1, 5]，越高越好

    自动按优先级尝试以下库：
        1. pyvisqol  （推荐，无需 Bazel，pip install pyvisqol）
        2. visqol    （官方库，需要预编译环境）

    Args:
        ref_path:        参考音频文件路径（干净语音）
        deg_path:        降质音频文件路径（待评估）
        use_speech_mode: True  → speech 模式，16kHz，适合语音（默认）
                         False → audio  模式，48kHz，适合音乐/宽带

    Returns:
        ViSQOL MOS-LQO 分数 (float)，失败时返回 float('nan')
    """
    # ── 优先尝试 pyvisqol ──
    # try:
    #     import pyvisqol  # noqa: F401
    #     _backend = "pyvisqol"
    # except ImportError:
    #     _backend = None

    # ── 其次尝试官方 visqol ──
    # if _backend is None:
    #     try:
    #         import visqol  # noqa: F401
    #         _backend = "visqol"
    #     except ImportError:
    #         _backend = None

    # if _backend is None:
    #     raise ImportError(
    #         "未找到 ViSQOL 库，请安装以下任意一个：\n"
    #         "  推荐（无需 Bazel）: pip install pyvisqol -i https://pypi.org/simple/\n"
    #         "  官方版本          : pip install visqol"
    #     )

    try:
        # if _backend == "pyvisqol":
        score = _visqol_via_pyvisqol(ref_path, deg_path, use_speech_mode)
        # else:
            # score = _visqol_via_official(ref_path, deg_path, use_speech_mode)
    except Exception as e:
        print(f"  [ViSQOL/_backend] 计算失败: {e}")
        score = float("nan")

    return float(score)


# ─────────────────────────────────────────────
# 统一评估接口
# ─────────────────────────────────────────────

def evaluate_pair(
    ref_path: str,
    deg_path: str,
    metrics: Tuple[str, ...] = ("pesq", "sisnr", "stoi", "visqol"),
    pesq_sr: int = 16000,
    stoi_extended: bool = False,
    visqol_speech_mode: bool = True,
) -> Dict[str, float]:
    """
    对一对音频文件计算所有指定指标

    Args:
        ref_path:          参考音频路径（干净语音）
        deg_path:          待评估音频路径
        metrics:           需要计算的指标列表
        pesq_sr:           PESQ 使用的采样率 (8000 或 16000)
        stoi_extended:     是否使用 Extended STOI
        visqol_speech_mode: ViSQOL 是否使用 speech 模式

    Returns:
        dict，键为指标名，值为分数
    """
    results: Dict[str, float] = {}

    # ── 加载音频（PESQ/STOI/SI-SNR 共用） ──
    need_waveform = any(m in metrics for m in ("pesq", "sisnr", "stoi"))
    if need_waveform:
        ref_wav, ref_sr = load_audio(ref_path)
        deg_wav, deg_sr = load_audio(deg_path)

        # 统一采样率（以参考音频为准）
        if ref_sr != deg_sr:
            deg_wav = librosa.resample(deg_wav, orig_sr=deg_sr, target_sr=ref_sr)
            deg_sr = ref_sr

    # ── PESQ ──
    if "pesq" in metrics:
        # PESQ 只支持 8k / 16k，需要重采样
        if ref_sr not in (8000, 16000):
            pesq_sr_use = 16000
        else:
            pesq_sr_use = ref_sr

        ref_pesq = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=pesq_sr_use) \
            if ref_sr != pesq_sr_use else ref_wav
        deg_pesq = librosa.resample(deg_wav, orig_sr=deg_sr, target_sr=pesq_sr_use) \
            if deg_sr != pesq_sr_use else deg_wav

        results["pesq"] = compute_pesq(ref_pesq, deg_pesq, pesq_sr_use)

    # ── SI-SNR ──
    if "sisnr" in metrics:
        results["sisnr"] = compute_sisnr(ref_wav, deg_wav)

    # ── STOI ──
    if "stoi" in metrics:
        results["stoi"] = compute_stoi(ref_wav, deg_wav, ref_sr, extended=stoi_extended)

    # ── ViSQOL ──
    if "visqol" in metrics:
        results["visqol"] = compute_visqol(ref_path, deg_path,
                                           use_speech_mode=visqol_speech_mode)

    return results
