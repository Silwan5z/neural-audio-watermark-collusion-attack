"""盲 CB（blind Code-Balanced）：不知道 payload 的凸混合攻击。

攻击者只有 K 个合法水印副本波形，不知道任何 payload（coalition 码字、target 码字）。

核心：用波形残差（median 消除公共内容）的余弦相关矩阵 Ĝ 估计真实码字 Gram，
再解凸优化 min aᵀĜa。

- residual_gram(wavs): K 个副本 → Ĝ（估计 Gram）
- blind_cb_weights(wavs, cap): K 个副本 → 凸混合权重 a（盲，不用码字）

用于攻击（evasion）。篡改用 exact framing（知道 target，见 v18_tamper）。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def residual_gram(wavs):
    """K 个副本波形 -> 残差余弦相关矩阵 Ĝ（估计码字 Gram，盲）。"""
    n = min(len(w) for w in wavs)
    X = np.stack([w[:n] for w in wavs])
    m = np.median(X, axis=0)
    R = X - m
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    return Rn @ Rn.T


def qp_from_gram(G, cap):
    """由 Gram（估计或真实）解凸混合权重 a = argmin aᵀGa。"""
    Gs = (G + G.T) / 2
    K = Gs.shape[0]
    def obj(a):
        a = np.asarray(a, float)
        return float(a @ Gs @ a)
    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    res = minimize(obj, np.full(K, 1.0 / K), method="SLSQP",
                   bounds=[(0, cap)] * K, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-14})
    a = np.clip(res.x, 0, cap)
    s = a.sum()
    return a / s if s > 1e-8 else np.ones(K) / K


def blind_cb_weights(wavs, cap=0.5):
    """盲 CB 权重：只用波形残差，不用任何 payload。"""
    G = residual_gram(wavs)
    return qp_from_gram(G, cap)
