"""minimax CB 变体：min_α max_i [Gα]_i（最小化最强成员证据）。

性质验证（STATUS 3.4）：
① 对称 Gram（等距码字）下是否退化为 Mean
② 不规则几何下是否允许非均匀权重
③ 与标准 CB 的比较

G = 码字 Gram（±1 编码），[Gα]_i = 成员 i 的残留证据强度。
minimax: 让最强的成员证据最小。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from convex import qp_from_gram  # noqa: E402


def gram_from_bits(C):
    """C [K,d] {0,1} -> G [K,K] ±1 内积。"""
    Cpm = C.astype(float) * 2 - 1
    return Cpm @ Cpm.T


def minimax_weights(G, cap=0.5):
    """min_α max_i [Gα]_i，约束 Σα=1, 0≤α≤cap。用光滑近似（softmax-LSE）解。"""
    K = G.shape[0]
    Gs = (G + G.T) / 2

    def obj(a):
        a = np.asarray(a, float)
        evidence = Gs @ a  # [K] 每个成员的残留证据
        # max 的光滑近似：LSE
        m = evidence.max()
        lse = m + np.log(np.sum(np.exp(evidence - m)))
        return float(lse)

    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    res = minimize(obj, np.full(K, 1.0 / K), method="SLSQP",
                   bounds=[(0, cap)] * K, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-14})
    a = np.clip(res.x, 0, cap)
    s = a.sum()
    return a / s if s > 1e-8 else np.ones(K) / K


def check_symmetry():
    """性质①：对称 Gram（等距码字，如正交）→ minimax 应≈Mean。"""
    K = 5
    # 正交码字：单位阵的 ±1 版本（Gram = K*I 对角）
    G_orth = np.eye(K) * K
    a_mm = minimax_weights(G_orth)
    a_mean = np.ones(K) / K
    print("性质① 对称 Gram（正交）：")
    print(f"  minimax = {np.round(a_mm, 4)}")
    print(f"  mean    = {np.round(a_mean, 4)}")
    print(f"  退化到 Mean? {np.allclose(a_mm, a_mean, atol=0.05)}")


def check_irregular():
    """性质②：不规则几何（码字聚集）→ 非均匀权重。"""
    K = 5
    d = 8
    rng = np.random.default_rng(0)
    # 制造不规则：前 3 个码字很近，后 2 个远
    C = rng.integers(0, 2, size=(K, d))
    C[1] = C[0]  # 完全重合 → 高度相关
    C[2] = C[0]
    C[4] = 1 - C[0]  # 完全相反
    G = gram_from_bits(C)
    a_mm = minimax_weights(G)
    a_std = qp_from_gram(G, 0.5)
    print("性质② 不规则几何：")
    print(f"  minimax = {np.round(a_mm, 3)}（非均匀? {len(set(np.round(a_mm, 2))) > 2}）")
    print(f"  标准CB = {np.round(a_std, 3)}")


def main():
    check_symmetry()
    print()
    check_irregular()


if __name__ == "__main__":
    main()
