"""
B: Lyapunov Spectrum & Kaplan–Yorke Dimension
================================================

从自由动力学轨迹估计**完整 Lyapunov 谱**（所有 N 个指数），并据此计算
**Kaplan–Yorke (Lyapunov) 维度** D_KY，回答以下问题：

  系统到底是极限环、准周期、弱混沌还是强混沌？
  190 维系统真实的动力学维度是多少？

方法：数据驱动线性映射近似
--------------------------
给定所有轨迹的相邻状态对 {x(t), x(t+1)}，拟合全局线性映射：

  x(t+1) ≈ A · x(t) + b       (最小二乘)

对 A 进行奇异值分解：A = U·Σ·Vᵀ

有限时间 Lyapunov 指数（FTLE）近似：

  λ_i = log(σ_i) / dt        (自然单位：nats/时间步)

其中 σ_i 为奇异值（σ₁ ≥ σ₂ ≥ ... ≥ σ_N）。

**物理含义**：
- σ_i > 1 → λ_i > 0：扩张方向（混沌成分）
- σ_i < 1 → λ_i < 0：收缩方向（耗散成分）
- σ_i ≈ 1 → λ_i ≈ 0：中性方向（极限环 / Hamiltonian 守恒量）

**Kaplan–Yorke 维度**（Lyapunov 维度）：

  D_KY = j + (Σᵢ₌₁ʲ λ_i) / |λ_{j+1}|

其中 j 是使 Σᵢ₌₁ʲ λ_i ≥ 0 的最大整数（降序排列后）。

D_KY 估计吸引子的分形维度：
- D_KY = 0：固定点（所有 λ < 0）
- D_KY = 1：极限环（λ₁ = 0，其余 < 0）
- 1 < D_KY < N：奇异吸引子（弱/强混沌）

**局限性**：
- 线性映射近似在强非线性区可能低估混沌方向数量。
- 对于 GNN（TwinBrainDigitalTwin），A 是全局有效 Jacobian 的近似；
  真实 Jacobian 随时间变化，此结果反映的是时间平均动力学线性化。
- 建议与 Rosenstein LLE（来自 twinbrain-dynamics 步骤 9）对比 λ₁。

输出文件
--------
  lyapunov_spectrum_{label}.json  — 谱数值 + D_KY + 分类结果
  lyapunov_spectrum_{label}.png   — 谱排名图（线性 + 半对数）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 中性稳定带（判断极限环的 λ₁ 阈值，单位：nats/step）
_NEUTRAL_TOL: float = 0.02
# 弱混沌 / 强混沌分界线
_WEAK_CHAOS_THR: float = 0.01
_STRONG_CHAOS_THR: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lyapunov_spectrum(
    trajectories: np.ndarray,
    dt: float = 1.0,
    burnin: int = 10,
    center: bool = True,
) -> Dict:
    """
    从轨迹数据估计完整 Lyapunov 谱及 Kaplan–Yorke 维度。

    使用数据驱动线性映射法：
      1. 拼接所有轨迹的相邻对 (x_t, x_{t+1})，跳过前 burnin 步瞬态。
      2. 拟合线性映射 A = X_next · pinv(X_curr)（最小二乘）。
      3. 计算 A 的奇异值 → 有限时间 Lyapunov 谱。
      4. 据谱计算 Kaplan–Yorke 维度 D_KY。

    Args:
        trajectories: shape (n_traj, T, N)，已去除瞬态（或通过 burnin 自动跳过）。
        dt:           时间步长（秒，用于将 λ 换算为 1/s）。
        burnin:       每条轨迹跳过开头的步数（消除瞬态）。
        center:       是否在拟合前去均值（减少 DC 偏置的影响）。

    Returns:
        dict 包含:
          spectrum           : np.ndarray (N,)，降序 Lyapunov 指数（nats/step）
          spectrum_per_sec   : np.ndarray (N,)，λ / dt（1/s）
          kaplan_yorke_dim   : float，D_KY
          lambda1            : float，最大 Lyapunov 指数
          n_positive         : int，λ > neutral_tol 的数量
          n_neutral          : int，|λ| ≤ neutral_tol 的数量
          n_negative         : int，λ < -neutral_tol 的数量
          sum_positive       : float，正 λ 之和（Kolmogorov–Sinai 熵估计）
          classification     : str，动力学分类
          n_pairs            : int，用于拟合的状态对数量
    """
    n_traj, T, N = trajectories.shape

    # ── Build (X_curr, X_next) pairs from all trajectories (skip burnin) ──────
    pairs: List[np.ndarray] = []
    for traj in trajectories:
        if T - burnin < 2:
            continue
        seg = traj[burnin:]  # (T-burnin, N)
        pairs.append(seg)

    if not pairs:
        raise ValueError(
            f"没有足够的轨迹数据（n_traj={n_traj}, T={T}, burnin={burnin}）。"
        )

    # Stack: X_curr = all x(t), X_next = all x(t+1)
    stacked = np.vstack(pairs)  # (M, N), M = n_traj * (T-burnin)
    X_curr = stacked[:-len(pairs)]   # exclude last step of each trajectory
    X_next = stacked[1:]             # exclude first step of each trajectory

    # Remove the cross-trajectory transitions (last→first)
    # We rebuild properly:
    X_curr_list, X_next_list = [], []
    for seg in pairs:
        X_curr_list.append(seg[:-1])   # x(t)   for t in [0, T-burnin-2]
        X_next_list.append(seg[1:])    # x(t+1) for t in [1, T-burnin-1]
    X_curr = np.vstack(X_curr_list).astype(np.float64)  # (M, N)
    X_next = np.vstack(X_next_list).astype(np.float64)  # (M, N)

    if center:
        mu = X_curr.mean(axis=0, keepdims=True)
        X_curr = X_curr - mu
        X_next = X_next - mu

    M = X_curr.shape[0]
    logger.info("B: 拟合线性映射，状态对数 M=%d, N=%d", M, N)

    # ── Fit linear map: A = X_next^T @ pinv(X_curr^T)
    # Equivalent: solve A such that X_next^T ≈ A @ X_curr^T (least squares)
    # Using SVD-based pseudoinverse for numerical stability.
    # A = (X_next^T @ X_curr) @ pinv(X_curr^T @ X_curr)
    # Efficient form via np.linalg.lstsq:
    # X_curr @ A^T = X_next   →  A^T = lstsq(X_curr, X_next)[0]
    try:
        A_T, _, _, _ = np.linalg.lstsq(X_curr, X_next, rcond=None)
        A = A_T.T  # (N, N)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"线性映射拟合失败: {exc}") from exc

    # ── Compute singular values → Lyapunov spectrum ───────────────────────────
    # σ_i of A → λ_i = log(σ_i)
    singular_values = np.linalg.svd(A, compute_uv=False)  # (N,), descending
    # Avoid log(0): clip singular values to minimum 1e-30
    singular_values = np.maximum(singular_values, 1e-30)
    spectrum = np.log(singular_values)  # (N,), descending order (nats/step)

    # ── Kaplan–Yorke dimension ────────────────────────────────────────────────
    dky = kaplan_yorke_dimension(spectrum)

    # ── Classification ────────────────────────────────────────────────────────
    n_pos = int((spectrum > _NEUTRAL_TOL).sum())
    n_neu = int((np.abs(spectrum) <= _NEUTRAL_TOL).sum())
    n_neg = int((spectrum < -_NEUTRAL_TOL).sum())
    l1 = float(spectrum[0])
    ks_entropy = float(spectrum[spectrum > 0].sum())

    if l1 < -_NEUTRAL_TOL:
        classification = "fixed_point"
    elif abs(l1) <= _NEUTRAL_TOL and n_pos == 0:
        classification = "limit_cycle"
    elif l1 > _STRONG_CHAOS_THR:
        classification = "strongly_chaotic"
    elif l1 > _WEAK_CHAOS_THR:
        classification = "weakly_chaotic"
    else:
        classification = "quasi_periodic"

    logger.info(
        "B: λ₁=%.4f, D_KY=%.2f, n_pos=%d, n_neutral=%d, 分类=[%s]",
        l1, dky, n_pos, n_neu, classification,
    )

    return {
        "spectrum": spectrum,
        "spectrum_per_sec": spectrum / max(dt, 1e-12),
        "kaplan_yorke_dim": round(dky, 4),
        "lambda1": round(l1, 6),
        "n_positive": n_pos,
        "n_neutral": n_neu,
        "n_negative": n_neg,
        "sum_positive": round(ks_entropy, 6),
        "classification": classification,
        "n_pairs": M,
    }


def kaplan_yorke_dimension(spectrum: np.ndarray) -> float:
    """
    从有序 Lyapunov 谱（降序）计算 Kaplan–Yorke 维度。

      D_KY = j + (Σᵢ₌₁ʲ λ_i) / |λ_{j+1}|

    其中 j 是使 Σᵢ₌₁ʲ λ_i ≥ 0 的最大整数。

    - 若 λ₁ < 0（固定点），返回 0.0。
    - 若 Σ 所有 λ ≥ 0（Hamiltonian/保守系统），返回 N（最大维度）。

    Args:
        spectrum: 降序排列的 Lyapunov 指数数组。

    Returns:
        D_KY: Kaplan–Yorke 维度（float）。
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    # Ensure descending order
    spectrum = np.sort(spectrum)[::-1]
    N = len(spectrum)

    if N == 0 or spectrum[0] < 0:
        return 0.0

    cumsum = np.cumsum(spectrum)

    # Find largest j such that cumsum[j-1] >= 0  (1-indexed j, 0-indexed array)
    j = int(np.sum(cumsum >= 0))  # number of terms with non-negative partial sum

    if j >= N:
        # All partial sums non-negative → conservative system
        return float(N)

    if j == 0:
        return 0.0

    total_pos = float(cumsum[j - 1])    # Σᵢ₌₁ʲ λᵢ  (0-indexed: cumsum[j-1])
    next_lambda = float(spectrum[j])    # λ_{j+1}    (0-indexed: spectrum[j])

    if abs(next_lambda) < 1e-20:
        return float(j)

    return float(j) + total_pos / abs(next_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Integration function
# ─────────────────────────────────────────────────────────────────────────────

def run_lyapunov_spectrum(
    trajectories: np.ndarray,
    dt: float = 1.0,
    burnin: int = 10,
    output_dir: Optional[Path] = None,
    label: str = "traj",
) -> Dict:
    """
    运行 Lyapunov 谱分析并保存结果。

    Args:
        trajectories: shape (n_traj, T, N)。
        dt:           时间步长（秒），用于换算单位。默认 1（以步为单位）。
        burnin:       跳过每条轨迹开头的步数（去除瞬态）。
        output_dir:   结果保存目录（None → 不保存）。
        label:        文件名标签。

    Returns:
        dict（compute_lyapunov_spectrum 的输出 + 文件路径）。
    """
    result = compute_lyapunov_spectrum(trajectories, dt=dt, burnin=burnin)

    # Convert numpy arrays to lists for JSON serialization
    json_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in result.items()}

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / f"lyapunov_spectrum_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        logger.info("B: 保存 Lyapunov 谱: %s", json_path)

        _try_plot_spectrum(result["spectrum"], out / f"lyapunov_spectrum_{label}.png", label)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_spectrum(
    spectrum: np.ndarray,
    output_path: Path,
    label: str,
) -> None:
    """绘制 Lyapunov 谱排名图（线性 + 半对数）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    N = len(spectrum)
    ranks = np.arange(1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: linear scale
    ax = axes[0]
    pos_mask = spectrum > 0
    neg_mask = spectrum <= 0
    ax.bar(ranks[pos_mask], spectrum[pos_mask], color="tomato", label="λ > 0")
    ax.bar(ranks[neg_mask], spectrum[neg_mask], color="steelblue", label="λ ≤ 0")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("λ (nats/step)")
    ax.set_title(f"Lyapunov Spectrum [{label}]")
    ax.legend(fontsize=8)
    ax.set_xlim(0, N + 1)

    # Right: cumulative sum
    ax2 = axes[1]
    cumsum = np.cumsum(spectrum)
    ax2.plot(ranks, cumsum, "k-", lw=1.5)
    ax2.axhline(0, color="red", lw=0.8, ls="--", label="Σλ = 0")
    ax2.fill_between(ranks, cumsum, 0, where=cumsum >= 0,
                     alpha=0.2, color="tomato", label="Σλ ≥ 0")
    ax2.fill_between(ranks, cumsum, 0, where=cumsum < 0,
                     alpha=0.2, color="steelblue", label="Σλ < 0")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Cumulative Σλ")
    ax2.set_title("Cumulative Lyapunov Sum")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("B: 保存谱图: %s", output_path)
