"""
E1: Connectome Spectral Analysis
==================================

检测连接矩阵的谱结构，验证假设 H1：

  H1  connectome 的谱结构是低秩的（少数特征值主导）

计算指标
--------
1. **谱半径** ρ(W) = max|λ_k|
2. **谱有效维度**（参与率 PR） = (Σ|λ_k|)² / Σ(|λ_k|²)，范围 [1, N]
3. **主导特征值数量**：|λ_k| > threshold × |λ_1| 的 k 数量
4. **谱间隙比**：|λ_1| / |λ_2|（越大表示谱越集中）
5. **特征值分布**：复平面散点图 + 幅值排名图

批判性注意事项
--------------
- 若使用响应矩阵 R（非对称），使用 ``np.linalg.eig``（复数特征值）。
- 若使用功能连接矩阵 FC（对称），使用 ``np.linalg.eigh``（实数特征值）。
- 谱半径 ρ ≈ 1 不意味着临界——这取决于非线性（tanh），需结合 E5 相图判断。
- 参与率 PR << N 是低秩结构的定量证据，但不直接等于动力学维度（需结合 E2/E3）。

输出文件（保存到 output_dir）
-----------------------------
  spectral_summary.json    — 所有数值指标
  eigenvalue_complex.png   — 复平面特征值散点图
  eigenvalue_rank.png      — 特征值幅值按秩排序图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .compute_connectivity import participation_ratio

logger = logging.getLogger(__name__)

# 主导特征值的幅值阈值（相对于最大特征值）
_DOMINANT_THRESHOLD = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_metrics(W: np.ndarray, symmetric: bool = False) -> Dict:
    """
    计算连接矩阵的谱特征指标。

    Args:
        W:          方阵 (N, N)，float。
        symmetric:  若 True，使用 eigh（实数特征值，速度更快）。
                    对响应矩阵 R（非对称）应设为 False。

    Returns:
        dict 包含:
          spectral_radius         : float，max|λ_k|
          participation_ratio     : float，谱有效维度
          n_dominant              : int，|λ_k| > 0.2×|λ_1| 的数量
          spectral_gap_ratio      : float，|λ_1|/|λ_2|
          top_k_eigenvalue_share  : float，前 k=5 特征值幅值之和占比
          eigenvalues             : np.ndarray，所有特征值（复数）
          eigenvalue_magnitudes   : np.ndarray，幅值排序后的结果
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]

    n_bad = (~np.isfinite(W)).sum()
    if n_bad > 0:
        logger.warning(
            "compute_spectral_metrics: %d non-finite value(s) in input matrix "
            "(NaN/Inf); replacing with 0 to allow spectral computation.",
            n_bad,
        )
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    if symmetric:
        # Real eigenvalues only, sorted ascending
        eigvals_real = np.linalg.eigh(W)[0]
        eigvals = eigvals_real.astype(np.complex128)
    else:
        eigvals = np.linalg.eigvals(W)

    mags = np.abs(eigvals)
    idx_sorted = np.argsort(mags)[::-1]
    mags_sorted = mags[idx_sorted]
    eigvals_sorted = eigvals[idx_sorted]

    rho = float(mags_sorted[0]) if N > 0 else 0.0
    gap_ratio = float(mags_sorted[0] / (mags_sorted[1] + 1e-30)) if N > 1 else 1.0

    n_dominant = int((mags_sorted >= _DOMINANT_THRESHOLD * rho).sum())

    pr = participation_ratio(mags_sorted)

    k = min(5, N)
    top_k_share = float(mags_sorted[:k].sum() / (mags_sorted.sum() + 1e-30))

    logger.info(
        "谱分析完成: N=%d, ρ=%.4f, PR=%.1f, n_dominant=%d, gap_ratio=%.2f, "
        "top5_share=%.2f",
        N, rho, pr, n_dominant, gap_ratio, top_k_share,
    )

    return {
        "n_regions": N,
        "spectral_radius": rho,
        "participation_ratio": pr,
        "n_dominant": n_dominant,
        "spectral_gap_ratio": gap_ratio,
        "top_k_eigenvalue_share": top_k_share,
        "eigenvalues": eigvals_sorted,
        "eigenvalue_magnitudes": mags_sorted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers (optional – gracefully skipped if matplotlib unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_complex(eigvals: np.ndarray, output_path: Path, label: str) -> None:
    """Scatter plot of eigenvalues in the complex plane."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    re = eigvals.real
    im = eigvals.imag
    mags = np.abs(eigvals)
    sc = ax.scatter(re, im, c=mags, cmap="plasma", s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="|λ|")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4, label="unit circle")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"Eigenvalue Complex Plane  [{label}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


def _try_plot_rank(mags: np.ndarray, output_path: Path, label: str) -> None:
    """Eigenvalue magnitude ranked plot (power-law vs flat spectrum)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    N = len(mags)
    ranks = np.arange(1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Linear scale
    axes[0].plot(ranks, mags, "o-", ms=3, lw=1.5)
    axes[0].axhline(mags[0] * _DOMINANT_THRESHOLD, ls="--", color="red",
                    label=f"threshold {_DOMINANT_THRESHOLD:.0%}x|lambda_1|")
    axes[0].set_xlabel("Rank (|λ| descending)")
    axes[0].set_ylabel("|λ|")
    axes[0].set_title(f"Eigenvalue Magnitude Rank (linear)  [{label}]")
    axes[0].legend(fontsize=8)

    # Log scale
    axes[1].semilogy(ranks, mags + 1e-12, "o-", ms=3, lw=1.5)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("|λ|  (log)")
    axes[1].set_title(f"Eigenvalue Magnitude Rank (log)  [{label}]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_spectral_analysis(
    W: np.ndarray,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    symmetric: bool = False,
) -> Dict:
    """
    运行 E1 谱分析并保存结果。

    Args:
        W:          连接矩阵 (N, N)。
        output_dir: 结果保存目录；None → 仅返回不保存。
        label:      矩阵标签（用于文件名和标题），如 "response_matrix" / "fc".
        symmetric:  是否对称（影响特征值算法选择）。

    Returns:
        metrics dict（不含 numpy 数组，可直接序列化为 JSON）。
    """
    metrics = compute_spectral_metrics(W, symmetric=symmetric)

    # Serializable subset (no numpy arrays)
    result = {k: v for k, v in metrics.items()
              if not isinstance(v, np.ndarray)}

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = out / f"spectral_summary_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("保存谱分析摘要: %s", json_path)

        # Save eigenvalue arrays
        np.save(out / f"eigenvalues_{label}.npy", metrics["eigenvalues"])

        # Plots
        _try_plot_complex(
            metrics["eigenvalues"],
            out / f"eigenvalue_complex_{label}.png",
            label,
        )
        _try_plot_rank(
            metrics["eigenvalue_magnitudes"],
            out / f"eigenvalue_rank_{label}.png",
            label,
        )

    return result
