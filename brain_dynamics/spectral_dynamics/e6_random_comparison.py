"""
E6: Random Network Spectral Comparison
========================================

将真实连接矩阵的谱结构与三类随机对照网络对比，
验证"低秩谱结构是真实连接组特有的"（而非随机网络偶然性质）。

三类随机对照
------------
1. **Erdős–Rényi (ER)**：随机稠密矩阵，谱半径归一化到 ρ(W)_real。
2. **保度随机化（DPR）**：Maslov-Sneppen 保留节点度分布。
3. **权重混洗（WS）**：保留拓扑，随机打乱连接权重。

比较指标（纯谱指标，无需运行任何模型）
--------------------------------------
  - 谱有效维度（参与率 PR）
  - 主导特征值数（n_dominant）
  - 谱半径（spectral_radius）
  - 谱间隙比（spectral_gap_ratio）

**核心假设验证**：
若真实连接矩阵 PR << ER 的 PR（均值 - 2σ 以下），则说明真实网络
具有统计显著的低秩谱结构，支持假设 H1。

输出文件
--------
  random_comparison_spectral_{label}.json   — 数值汇总
  random_comparison_spectral_{label}.png    — 小提琴图对比
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .e1_spectral_analysis import compute_spectral_metrics
from .e4_structural_perturbation import (
    weight_shuffle,
    degree_preserving_rewire,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Random network generators
# ─────────────────────────────────────────────────────────────────────────────

def erdos_renyi(n: int, target_rho: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    生成 Erdős–Rényi 随机矩阵（高斯稠密），谱半径归一化到 target_rho。

    Args:
        n:           矩阵维度。
        target_rho:  目标谱半径。
        seed:        随机种子。

    Returns:
        W_er: shape (n, n), float32。
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n)).astype(np.float64) / np.sqrt(n)
    eigvals = np.linalg.eigvals(W)
    rho = float(np.abs(eigvals).max())
    if rho > 1e-8:
        W = W * (target_rho / rho)
    return W.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-matrix metrics helper (pure spectral, no dynamics simulation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(W: np.ndarray) -> Dict:
    """Compute pure spectral metrics for a single matrix."""
    spec = compute_spectral_metrics(W, symmetric=False)
    return {
        "spectral_radius": spec["spectral_radius"],
        "participation_ratio": spec["participation_ratio"],
        "n_dominant": spec["n_dominant"],
        "spectral_gap_ratio": spec["spectral_gap_ratio"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_random_spectral_comparison(
    W: np.ndarray,
    n_random: int = 10,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    seed: int = 42,
    # Legacy params kept for backward-compat (ignored)
    n_traj_lle: int = 0,
    steps_lle: int = 0,
    g_lle: float = 0.9,
) -> Dict:
    """
    运行 E6：真实矩阵 vs 三类随机对照的谱结构比较。

    纯矩阵谱分析，不运行任何动力学模拟。

    Args:
        W:          真实连接矩阵 (N, N)。
        n_random:   每类对照的随机实现数。
        output_dir: 结果保存目录。
        label:      矩阵标签。
        seed:       随机种子。

    Returns:
        summary dict（可序列化为 JSON）。
    """
    N = W.shape[0]
    rng_seeds = list(range(seed, seed + n_random))

    logger.info("E6 随机网络谱比较: N=%d, n_random=%d", N, n_random)

    # 1. Real network
    real_metrics = _compute_metrics(W)

    # 2. ER random
    target_rho = real_metrics["spectral_radius"]
    er_list = [_compute_metrics(erdos_renyi(N, target_rho=target_rho, seed=s))
               for s in rng_seeds]

    # 3. Degree-preserving rewire
    dpr_list = [_compute_metrics(degree_preserving_rewire(W, seed=s))
                for s in rng_seeds]

    # 4. Weight shuffle
    ws_list = [_compute_metrics(weight_shuffle(W, seed=s))
               for s in rng_seeds]

    def _agg(lst: List[Dict]) -> Dict:
        keys = [k for k in lst[0] if isinstance(lst[0][k], (int, float))]
        out: Dict = {}
        for k in keys:
            vals = np.array([d[k] for d in lst if np.isfinite(d.get(k, float("nan")))])
            out[f"{k}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            out[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else float("nan")
        return out

    er_agg  = _agg(er_list)
    dpr_agg = _agg(dpr_list)
    ws_agg  = _agg(ws_list)

    def _z(val: float, mean: float, std: float) -> float:
        return (val - mean) / std if std > 1e-10 else float("nan")

    result: Dict = {
        "real": real_metrics,
        "er_random": er_agg,
        "degree_preserving_rewire": dpr_agg,
        "weight_shuffle": ws_agg,
        "z_scores_vs_er": {
            "pr_z": _z(
                real_metrics["participation_ratio"],
                er_agg["participation_ratio_mean"],
                er_agg["participation_ratio_std"],
            ),
            "n_dominant_z": _z(
                real_metrics["n_dominant"],
                er_agg["n_dominant_mean"],
                er_agg["n_dominant_std"],
            ),
            "gap_ratio_z": _z(
                real_metrics["spectral_gap_ratio"],
                er_agg["spectral_gap_ratio_mean"],
                er_agg["spectral_gap_ratio_std"],
            ),
        },
        "h1_supported": bool(
            _z(
                real_metrics["participation_ratio"],
                er_agg["participation_ratio_mean"],
                er_agg["participation_ratio_std"],
            ) < -2.0
        ),
        "config": {"n_random": n_random},
    }

    logger.info(
        "E6 结果: PR_real=%.1f, PR_ER=%.1f±%.1f, z=%.2f, H1_supported=%s",
        real_metrics["participation_ratio"],
        er_agg["participation_ratio_mean"],
        er_agg["participation_ratio_std"],
        result["z_scores_vs_er"]["pr_z"],
        result["h1_supported"],
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"random_comparison_spectral_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("保存谱比较摘要: %s", json_path)
        _try_plot_spectral_comparison(
            real_metrics, er_list, dpr_list, ws_list,
            out / f"random_comparison_spectral_{label}.png",
            label,
        )

    return result


def _try_plot_spectral_comparison(
    real_metrics: Dict,
    er_list: List[Dict],
    dpr_list: List[Dict],
    ws_list: List[Dict],
    output_path: Path,
    label: str,
) -> None:
    """Violin/box plot comparison: PR, n_dominant, gap_ratio, spectral_radius."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    metrics_keys  = ["participation_ratio", "n_dominant", "spectral_gap_ratio", "spectral_radius"]
    metric_labels = ["Effective dim. PR", "Dominant eigenvalues", "Spectral gap ratio", "Spectral radius rho"]

    fig, axes = plt.subplots(1, len(metrics_keys), figsize=(14, 5))
    group_names  = ["Real", "ER", "DPR", "WS"]
    group_lists  = [None, er_list, dpr_list, ws_list]
    group_colors = ["steelblue", "tomato", "darkorange", "forestgreen"]

    for ax, key, ml in zip(axes, metrics_keys, metric_labels):
        real_val = real_metrics.get(key, float("nan"))
        ax.axhline(real_val, ls="--", color="steelblue", lw=1.5, label="Real")

        positions: list = []
        data_groups: list = []
        colors_used: list = []

        for i, (name, lst, color) in enumerate(
            zip(group_names[1:], group_lists[1:], group_colors[1:]), start=1
        ):
            vals = [d.get(key, float("nan")) for d in lst
                    if np.isfinite(d.get(key, float("nan")))]
            if vals:
                data_groups.append(vals)
                positions.append(i)
                colors_used.append(color)

        if data_groups:
            vp = ax.violinplot(data_groups, positions=positions, showmedians=True)
            for pc, col in zip(vp["bodies"], colors_used):
                pc.set_facecolor(col)
                pc.set_alpha(0.6)
            vp["cmedians"].set_color("black")
            ax.set_xticks([0] + positions)
            ax.set_xticklabels(["Real"] + group_names[1:len(data_groups) + 1], fontsize=8)
            ax.scatter([0], [real_val], color="steelblue", zorder=5, s=60)

        ax.set_ylabel(ml)
        ax.set_title(ml)

    fig.suptitle(f"Random Network Spectral Structure Comparison  [{label}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


