"""
E6: Random Network Spectral Comparison
========================================

将真实连接矩阵的谱结构与三类随机对照网络对比，
验证"低秩谱结构是真实连接组特有的"（而非随机网络偶然性质）。

三类随机对照
------------
1. **Erdős–Rényi (ER)**
   随机稠密矩阵 W_ij ~ N(0, 1/√N)，谱半径归一化到 ρ(W)_real。
   理论上：特征值服从 Wigner 半圆律，谱高度分散（PR ≈ N）。

2. **保度随机化（Degree-Preserving，DPR）**
   使用 Maslov-Sneppen Configuration Model 保留原矩阵的节点度分布。
   比 ER 更接近真实网络，但打乱了模块结构和功能梯度。

3. **权重混洗（Weight-Shuffled，WS）**
   保留拓扑（哪些节点相连），随机打乱连接权重。
   破坏权重的功能分级（局部强→远程弱），但保留稀疏模式。

比较指标
--------
对每类对照，重复 n_random 次（不同随机种子），报告均值 ± 标准差：
  - 谱有效维度（参与率 PR）
  - 主导特征值数（n_dominant）
  - 谱间隙比（|λ₁|/|λ₂|）
  - WC LLE（耦合强度 g=0.9，低于混沌边界）

**核心假设验证**：
若真实连接矩阵 PR << ER 的 PR（均值 - 2σ 以下），则说明真实网络
具有统计显著的低秩谱结构，支持假设 H1。

批判性注意事项
--------------
1. **ER 谱比较有理论上界**：Wigner 半圆律给出 PR_ER ≈ N（白色谱），
   真实 PR 与 N 之比是低秩程度的直观度量。
2. **度保持随机化的谱**：对于度均匀分布的网络，DPR ≈ ER；
   对度异质网络（枢纽节点），DPR 的谱可能也有集中（非 H1 的证据）。
   需要单独检验 "DPR 的 PR 是否也低"。
3. **LLE 比较需谨慎**：若所有对照在 g=0.9 下都是稳定的（LLE < 0），
   则 LLE 差异很小，不能作为区分指标。应将 g 提高到临界附近再比较。

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
    _rosenstein_lle_on_wc,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Random network generators
# ─────────────────────────────────────────────────────────────────────────────

def erdos_renyi(n: int, target_rho: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    生成 Erdős–Rényi 随机矩阵（高斯稠密），谱半径归一化到 target_rho。

    特征值理论分布：Wigner 半圆律，PR ≈ N（均匀谱，无主导模式）。

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
# Per-matrix metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    W: np.ndarray,
    n_traj_lle: int = 20,
    steps_lle: int = 200,
    g_lle: float = 0.9,
    seed: int = 42,
) -> Dict:
    """Compute spectral + WC LLE metrics for a single matrix."""
    spec = compute_spectral_metrics(W, symmetric=False)
    lle = _rosenstein_lle_on_wc(W, n_traj=n_traj_lle, steps=steps_lle, g=g_lle, seed=seed)
    return {
        "spectral_radius": spec["spectral_radius"],
        "participation_ratio": spec["participation_ratio"],
        "n_dominant": spec["n_dominant"],
        "spectral_gap_ratio": spec["spectral_gap_ratio"],
        "lle_wc": lle,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_random_spectral_comparison(
    W: np.ndarray,
    n_random: int = 10,
    n_traj_lle: int = 20,
    steps_lle: int = 200,
    g_lle: float = 0.9,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    seed: int = 42,
) -> Dict:
    """
    运行 E6：真实矩阵 vs 三类随机对照的谱结构比较。

    Args:
        W:          真实连接矩阵 (N, N)。
        n_random:   每类对照的随机实现数（用于估计均值 ± std）。
        n_traj_lle: LLE 估计用的轨迹数。
        steps_lle:  LLE 估计用的轨迹步数。
        g_lle:      WC 耦合强度（LLE 估计用）。
        output_dir: 结果保存目录。
        label:      矩阵标签。
        seed:       随机种子（各对照类型依次 +1）。

    Returns:
        summary dict（可序列化为 JSON）。
    """
    N = W.shape[0]
    rng_seeds = list(range(seed, seed + n_random))

    logger.info("E6 随机网络谱比较: N=%d, n_random=%d", N, n_random)

    # 1. Real network
    logger.info("  真实矩阵...")
    real_metrics = _compute_metrics(W, n_traj_lle=n_traj_lle, steps_lle=steps_lle,
                                    g_lle=g_lle, seed=seed)

    # 2. ER random
    target_rho = real_metrics["spectral_radius"]
    logger.info("  Erdős–Rényi (%d 实现)...", n_random)
    er_list = [
        _compute_metrics(erdos_renyi(N, target_rho=target_rho, seed=s),
                         n_traj_lle=n_traj_lle, steps_lle=steps_lle, g_lle=g_lle, seed=s)
        for s in rng_seeds
    ]

    # 3. Degree-preserving rewire
    logger.info("  保度随机重连 (%d 实现)...", n_random)
    dpr_list = [
        _compute_metrics(degree_preserving_rewire(W, seed=s),
                         n_traj_lle=n_traj_lle, steps_lle=steps_lle, g_lle=g_lle, seed=s)
        for s in rng_seeds
    ]

    # 4. Weight shuffle
    logger.info("  权重混洗 (%d 实现)...", n_random)
    ws_list = [
        _compute_metrics(weight_shuffle(W, seed=s),
                         n_traj_lle=n_traj_lle, steps_lle=steps_lle, g_lle=g_lle, seed=s)
        for s in rng_seeds
    ]

    def _agg(lst: List[Dict]) -> Dict:
        """Aggregate list of metric dicts → mean ± std."""
        keys = [k for k in lst[0] if isinstance(lst[0][k], (int, float))]
        out: Dict = {}
        for k in keys:
            vals = np.array([d[k] for d in lst if np.isfinite(d.get(k, float("nan")))])
            out[f"{k}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            out[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else float("nan")
        return out

    er_agg = _agg(er_list)
    dpr_agg = _agg(dpr_list)
    ws_agg = _agg(ws_list)

    # Z-scores: how many std away from ER mean is the real value?
    def _z(val: float, mean: float, std: float) -> float:
        if std < 1e-10:
            return float("nan")
        return (val - mean) / std

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
            ) < -2.0  # Real PR is > 2σ below ER mean → significantly lower rank
        ),
        "config": {
            "n_random": n_random,
            "g_lle": g_lle,
        },
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
    """绘制四组比较的小提琴/箱线图：PR、n_dominant、gap_ratio。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics_keys = ["participation_ratio", "n_dominant", "spectral_gap_ratio", "lle_wc"]
    metric_labels = ["谱有效维度 PR", "主导特征值数", "谱间隙比", "WC LLE"]

    fig, axes = plt.subplots(1, len(metrics_keys), figsize=(14, 5))
    group_names = ["Real", "ER", "DPR", "WS"]
    group_lists = [None, er_list, dpr_list, ws_list]
    group_colors = ["steelblue", "tomato", "darkorange", "forestgreen"]

    for ax, key, ml in zip(axes, metrics_keys, metric_labels):
        # Real: single point
        real_val = real_metrics.get(key, float("nan"))
        ax.axhline(real_val, ls="--", color="steelblue", lw=1.5, label="Real")

        positions: list = []
        data_groups: list = []
        colors_used: list = []

        for i, (name, lst, color) in enumerate(zip(group_names[1:], group_lists[1:], group_colors[1:]), start=1):
            vals = [d.get(key, float("nan")) for d in lst if np.isfinite(d.get(key, float("nan")))]
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
            # Mark real value
            ax.scatter([0], [real_val], color="steelblue", zorder=5, s=60)

        ax.set_ylabel(ml)
        ax.set_title(ml)

    fig.suptitle(f"随机网络谱结构对比  [{label}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)
