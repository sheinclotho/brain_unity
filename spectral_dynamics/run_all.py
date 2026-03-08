"""
spectral_dynamics — Master Runner
====================================

一键运行全部六个实验（E1–E6），验证假设链：

  connectome structure
  → spectral concentration (H1)
  → low-dimensional dynamics (H2)
  → near-critical behavior (H3)

使用方法
--------
**基本用法（使用预计算的 twinbrain-dynamics 输出）**::

    python -m spectral_dynamics.run_all \\
        --trajectories outputs/trajectories.npy \\
        --response-matrix outputs/response_matrix.npy \\
        --output-dir outputs/spectral_dynamics

**从图缓存提取结构连接（附加选项）**::

    python -m spectral_dynamics.run_all \\
        --trajectories outputs/trajectories.npy \\
        --response-matrix outputs/response_matrix.npy \\
        --graph-cache outputs/graph_cache/sub-01_notask_xx.pt \\
        --output-dir outputs/spectral_dynamics

**快速模式（合成数据测试，无需任何预计算文件）**::

    python -m spectral_dynamics.run_all --synthetic --n-regions 50

**选择性运行实验**::

    python -m spectral_dynamics.run_all \\
        --trajectories outputs/trajectories.npy \\
        --response-matrix outputs/response_matrix.npy \\
        --experiments E1 E4 E5

输出目录结构
------------
  outputs/spectral_dynamics/
    spectral_summary_response_matrix.json
    spectral_summary_fc.json
    modal_energy_response_matrix.json
    perturbation_summary_response_matrix.json
    phase_diagram_response_matrix.json
    random_comparison_spectral_response_matrix.json
    *.png  (各实验图表)
    run_summary.json  (总结报告)

批判性注意事项（针对本流程）
----------------------------
1. 若响应矩阵 R 不可用，所有实验使用功能连接矩阵 FC（从轨迹计算），
   但 FC 与动力学 Jacobian 的对应关系较弱，分析结论需相应调整。
2. E5 相图仅对 WC 框架有效。若使用 FC（取值范围 [-1, 1]），
   WC 的 tanh(g*FC@x) 动力学可能产生振荡行为与真实 GNN 不同。
3. E4/E6 中的 LLE 估计需要足够长的轨迹（建议 steps >= 200）；
   合成/快速模式下 steps=100，结果精度较低。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("spectral_dynamics.run_all")

# ── Module imports ────────────────────────────────────────────────────────────
from .compute_connectivity import (
    load_response_matrix,
    load_trajectories,
    compute_fc_from_trajectories,
    participation_ratio,
)
# E1/B: Spectral analysis
from .e1_spectral_analysis import run_spectral_analysis
# E2/E3: Modal projection & energy
from .e2_e3_modal_projection import run_modal_projection
# E4: Structural perturbation
from .e4_structural_perturbation import run_structural_perturbation
# E5: Phase diagram
from .e5_phase_diagram import run_phase_diagram
# E6: Random network comparison
from .e6_random_comparison import run_random_spectral_comparison
# New modules (A, C, D, F)
from .a_connectivity_visualization import run_connectivity_visualization
from .c_community_structure import run_community_structure
from .d_hierarchical_structure import run_hierarchical_structure
from .f_pca_attractor import run_pca_attractor
# New analyses (H, I)
from .h_power_spectrum import run_power_spectrum
from .i_energy_constraint import run_energy_budget

_ALL_EXPERIMENTS = ["A", "B_E1", "C", "D", "E2E3", "E4", "E5", "E6", "F",
                    "H", "I"]
# Aliases so old E1/B keys still work
_EXPERIMENT_ALIASES: dict = {
    "E1":   "B_E1",
    "B":    "B_E1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data for testing / demo
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_data(n_regions: int = 50, n_traj: int = 20, steps: int = 100, seed: int = 42):
    """
    生成合成数据用于无模型测试。

    连接矩阵：低秩结构（前 5 个模态主导），谱半径 ≈ 0.95。
    轨迹：WC 动力学 x(t+1) = clip(tanh(0.9 * W @ x(t)), 0, 1)。
    响应矩阵：对称低秩近似（模拟因果效应矩阵）。
    """
    rng = np.random.default_rng(seed)
    N = n_regions

    # Low-rank base (5 dominant modes)
    k = 5
    U = rng.standard_normal((N, k)) / np.sqrt(k)
    sv = np.array([5.0, 3.5, 2.5, 1.8, 1.2])
    W_lr = U @ np.diag(sv) @ U.T
    # Add small noise
    W_noise = rng.standard_normal((N, N)) * 0.3
    W = W_lr + W_noise

    # Normalize spectral radius to 0.95
    eigvals = np.linalg.eigvals(W)
    rho = np.abs(eigvals).max()
    W = (W * 0.95 / rho).astype(np.float32)

    # Generate WC trajectories
    trajs = np.empty((n_traj, steps, N), dtype=np.float32)
    for i in range(n_traj):
        x = rng.random(N).astype(np.float32)
        for t in range(steps):
            trajs[i, t] = x
            x = np.clip(np.tanh(0.9 * (W @ x)), 0.0, 1.0).astype(np.float32)

    # Response matrix: slightly asymmetric variant of W (simulate Jacobian)
    R = (W + 0.1 * rng.standard_normal((N, N)).astype(np.float32)).astype(np.float32)

    return W, trajs, R


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(
    trajectories: Optional[np.ndarray] = None,
    response_matrix: Optional[np.ndarray] = None,
    output_dir: Path = Path("outputs/spectral_dynamics"),
    experiments: Optional[List[str]] = None,
    # Community detection
    k_range: Optional[List[int]] = None,
    # E4/E6 performance params
    n_random: int = 10,
    g_min: float = 0.1,
    g_max: float = 3.0,
    g_step: float = 0.2,
    # PCA
    burnin: int = 0,
    seed: int = 42,
    # Legacy params (ignored — no longer used)
    n_traj_lle: int = 0,
    steps_lle: int = 0,
    g_lle: float = 0.9,
    n_traj_phase: int = 0,
    steps_phase: int = 0,
) -> dict:
    """
    运行全部（或指定的）谱动力学实验。

    所有实验仅对 GNN 生成的轨迹和响应矩阵做分析，不运行任何独立的动力学模拟。

    Args:
        trajectories:   shape (n_init, T, N)，GNN 自由动力学轨迹；
                        None → 仅用 FC 分析。
        response_matrix: shape (N, N)，GNN 有效连接/响应矩阵；
                         None → 仅用 FC 作为连接矩阵。
        output_dir:     所有输出文件的保存目录。
        experiments:    要运行的实验列表（如 ["E1", "E4"]）；
                        None → 全部运行。
        n_random:       E6 中随机对照的实现数。
        g_min/max/step: E5 相图扫描范围（谱半径解析扫描，无仿真）。
        seed:           全局随机种子。

    Returns:
        run_summary dict，含各实验结果摘要。
    """
    # Warn if caller passes legacy WC params (they no longer have any effect)
    if n_traj_lle or steps_lle or n_traj_phase or steps_phase:
        import warnings
        warnings.warn(
            "n_traj_lle, steps_lle, n_traj_phase, steps_phase are ignored — "
            "WC simulation has been removed. spectral_dynamics now operates "
            "purely on GNN-generated trajectories and response matrices.",
            DeprecationWarning, stacklevel=2,
        )
    if experiments is None:
        experiments = _ALL_EXPERIMENTS
    # Normalize aliases (E1 → B_E1, B → B_E1)
    experiments = [_EXPERIMENT_ALIASES.get(e.upper(), e.upper()) for e in experiments]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary: dict = {
        "experiments_run": experiments,
        "output_dir": str(output_dir),
        "hypotheses": {},
        "results": {},
    }

    # ── Determine connectivity matrix ─────────────────────────────────────────
    if response_matrix is not None:
        W_main = response_matrix
        W_label = "response_matrix"
        logger.info("使用响应矩阵作为主要连接矩阵: shape=%s", W_main.shape)
    elif trajectories is not None:
        logger.warning(
            "响应矩阵不可用，改用功能连接矩阵（FC）。"
            "注意：FC 与动力学 Jacobian 的对应关系较弱。"
        )
        W_main = compute_fc_from_trajectories(trajectories)
        W_label = "fc"
    else:
        raise ValueError(
            "至少需要提供 trajectories 或 response_matrix 之一。"
        )

    # Always compute FC (for E6 comparison and E2/E3 reference)
    W_fc = (
        compute_fc_from_trajectories(trajectories)
        if trajectories is not None
        else None
    )

    # ── A: Connectivity Visualization ─────────────────────────────────────────
    _community_labels_for_A: Optional[np.ndarray] = None  # filled by C

    if "A" in experiments:
        t0 = time.time()
        logger.info("═══ A: 连接矩阵可视化 ═══")
        # A1: raw heatmap (community_labels not yet available here)
        rA = run_connectivity_visualization(W_main, community_labels=None,
                                            output_dir=output_dir, label=W_label)
        run_summary["results"]["A"] = rA
        logger.info("A (原始热图) 完成 (%.1fs)", time.time() - t0)

    # ── B/E1: Spectral Analysis ───────────────────────────────────────────────
    if "B_E1" in experiments:
        t0 = time.time()
        logger.info("═══ B/E1: 谱结构分析 ═══")
        r1_main = run_spectral_analysis(W_main, output_dir=output_dir,
                                        label=W_label, symmetric=False)
        run_summary["results"]["E1_main"] = r1_main

        if W_fc is not None and W_label != "fc":
            r1_fc = run_spectral_analysis(W_fc, output_dir=output_dir,
                                          label="fc", symmetric=True)
            run_summary["results"]["E1_fc"] = r1_fc

        run_summary["hypotheses"]["H1_spectral_concentration"] = {
            "participation_ratio": r1_main["participation_ratio"],
            "n_regions": r1_main["n_regions"],
            "pr_fraction": round(r1_main["participation_ratio"] / r1_main["n_regions"], 3),
            "n_dominant": r1_main["n_dominant"],
            "supported_pr_below_20pct": bool(
                r1_main["participation_ratio"] < 0.20 * r1_main["n_regions"]
            ),
        }
        logger.info("B/E1 完成 (%.1fs)", time.time() - t0)

    # ── C: Community Structure ────────────────────────────────────────────────
    if "C" in experiments:
        t0 = time.time()
        logger.info("═══ C: 社区结构分析 ═══")
        rC = run_community_structure(
            W_main, k_range=k_range, output_dir=output_dir, label=W_label, seed=seed,
        )
        run_summary["results"]["C"] = {
            "method": rC["method"],
            "n_communities": rC["n_communities"],
            "modularity_q": rC["modularity_q"],
            "q_interpretation": rC["q_interpretation"],
            "community_sizes": rC["community_sizes"],
        }
        _community_labels_for_A = np.asarray(rC["community_labels"], dtype=np.int32)

        # Now generate A2 (community-reordered heatmap) since C results are available
        if "A" in experiments and _community_labels_for_A is not None:
            from .a_connectivity_visualization import plot_connectivity_reordered
            plot_connectivity_reordered(
                W_main, _community_labels_for_A, Path(output_dir), label=W_label,
            )
            run_summary["results"]["A"]["community_reordered_plot"] = "generated"

        logger.info("C 完成 (%.1fs)", time.time() - t0)

    # ── D: Hierarchical Structure ─────────────────────────────────────────────
    if "D" in experiments:
        t0 = time.time()
        logger.info("═══ D: 层级结构检测 ═══")
        rD = run_hierarchical_structure(
            W_main, method="ward", output_dir=output_dir, label=W_label,
        )
        run_summary["results"]["D"] = {
            "is_hierarchical": rD["is_hierarchical"],
            "hierarchy_index": rD["hierarchy_index"],
            "n_clusters_at_20pct": rD["n_clusters_at_20pct_height"],
            "n_clusters_at_80pct": rD["n_clusters_at_80pct_height"],
        }
        logger.info("D 完成 (%.1fs)", time.time() - t0)

    # ── E2+E3: Modal Projection & Energy ─────────────────────────────────────
    if "E2E3" in experiments and trajectories is not None:
        t0 = time.time()
        logger.info("═══ E2+E3: 模态投影与能量分布 ═══")
        sym = (W_label == "fc")
        r23 = run_modal_projection(
            trajectories, W_main, output_dir=output_dir,
            label=W_label, symmetric=sym,
        )
        run_summary["results"]["E2E3"] = r23
        run_summary["hypotheses"]["H2_low_dimensional_dynamics"] = {
            "n_modes_80pct": r23["n_modes_80pct"],
            "energy_top2_pct": r23["energy_top2_pct"],
            "energy_top5_pct": r23["energy_top5_pct"],
            "supported": r23["h2_supported"],
        }
        logger.info("E2+E3 完成 (%.1fs)", time.time() - t0)
    elif "E2E3" in experiments:
        logger.warning("E2+E3 跳过：未提供轨迹数据。")

    # ── E4: Structural Perturbation ───────────────────────────────────────────
    if "E4" in experiments:
        t0 = time.time()
        logger.info("═══ E4: 结构扰动实验 ═══")
        r4 = run_structural_perturbation(
            W_main, output_dir=output_dir, label=W_label, seed=seed,
        )
        run_summary["results"]["E4"] = {
            "original_pr": r4["original"]["participation_ratio"],
            "shuffle_delta_pr": r4["weight_shuffle"].get("delta_pr"),
            "rewire_delta_pr": r4["degree_preserving_rewire"].get("delta_pr"),
        }
        logger.info("E4 完成 (%.1fs)", time.time() - t0)

    # ── E5: Phase Diagram ─────────────────────────────────────────────────────
    if "E5" in experiments:
        t0 = time.time()
        logger.info("═══ E5: 耦合强度相图 ═══")
        # Compute LLE reference from GNN trajectories produced by earlier pipeline
        # steps — E5 itself is pure spectral analysis; it receives a precomputed float.
        lle_reference = None
        if trajectories is not None:
            try:
                from analysis.lyapunov import rosenstein_lyapunov
                lles = [
                    rosenstein_lyapunov(trajectories[i])[0]
                    for i in range(min(10, len(trajectories)))
                ]
                valid = [v for v in lles if np.isfinite(v)]
                if valid:
                    lle_reference = float(np.median(valid))
                    logger.info("E5: 流水线 LLE 参考值 (g=1) = %.4f", lle_reference)
            except Exception as exc:
                logger.debug("E5: LLE 参考值计算失败 (%s)，跳过标注。", exc)
        r5 = run_phase_diagram(
            W_main, g_min=g_min, g_max=g_max, g_step=g_step,
            lle_reference=lle_reference,
            output_dir=output_dir, label=W_label,
        )
        run_summary["results"]["E5"] = {
            "actual_rho_W": r5["actual_rho_W"],
            "g_linear_critical": r5["g_linear_critical"],
            "h3_supported": r5["h3_supported"],
        }
        run_summary["hypotheses"]["H3_near_critical"] = {
            "spectral_radius": r5["actual_rho_W"],
            "g_linear_critical": r5["g_linear_critical"],
            "h3_supported": r5["h3_supported"],
        }
        logger.info("E5 完成 (%.1fs)", time.time() - t0)

    # ── E6: Random Network Comparison ─────────────────────────────────────────
    if "E6" in experiments:
        t0 = time.time()
        logger.info("═══ E6: 随机网络谱比较 ═══")
        r6 = run_random_spectral_comparison(
            W_main, n_random=n_random,
            output_dir=output_dir, label=W_label, seed=seed,
        )
        run_summary["results"]["E6"] = {
            "real_pr": r6["real"]["participation_ratio"],
            "er_pr_mean": r6["er_random"].get("participation_ratio_mean"),
            "pr_z_score": r6["z_scores_vs_er"]["pr_z"],
            "h1_supported": r6["h1_supported"],
        }
        if "H1_spectral_concentration" not in run_summary["hypotheses"]:
            run_summary["hypotheses"]["H1_spectral_concentration"] = {}
        run_summary["hypotheses"]["H1_spectral_concentration"]["z_vs_er"] = (
            r6["z_scores_vs_er"]["pr_z"]
        )
        logger.info("E6 完成 (%.1fs)", time.time() - t0)

    # ── F: PCA + Attractor Projection ────────────────────────────────────────
    if "F" in experiments and trajectories is not None:
        t0 = time.time()
        logger.info("═══ F: PCA 维度估计 + 吸引子可视化 ═══")
        rF = run_pca_attractor(
            trajectories, output_dir=output_dir, label=W_label,
            burnin=burnin, seed=seed,
        )
        run_summary["results"]["F"] = rF
        # Also contribute to H2 evidence
        if "H2_low_dimensional_dynamics" not in run_summary["hypotheses"]:
            run_summary["hypotheses"]["H2_low_dimensional_dynamics"] = {}
        run_summary["hypotheses"]["H2_low_dimensional_dynamics"].update({
            "pca_n_components_90pct": rF["n_components_90pct"],
            "pca_efficiency_ratio": rF["pca_efficiency_ratio"],
            "variance_top2_pct": rF["variance_top2_pct"],
        })
        logger.info("F 完成 (%.1fs)", time.time() - t0)
    elif "F" in experiments:
        logger.warning("F 跳过：未提供轨迹数据。")

    # ── H: Power spectrum + spatial oscillation modes ─────────────────────────
    if "H" in experiments and trajectories is not None:
        t0 = time.time()
        logger.info("═══ H: 功率谱 + 空间振荡模态 ═══")
        try:
            rH = run_power_spectrum(
                trajectories, dt=1.0, burnin=burnin,
                output_dir=output_dir, label=W_label,
            )
            band_info = rH.get("band_analysis", {})
            run_summary["results"]["H"] = {
                "dominant_freq_hz": band_info.get("dominant_freq_hz"),
                "dominant_freq_band": band_info.get("dominant_freq_band"),
                "band_power_fractions": band_info.get("band_power_fractions"),
                "nyquist_hz": band_info.get("nyquist_hz"),
                "peak_region_per_band": rH.get("spatial_modes", {}).get(
                    "peak_region_per_band", {}
                ),
            }
            run_summary["hypotheses"].setdefault(
                "H4_neural_oscillations", {}
            ).update({
                "dominant_freq_hz": band_info.get("dominant_freq_hz"),
                "dominant_band": band_info.get("dominant_freq_band"),
                "band_fractions": band_info.get("band_power_fractions", {}),
            })
            logger.info(
                "H 完成 (%.1fs): dominant_freq=%.4f Hz [%s]",
                time.time() - t0,
                band_info.get("dominant_freq_hz", 0),
                band_info.get("dominant_freq_band", "?"),
            )
        except Exception as exc:
            logger.warning("H 失败: %s", exc)
    elif "H" in experiments:
        logger.warning("H 跳过：未提供轨迹数据。")

    # ── I: Energy constraint experiment ───────────────────────────────────────
    if "I" in experiments and trajectories is not None:
        t0 = time.time()
        logger.info("═══ I: 能量约束预算分析 ═══")
        try:
            rI = run_energy_budget(
                trajectories,
                output_dir=output_dir, label=W_label,
            )
            run_summary["results"]["I"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in rI.items()
            }
            run_summary["hypotheses"].setdefault(
                "H5_energy_constraint_criticality", {}
            ).update({
                "E_mean": rI.get("E_mean"),
                "recommended_budgets": rI.get("recommended_budgets"),
            })
            logger.info(
                "I 完成 (%.1fs): E*=%.4f",
                time.time() - t0,
                rI.get("E_mean") or float("nan"),
            )
        except Exception as exc:
            logger.warning("I 失败: %s", exc)
    elif "I" in experiments:
        logger.warning("I 跳过：未提供连接矩阵 W_main。")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
    logger.info("总结报告保存至: %s", summary_path)

    # Print hypothesis evaluation
    _print_hypothesis_summary(run_summary["hypotheses"])

    return run_summary


def _print_hypothesis_summary(hypotheses: dict) -> None:
    """在日志中打印假设验证摘要。"""
    logger.info("=" * 60)
    logger.info("假设验证摘要")
    logger.info("=" * 60)

    h1 = hypotheses.get("H1_spectral_concentration", {})
    if h1:
        pr = h1.get("participation_ratio", "?")
        N = h1.get("n_regions", "?")
        z = h1.get("z_vs_er", "?")
        sup = h1.get("supported_pr_below_20pct", "?")
        logger.info(
            "H1 谱集中度: PR=%.1f/N=%s (%.0f%%), vs ER z=%.2f → %s",
            pr if isinstance(pr, float) else 0,
            N, (pr / N * 100) if isinstance(pr, (int, float)) and isinstance(N, (int, float)) else 0,
            z if isinstance(z, float) else 0,
            "✓ 支持" if sup else "✗ 不支持" if sup is False else "?",
        )

    h2 = hypotheses.get("H2_low_dimensional_dynamics", {})
    if h2:
        logger.info(
            "H2 低维动力学: 80%%能量需 %d 个模态, 前2模态=%.1f%%, → %s",
            h2.get("n_modes_80pct", "?"),
            h2.get("energy_top2_pct", 0),
            "✓ 支持" if h2.get("supported") else "✗ 不支持",
        )

    h3 = hypotheses.get("H3_near_critical", {})
    if h3:
        logger.info(
            "H3 近临界: ρ(W)=%.3f, 线性临界g=%.2f, LLE≈0时g=%.2f → %s",
            h3.get("spectral_radius", 0),
            h3.get("g_linear_critical", 0),
            h3.get("critical_g_where_lle0", 0),
            "✓ 支持" if h3.get("h3_supported") else "✗ 不支持",
        )

    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="spectral_dynamics: 验证连接组谱结构→低维动力学→近临界假设链",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("数据输入")
    data.add_argument("--trajectories", type=Path, default=None,
                      help="twinbrain-dynamics 输出的轨迹文件 (trajectories.npy)")
    data.add_argument("--response-matrix", type=Path, default=None,
                      help="twinbrain-dynamics 输出的响应矩阵 (response_matrix.npy)")
    data.add_argument("--graph-cache", type=Path, default=None,
                      help="图缓存 .pt 文件（用于提取结构连接矩阵）")
    data.add_argument("--synthetic", action="store_true",
                      help="使用合成数据（测试模式，无需任何输入文件）")
    data.add_argument("--n-regions", type=int, default=50,
                      help="合成模式下的脑区数")

    out = p.add_argument_group("输出")
    out.add_argument("--output-dir", type=Path,
                     default=Path("outputs/spectral_dynamics"),
                     help="结果保存目录")

    exp = p.add_argument_group("实验选择")
    exp.add_argument("--experiments", nargs="*", default=None,
                     choices=_ALL_EXPERIMENTS + [e.lower() for e in _ALL_EXPERIMENTS]
                             + ["E1", "B", "e1", "b", "LYA", "lya"],
                     help="指定要运行的实验子集；不指定则全部运行")

    perf = p.add_argument_group("性能参数")
    perf.add_argument("--burnin", type=int, default=0,
                      help="F 模块 PCA 中跳过的前 N 步（去除瞬态）")
    perf.add_argument("--k-range", nargs="*", type=int, default=None,
                      help="C 模块社区检测候选 k 列表，如 3 4 5 6 7 8")
    perf.add_argument("--n-random", type=int, default=10,
                      help="E6 随机对照实现数")
    perf.add_argument("--g-min", type=float, default=0.1)
    perf.add_argument("--g-max", type=float, default=3.0)
    perf.add_argument("--g-step", type=float, default=0.2)
    perf.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.synthetic:
        logger.info("使用合成数据（N=%d）进行测试...", args.n_regions)
        W, trajs, R = _make_synthetic_data(n_regions=args.n_regions)
        trajectories = trajs
        response_matrix = R
    else:
        # Load from files
        trajectories: Optional[np.ndarray] = None
        response_matrix: Optional[np.ndarray] = None

        if args.trajectories is not None:
            # Accept either a .npy file path or the containing directory
            traj_dir = (args.trajectories.parent
                        if args.trajectories.name == "trajectories.npy"
                        else args.trajectories)
            trajectories = load_trajectories(traj_dir)
        if args.response_matrix is not None:
            rm_dir = (args.response_matrix.parent
                      if args.response_matrix.name == "response_matrix.npy"
                      else args.response_matrix)
            response_matrix = load_response_matrix(rm_dir)

        if trajectories is None and response_matrix is None:
            logger.error(
                "请提供 --trajectories 和/或 --response-matrix，"
                "或使用 --synthetic 运行测试模式。"
            )
            sys.exit(1)

    run_all(
        trajectories=trajectories,
        response_matrix=response_matrix,
        output_dir=args.output_dir,
        experiments=args.experiments,
        k_range=args.k_range,
        n_random=args.n_random,
        g_min=args.g_min,
        g_max=args.g_max,
        g_step=args.g_step,
        burnin=args.burnin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
