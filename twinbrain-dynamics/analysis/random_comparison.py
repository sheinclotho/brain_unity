"""
Random Model Comparison
========================

用随机动力学系统作为对照实验，验证当前模型的动力学结构 **不是随机模型偶然产生的**。

方法：
  构造随机线性-非线性动力系统：

    x(t+1) = clip(tanh(W @ x(t)), 0, 1)

  其中 W ~ N(0, σ)，归一化后谱半径 ≈ target_spectral_radius。

  对比指标（output: analysis_comparison.json）：

    n_attractors              — KMeans 检测到的吸引子数
    mean_lyapunov             — 平均 Lyapunov 指数
    trajectory_variance       — 轨迹总方差
    response_matrix_norm      — 响应矩阵 Frobenius 范数

输出：
  random_trajectories.npy     — 随机系统轨迹
  analysis_comparison.json    — 真实模型 vs 随机模型对比
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _make_random_dynamics_matrix(
    n_regions: int,
    target_spectral_radius: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """
    生成谱半径归一化的随机连接矩阵。

    W ~ N(0, 1/sqrt(n)), 之后缩放使谱半径 = target_spectral_radius。

    Args:
        n_regions:              脑区数量。
        target_spectral_radius: 目标谱半径（< 1 保证线性稳定性）。
        seed:                   随机种子。

    Returns:
        W: shape (n_regions, n_regions), float32。
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_regions, n_regions)).astype(np.float32)
    W /= np.sqrt(n_regions)

    # Compute largest eigenvalue magnitude
    eigvals = np.linalg.eigvals(W)
    sr = float(np.abs(eigvals).max())
    if sr > 1e-8:
        W = W * (target_spectral_radius / sr)
    return W


def run_random_trajectories(
    n_regions: int = 200,
    n_init: int = 200,
    steps: int = 1000,
    spectral_radius: float = 0.9,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    运行随机动力学系统，生成 n_init 条轨迹。

    动力学：x(t+1) = clip(tanh(W @ x(t)), 0, 1)

    Args:
        n_regions:        脑区数量。
        n_init:           随机初始状态数量。
        steps:            每条轨迹的步数。
        spectral_radius:  随机矩阵目标谱半径。
        seed:             随机种子。
        output_dir:       保存 random_trajectories.npy；None → 不保存。

    Returns:
        trajectories: shape (n_init, steps, n_regions), float32。
    """
    logger.info(
        "随机模型轨迹: n_init=%d, steps=%d, n_regions=%d, ρ=%.2f",
        n_init,
        steps,
        n_regions,
        spectral_radius,
    )
    W = _make_random_dynamics_matrix(
        n_regions, target_spectral_radius=spectral_radius, seed=seed
    )

    rng = np.random.default_rng(seed)
    X0 = rng.random((n_init, n_regions)).astype(np.float32)

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
    for i in range(n_init):
        x = X0[i].copy()
        for t in range(steps):
            x = np.clip(np.tanh(W @ x), 0.0, 1.0)
            trajectories[i, t] = x

    logger.info("  随机模型轨迹生成完成。")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "random_trajectories.npy", trajectories)
        logger.info("  → 已保存: %s/random_trajectories.npy", output_dir)

    return trajectories


def _compute_summary_stats(
    trajectories: np.ndarray,
    attractor_results: Optional[Dict] = None,
    lyapunov_results: Optional[Dict] = None,
    response_matrix: Optional[np.ndarray] = None,
) -> Dict:
    """计算一套可比较的汇总统计量。"""
    # Trajectory variance (mean per-region variance, averaged over trajectories)
    traj_var = float(np.var(trajectories, axis=1).mean())

    # Number of attractors (from attractor analysis if available)
    n_attractors = (
        int(attractor_results.get("kmeans_k", 0))
        if attractor_results is not None
        else None
    )

    # Mean Lyapunov exponent
    mean_lyapunov = (
        float(lyapunov_results["mean_lyapunov"])
        if lyapunov_results is not None
        else None
    )

    # Response matrix Frobenius norm
    rm_norm = (
        float(np.linalg.norm(response_matrix, "fro"))
        if response_matrix is not None
        else None
    )

    return {
        "n_attractors": n_attractors,
        "mean_lyapunov": mean_lyapunov,
        "trajectory_variance": traj_var,
        "response_matrix_norm": rm_norm,
    }


def run_random_model_comparison(
    trajectories: np.ndarray,
    attractor_results: Optional[Dict] = None,
    lyapunov_results: Optional[Dict] = None,
    response_matrix: Optional[np.ndarray] = None,
    random_n_init: int = 200,
    random_steps: int = 1000,
    spectral_radius: float = 0.9,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    运行随机模型对照实验，生成 analysis_comparison.json。

    Args:
        trajectories:     真实模型的轨迹 (n_init, steps, n_regions)。
        attractor_results: 真实模型吸引子分析结果（可选）。
        lyapunov_results:  真实模型 Lyapunov 分析结果（可选）。
        response_matrix:   真实模型响应矩阵 (可选)。
        random_n_init:    随机模型轨迹数量（默认 200）。
        random_steps:     随机模型步数（默认 1000）。
        spectral_radius:  随机矩阵谱半径（默认 0.9）。
        seed:             随机种子。
        output_dir:       保存输出文件；None → 不保存。

    Returns:
        comparison: {
            "model": Dict,   真实模型汇总统计
            "random": Dict,  随机模型汇总统计
        }
    """
    n_regions = trajectories.shape[2]
    logger.info("=" * 50)
    logger.info("随机模型对照实验")

    # --- Real model stats ---
    model_stats = _compute_summary_stats(
        trajectories,
        attractor_results=attractor_results,
        lyapunov_results=lyapunov_results,
        response_matrix=response_matrix,
    )
    logger.info("  真实模型统计: %s", model_stats)

    # --- Random model ---
    rand_trajs = run_random_trajectories(
        n_regions=n_regions,
        n_init=min(random_n_init, 50),   # cap for speed; 50 is sufficient for stats
        steps=min(random_steps, 200),
        spectral_radius=spectral_radius,
        seed=seed,
        output_dir=output_dir,
    )

    # Run attractor analysis on random trajectories
    rand_attractor_results = None
    try:
        from experiments.attractor_analysis import run_attractor_analysis
        rand_attractor_results = run_attractor_analysis(
            rand_trajs,
            tail_steps=min(10, rand_trajs.shape[1] // 4),
            k_candidates=[2, 3, 4],
            k_best=2,
        )
    except Exception as exc:
        logger.debug("  随机模型吸引子分析跳过: %s", exc)

    # Run stability analysis on random trajectories and use fraction_unstable
    # as a *proxy* for the Lyapunov sign.  This is a coarse heuristic:
    #   fraction_unstable − fraction_converged maps to (−1, +1) and qualitatively
    # indicates whether trajectories tend to diverge (positive) or converge
    # (negative).  It is NOT numerically comparable to a true Lyapunov exponent;
    # it is included only so the comparison JSON has a "lyapunov-like" field for
    # directional comparison.  The proxy is stored as `mean_lyapunov` in
    # rand_lyapunov_results and forwarded to `analysis_comparison.json` under
    # `random.mean_lyapunov`.  Run actual Wolf-method lyapunov estimation on the
    # random trajectories when a true numerical comparison is needed.
    rand_lyapunov_results = None
    try:
        from analysis.stability_analysis import run_stability_analysis
        rand_stab = run_stability_analysis(rand_trajs)
        proxy = rand_stab["fraction_unstable"] - rand_stab["fraction_converged"]
        rand_lyapunov_results = {"mean_lyapunov": proxy}
        logger.info(
            "  随机模型稳定性代理 λ=%.4f（非真实 Lyapunov 指数，仅用于对比方向）",
            proxy,
        )
    except Exception as exc:
        logger.debug("  随机模型稳定性分析跳过: %s", exc)

    rand_stats = _compute_summary_stats(
        rand_trajs,
        attractor_results=rand_attractor_results,
        lyapunov_results=rand_lyapunov_results,
    )
    logger.info("  随机模型统计: %s", rand_stats)

    comparison = {
        "model": model_stats,
        "random": rand_stats,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "analysis_comparison.json"

        def _json_safe(d: Dict) -> Dict:
            return {k: (v if v is not None else "N/A") for k, v in d.items()}

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: _json_safe(v) for k, v in comparison.items()},
                fh,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("  → 已保存: %s", out_path)

    return comparison
