"""
Random Model Comparison
========================

用随机动力学系统作为对照实验，验证当前模型的动力学结构 **不是随机模型偶然产生的**。

方法：
  构造随机线性-非线性动力系统：

    x(t+1) = clip(tanh(W @ x(t)), 0, 1)

  其中 W ~ N(0, σ)，归一化后谱半径 ≈ target_spectral_radius。

  关键设计原则：
  - 多随机种子（默认 5 个）：每次运行使用不同的随机矩阵 W，
    报告均值 ± 标准差。一个固定种子的结果对不同数据/模型毫无说服力。
  - 多谱半径：默认 ρ ∈ {0.9, 1.5, 2.0}，分别对应稳定/近临界/混沌。
    ρ < 1 数学上保证稳定（tanh 压缩 + Banach 不动点定理）。
    ρ > 1 不保证混沌：tanh 非线性压缩使实际混沌边界高于线性理论 ρ=1，
    对 n≈190 的随机网络实测约为 ρ≈1.5。
  - 真实 Wolf LLE：对每个随机模型运行 Wolf-Benettin 方法计算真实 LLE，
    而非用稳定性代理指标近似。

  对比指标（output: analysis_comparison.json）：

    n_attractors              — KMeans 检测到的吸引子数
    mean_lyapunov_mean/std    — 多种子平均 LLE 及标准差（真实 Wolf LLE）
    trajectory_variance       — 轨迹总方差
    response_matrix_norm      — 响应矩阵 Frobenius 范数

输出：
  random_trajectories.npy     — 随机系统轨迹（最后一个谱半径）
  analysis_comparison.json    — 真实模型 vs 随机模型对比
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared trajectory-metric helpers (imported by node_ablation,
# input_dimension_control, and this module)
# ─────────────────────────────────────────────────────────────────────────────

def avg_rosenstein_lle(
    trajs: np.ndarray,
    max_lag: int = 50,
    min_temporal_sep: int = 20,
) -> float:
    """Estimate LLE by averaging Rosenstein estimates across all trajectories.

    Calls ``rosenstein_lyapunov`` directly (no simulator required) on each
    trajectory (shape T × N) and returns the mean of finite values.

    Args:
        trajs:             shape (n_traj, T, N).
        max_lag:           Rosenstein maximum tracking lag.
        min_temporal_sep:  Minimum temporal separation for nearest-neighbour.

    Returns:
        Mean LLE (float); NaN when no finite estimate could be obtained.
    """
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        vals = []
        for traj in trajs:   # traj: (T, N)
            lle, _ = rosenstein_lyapunov(
                traj.astype(np.float64),
                max_lag=max_lag,
                min_temporal_sep=min_temporal_sep,
            )
            if np.isfinite(lle):
                vals.append(lle)
        if not vals:
            logger.warning(
                "avg_rosenstein_lle: all %d trajectories returned NaN. "
                "Check that T >> max_lag (%d) and N is reasonable.",
                len(trajs), max_lag,
            )
            return float("nan")
        return float(np.mean(vals))
    except Exception as exc:
        logger.debug("avg_rosenstein_lle failed: %s", exc)
        return float("nan")


def _degree_preserving_rewire(
    W: np.ndarray,
    n_swaps: int,
    seed: int = 0,
) -> np.ndarray:
    """Maslov-Sneppen degree-preserving rewire with an absolute swap count.

    Identical logic to ``spectral_dynamics.e4_structural_perturbation
    .degree_preserving_rewire`` but accepts *n_swaps* as an absolute integer
    (rather than a factor × nnz), so callers can control the randomisation
    budget directly.

    Args:
        W:       Square connectivity matrix (N, N).
        n_swaps: Total number of attempted edge swaps.
        seed:    Random seed.

    Returns:
        W_rewired: Same shape as W, edges randomly rewired.
    """
    rng = np.random.default_rng(seed)
    W_out = W.copy().astype(np.float32)
    rows, cols = np.where(W_out != 0)
    n_edges = len(rows)
    if n_edges < 4:
        logger.warning("Too few edges (%d) for degree-preserving rewire.", n_edges)
        return W_out

    for _ in range(n_swaps):
        # Use replace=False to guarantee two distinct indices in one call
        idx = rng.choice(n_edges, size=2, replace=False)
        i, j = int(rows[idx[0]]), int(cols[idx[0]])
        k, l = int(rows[idx[1]]), int(cols[idx[1]])
        if i == l or k == j:
            continue
        if W_out[i, l] != 0 or W_out[k, j] != 0:
            continue
        wij = W_out[i, j]
        wkl = W_out[k, l]
        W_out[i, j] = 0.0
        W_out[k, l] = 0.0
        W_out[i, l] = wij
        W_out[k, j] = wkl
        rows[idx[0]], cols[idx[0]] = i, l
        rows[idx[1]], cols[idx[1]] = k, j

    return W_out


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
        target_spectral_radius: 目标谱半径（< 1 稳定，= 1 临界，> 1 混沌）。
        seed:                   随机种子。不同种子生成不同 W 矩阵！
                                固定种子只能说明单个随机矩阵的结果，
                                换数据仍得到相同结论没有说服力。
                                调用者应循环不同种子取均值 ± 标准差。

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


def _wolf_lle_random(
    W: np.ndarray,
    x0: np.ndarray,
    steps: int = 500,
    renorm_steps: int = 20,
    eps: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    用 Wolf-Benettin 法计算随机模型 x(t+1)=clip(tanh(W@x),0,1) 的最大 Lyapunov 指数。

    这是对随机模型的 **真实** LLE 计算，而非基于稳定性分类的代理指标。
    使用实际扰动演化，结果可以直接与 TwinBrain 模型的 Wolf LLE 数值比较。

    Args:
        W:            连接矩阵 (n, n)。
        x0:           初始状态 (n,)。
        steps:        总步数（越多越准确）。
        renorm_steps: 每次重归一化之前的步数。
        eps:          初始扰动幅度（归一化后的实际扰动）。
        rng:          随机数生成器（用于初始扰动方向）。

    Returns:
        LLE: 每步的最大 Lyapunov 指数（负 = 稳定，正 = 混沌）。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x0)

    # Floor for collapsed perturbations: log(1e-30 / eps) per period.
    # This represents the minimum measurable separation (machine precision floor).
    # Recording floor values rather than skipping gives the correct large-negative
    # LLE for strongly contracting systems (e.g. tanh saturation with FC matrices).
    _FLOOR_LOG_GROWTH = np.log(1e-30 / eps)  # ≈ -55.3 for eps=1e-6

    def _step(x: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(W @ x), 0.0, 1.0)

    # Warm up base trajectory to avoid transient
    x_base = x0.copy()
    for _ in range(renorm_steps * 3):
        x_base = _step(x_base)

    # Random unit-norm perturbation direction
    direction = rng.standard_normal(n).astype(np.float64)
    direction /= np.linalg.norm(direction) + 1e-300
    x_pert = x_base + eps * direction

    log_growths: List[float] = []
    n_periods = max(1, steps // renorm_steps)

    for _ in range(n_periods):
        for _ in range(renorm_steps):
            x_base = _step(x_base)
            x_pert = _step(x_pert)

        delta = x_pert - x_base
        r = float(np.linalg.norm(delta))
        if r > 0.0:
            log_growths.append(np.log(r / eps))
            # Renormalize: keep separation = eps
            x_pert = x_base + eps * delta / r
        else:
            # Perturbation collapsed to exactly zero (machine precision).
            # Record the floor log-growth (genuinely large-negative for a
            # strongly contracting system) and re-perturb to continue.
            log_growths.append(_FLOOR_LOG_GROWTH)
            direction = rng.standard_normal(n).astype(np.float64)
            direction /= np.linalg.norm(direction) + 1e-300
            x_pert = x_base + eps * direction

    if not log_growths:
        return float(_FLOOR_LOG_GROWTH / renorm_steps)
    return float(np.mean(log_growths)) / renorm_steps


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
        seed:             随机种子（用于 W 矩阵和初始状态）。
        output_dir:       保存 random_trajectories.npy；None → 不保存。

    Returns:
        trajectories: shape (n_init, steps, n_regions), float32。
    """
    logger.info(
        "随机模型轨迹: n_init=%d, steps=%d, n_regions=%d, ρ=%.2f, seed=%d",
        n_init,
        steps,
        n_regions,
        spectral_radius,
        seed,
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

    # Mean Lyapunov exponent — prefer Rosenstein over Wolf when bias is detected.
    #
    # When Wolf bias is detected (wolf_bias_warning=True), the Wolf mean LLE is
    # systematically wrong for twin mode (context-dilution effect: perturbation
    # injected only at the last context step is diluted by the L-1 identical
    # history steps, making all trajectories appear to converge at the same rate
    # regardless of their starting state).  The Rosenstein method works directly
    # on the trajectory data and is unaffected by this bias.
    #
    # Using the biased Wolf value in the comparison note would produce an
    # incorrect conclusion — e.g., the log showed:
    #   Wolf mean  = -0.05589  (biased, all 200 trajectories gave same value)
    #   Rosenstein = +0.01228  (unbiased, weakly chaotic)
    # Using Wolf gave: "real model is between stable and chaotic"
    # Using Rosenstein gives the correct: "real model is weakly chaotic"
    mean_lyapunov = None
    mean_lyapunov_source = None
    if lyapunov_results is not None:
        bias_detected = lyapunov_results.get("wolf_bias_warning", False)
        rosen = lyapunov_results.get("mean_rosenstein")
        if bias_detected and rosen is not None and np.isfinite(float(rosen)):
            mean_lyapunov = float(rosen)
            mean_lyapunov_source = "rosenstein"
        else:
            mean_lyapunov = float(lyapunov_results["mean_lyapunov"])
            mean_lyapunov_source = "wolf"
        if bias_detected:
            wolf_val = float(lyapunov_results["mean_lyapunov"])
            logger.info(
                "  随机对照: Wolf 偏差已检测。使用 Rosenstein LLE=%.5f "
                "（Wolf 偏差值 %.5f 已弃用）作为真实模型 LLE。",
                mean_lyapunov, wolf_val,
            )

    # Response matrix Frobenius norm
    rm_norm = (
        float(np.linalg.norm(response_matrix, "fro"))
        if response_matrix is not None
        else None
    )

    stats = {
        "n_attractors": n_attractors,
        "mean_lyapunov": mean_lyapunov,
        "trajectory_variance": traj_var,
        "response_matrix_norm": rm_norm,
    }
    if mean_lyapunov_source is not None:
        stats["mean_lyapunov_source"] = mean_lyapunov_source
    return stats


def _random_lle_multi_seed(
    n_regions: int,
    spectral_radius: float,
    n_seeds: int,
    lle_steps: int = 400,
    renorm_steps: int = 20,
    seed_base: int = 100,
) -> Dict[str, float]:
    """
    对同一谱半径运行 n_seeds 个不同随机矩阵，计算均值 ± 标准差。

    **为什么需要多种子**：
    单一固定种子只能说明「这一个随机矩阵」的 LLE，无法说明「谱半径为 ρ
    的随机矩阵族」的典型行为。换数据后得到完全相同的随机基线结果只是因为
    W 矩阵没有变化，而非因为结果本身稳健。多种子均值 ± 标准差才具有
    统计说服力。

    Args:
        n_regions:        脑区数量。
        spectral_radius:  目标谱半径。
        n_seeds:          随机种子数量（独立 W 矩阵的数量）。
        lle_steps:        每次 Wolf 方法的总步数。
        renorm_steps:     每次重归一化步数。
        seed_base:        种子起点（使用 seed_base, seed_base+1, …）。

    Returns:
        {mean_lyapunov, std_lyapunov, min_lyapunov, max_lyapunov}
    """
    lles = []
    rng_pert = np.random.default_rng(0)  # fixed for perturbation direction only
    for k in range(n_seeds):
        seed = seed_base + k
        W = _make_random_dynamics_matrix(n_regions, spectral_radius, seed=seed)
        rng_init = np.random.default_rng(seed)
        x0 = rng_init.random(n_regions).astype(np.float32)
        lle = _wolf_lle_random(
            W, x0, steps=lle_steps, renorm_steps=renorm_steps, rng=rng_pert
        )
        lles.append(lle)
        logger.debug(
            "  random ρ=%.2f seed=%d: LLE=%.5f", spectral_radius, seed, lle
        )

    arr = np.array(lles, dtype=np.float64)
    return {
        "mean_lyapunov": float(arr.mean()),
        "std_lyapunov":  float(arr.std()),
        "min_lyapunov":  float(arr.min()),
        "max_lyapunov":  float(arr.max()),
        "n_seeds":       n_seeds,
        "spectral_radius": spectral_radius,
        "stability": "chaotic" if arr.mean() > 0.01 else (
            "critical" if arr.mean() > -0.01 else "stable"
        ),
    }


def run_random_model_comparison(
    trajectories: np.ndarray,
    attractor_results: Optional[Dict] = None,
    lyapunov_results: Optional[Dict] = None,
    response_matrix: Optional[np.ndarray] = None,
    random_n_init: int = 200,
    random_steps: int = 1000,
    spectral_radius: float = 0.9,
    spectral_radii: Optional[List[float]] = None,
    n_seeds: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    运行随机模型对照实验，生成 analysis_comparison.json。

    **关键改进**：
    1. 多随机种子（n_seeds，默认 5）：每个谱半径下使用 n_seeds 个不同的随机
       矩阵 W，报告 LLE 均值 ± 标准差。标准差量化了结论的鲁棒性，而非依赖
       一个固定种子的单一结果（换数据后 W 不变，结果当然完全相同）。
    2. 多谱半径：默认比较 ρ ∈ {0.9, 1.5, 2.0}，分别对应稳定/近临界/混沌。
       实测混沌边界（tanh 非线性网络，n≈190）约为 ρ≈1.5，而非线性理论的 ρ=1。
       ρ < 1 数学保证稳定；ρ > 1 不保证混沌（tanh 压缩效应）。
    3. 真实 Wolf LLE：用 _wolf_lle_random() 计算每个随机矩阵的真实 LLE，
       而非依赖稳定性分类的代理指标（后者不可与模型 LLE 数值比较）。

    Args:
        trajectories:      真实模型的轨迹 (n_init, steps, n_regions)。
        attractor_results: 真实模型吸引子分析结果（可选）。
        lyapunov_results:  真实模型 Lyapunov 分析结果（可选）。
        response_matrix:   真实模型响应矩阵（可选）。
        random_n_init:     随机模型轨迹数量。
        random_steps:      随机模型步数。
        spectral_radius:   旧式单浮点参数（向后兼容；优先使用 spectral_radii）。
        spectral_radii:    谱半径列表（None → [spectral_radius, 1.5, 2.0]）。
        n_seeds:           每个谱半径下使用的随机种子数（默认 5）。
        seed:              基础随机种子（multi-seed 从 seed*10+100 开始）。
        output_dir:        保存输出文件；None → 不保存。

    Returns:
        comparison: {
            "model":             真实模型汇总统计,
            "random_sr{X.XX}":   各谱半径多种子统计（均值±标准差）,
            "chaos_boundary_note": 解释文字,
        }
    """
    n_regions = trajectories.shape[2]
    logger.info("=" * 50)
    logger.info("随机模型对照实验（多种子 × 多谱半径）")

    # Resolve spectral_radii list
    if spectral_radii is None:
        spectral_radii = sorted({round(spectral_radius, 3), 1.5, 2.0})

    # --- Real model stats ---
    model_stats = _compute_summary_stats(
        trajectories,
        attractor_results=attractor_results,
        lyapunov_results=lyapunov_results,
        response_matrix=response_matrix,
    )
    logger.info("  真实模型统计: %s", model_stats)

    # --- Random model: multiple seeds × multiple spectral radii ---
    comparison: Dict[str, object] = {"model": model_stats}
    seed_base = seed * 10 + 100  # avoid overlap with training seeds

    for sr in spectral_radii:
        logger.info(
            "  随机模型 ρ=%.2f (%d 个随机种子)…", sr, n_seeds
        )
        # True Wolf LLE across n_seeds independent W matrices
        lle_stats = _random_lle_multi_seed(
            n_regions=n_regions,
            spectral_radius=sr,
            n_seeds=n_seeds,
            lle_steps=min(random_steps, 600),
            renorm_steps=20,
            seed_base=seed_base,
        )
        # Trajectory statistics using the FIRST seed's trajectories
        # (generating n_init trajectories × n_seeds is expensive; one is enough
        # for variance/attractor stats since LLE already uses n_seeds)
        rand_trajs = run_random_trajectories(
            n_regions=n_regions,
            n_init=min(random_n_init, 30),
            steps=min(random_steps, 200),
            spectral_radius=sr,
            seed=seed_base,
            output_dir=output_dir if sr == spectral_radii[-1] else None,
        )
        traj_var = float(np.var(rand_trajs, axis=1).mean())

        rand_stats = {
            **lle_stats,
            "trajectory_variance": traj_var,
            "note": (
                f"ρ={sr:.2f}: {lle_stats['stability']}. "
                f"LLE = {lle_stats['mean_lyapunov']:.5f} ± {lle_stats['std_lyapunov']:.5f} "
                f"(over {n_seeds} independent random W matrices). "
                + ("ρ<1 guarantees stability (mathematical certainty for tanh contraction). " if sr < 1.0 else "")
                + ("ρ>1: chaos boundary depends on n and tanh compression; "
                   "for n≈190 the empirical chaos boundary is ρ≈1.5 "
                   "(tanh prevents linear-theory ρ=1 prediction). " if sr > 1.0 else "")
                + ("ρ=1 is the linear stability boundary (may still be stable due to tanh). " if sr == 1.0 else "")
            ),
        }
        key = f"random_sr{sr:.2f}"
        comparison[key] = rand_stats
        logger.info(
            "  随机模型 ρ=%.2f: LLE = %.5f ± %.5f [%s]",
            sr, lle_stats["mean_lyapunov"], lle_stats["std_lyapunov"], lle_stats["stability"]
        )

    # Summary note: where does the real model sit?
    real_lle = model_stats.get("mean_lyapunov")
    real_lle_source = model_stats.get("mean_lyapunov_source", "wolf")
    if real_lle is not None:
        stable_lles = [
            v["mean_lyapunov"]
            for k, v in comparison.items()
            if k.startswith("random_sr") and isinstance(v, dict)
            and v.get("spectral_radius", 1) < 1.0
        ]
        chaotic_lles = [
            v["mean_lyapunov"]
            for k, v in comparison.items()
            if k.startswith("random_sr") and isinstance(v, dict)
            and v.get("spectral_radius", 0) > 1.0
        ]
        source_label = f"({real_lle_source.capitalize()} LLE)" if real_lle_source != "wolf" else ""
        if stable_lles and chaotic_lles:
            mean_chaotic = np.mean(chaotic_lles)
            mean_stable  = np.mean(stable_lles)
            if real_lle < mean_stable:
                position = "比随机稳定基线更稳定"
            elif real_lle < mean_chaotic:
                position = "介于稳定与混沌之间"
            else:
                position = "与随机混沌系统相当（弱混沌或以上）"
            note = (
                f"真实模型 LLE={real_lle:.5f}{source_label}。"
                f"随机稳定基线(ρ<1) LLE≈{mean_stable:.5f}，"
                f"随机混沌基线(ρ>1) LLE≈{mean_chaotic:.5f}。"
                f"真实模型{position}于随机基线。"
            )
        else:
            note = f"真实模型 LLE={real_lle:.5f}{source_label}。"
        comparison["chaos_boundary_note"] = note
        logger.info("  %s", note)

    comparison["multi_seed_note"] = (
        f"每个谱半径使用 {n_seeds} 个独立随机矩阵（不同种子），报告均值±标准差。"
        "固定种子只能测试单个随机矩阵，换数据得到完全相同结果是因为 W 不变，"
        "而非因为结论鲁棒。"
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "analysis_comparison.json"

        def _json_safe(obj):
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if obj is None:
                return "N/A"
            return obj

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(comparison), fh, indent=2, ensure_ascii=False)
        logger.info("  → 已保存: %s", out_path)

    return comparison



# ---------------------------------------------------------------------------
# TASK 1: Graph-structure comparison (no surrogate dynamics)
# ---------------------------------------------------------------------------

def _run_tanh_trajectories(
    W: np.ndarray,
    n_init: int = 10,
    steps: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Simulate ``x(t+1) = clip(tanh(W @ x), 0, 1)`` from random initial states.

    This is the same nonlinear dynamics rule used internally by
    ``_wolf_lle_random`` and ``run_random_trajectories``.  It is used here
    to generate trajectory-based metrics (LLE, PCA dim, D2) for any arbitrary
    connectivity matrix ``W`` — brain, degree-preserving, or fully-random —
    so that all three network types are compared under **the same dynamics
    rule**, making the comparison scientifically valid.

    Parameters
    ----------
    W:
        Square connectivity matrix ``(N, N)``.
    n_init:
        Number of independent trajectories from uniform random initial states.
    steps:
        Prediction steps per trajectory.
    seed:
        Base random seed.

    Returns
    -------
    trajectories: shape ``(n_init, steps, N)``, dtype float32.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    trajs = np.zeros((n_init, steps, N), dtype=np.float32)
    for i in range(n_init):
        x = rng.random(N)
        for t in range(steps):
            x = np.clip(np.tanh(W @ x), 0.0, 1.0)
            trajs[i, t] = x.astype(np.float32)
    return trajs


def _tanh_dynamics_metrics(
    W: np.ndarray,
    n_init: int,
    steps: int,
    lle_steps: int,
    seed: int,
    rng: np.random.Generator,
) -> Dict:
    """Compute LLE, PCA dim-90%, and D2 for ``tanh(W @ x)`` dynamics.

    Parameters
    ----------
    W:
        Connectivity matrix ``(N, N)``.
    n_init:
        Trajectories for PCA/D2 computation.
    steps:
        Steps per trajectory for PCA/D2.
    lle_steps:
        Total Wolf-Benettin steps for LLE estimation.
    seed:
        Random seed for trajectories.
    rng:
        Shared RNG used for the Wolf-Benettin perturbation direction.

    Returns
    -------
    dict with keys ``lle``, ``pca_dim_90pct``, ``d2``.
    """
    N = W.shape[0]

    # LLE via Wolf-Benettin (same method as _wolf_lle_random)
    x0 = rng.random(N)
    lle = _wolf_lle_random(W, x0=x0, steps=lle_steps, rng=rng)

    # Trajectories for PCA and D2
    trajs = _run_tanh_trajectories(W, n_init=n_init, steps=steps, seed=seed)

    # PCA intrinsic dimension (90 % variance threshold)
    X = trajs.reshape(-1, N).astype(np.float64)
    X -= X.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        v = s ** 2
        cumvar = np.cumsum(v) / (v.sum() + 1e-30)
        pca_dim = int(np.searchsorted(cumvar, 0.90) + 1)
    except Exception:
        pca_dim = -1

    # Correlation dimension D2 (use first-region time series for speed)
    try:
        from analysis.embedding_dimension import correlation_dimension
        ts = trajs[:, :, 0].reshape(-1).astype(np.float64)
        d2 = float(correlation_dimension(ts, max_dim=20, n_points=min(2000, len(ts))))
    except Exception:
        d2 = float("nan")

    return {"lle": float(lle), "pca_dim_90pct": pca_dim, "d2": d2}

    """Maslov-Sneppen degree-preserving edge rewiring for a weighted matrix.

    Rewires ``n_swaps`` valid edge-pairs while preserving the row/column
    (degree) distribution.  Weights are also swapped so the weight
    distribution is preserved.
    """
    W_rw = W.copy()
    rng = np.random.default_rng(seed)
    # Work on the flat list of non-zero edges
    rows, cols = np.nonzero(W_rw)
    if len(rows) < 4:
        return W_rw
    completed = 0
    max_attempts = n_swaps * 10  # guard against infinite loops on dense graphs
    for _attempt in range(max_attempts):
        if completed >= n_swaps:
            break
        idx = rng.choice(len(rows), size=2, replace=False)
        r1, c1 = rows[idx[0]], cols[idx[0]]
        r2, c2 = rows[idx[1]], cols[idx[1]]
        if r1 == r2 or c1 == c2 or r1 == c2 or r2 == c1:
            continue
        # Swap (r1,c1)↔(r1,c2) and (r2,c2)↔(r2,c1)
        w1 = W_rw[r1, c1]
        w2 = W_rw[r2, c2]
        W_rw[r1, c1] = 0.0
        W_rw[r2, c2] = 0.0
        W_rw[r1, c2] = w1
        W_rw[r2, c1] = w2
        rows[idx[0]], cols[idx[0]] = r1, c2
        rows[idx[1]], cols[idx[1]] = r2, c1
        completed += 1
    return W_rw


def _spectral_metrics_rc(W: np.ndarray) -> Dict:
    """Return spectral radius, participation ratio, spectral gap."""
    try:
        eigs = np.linalg.eigvals(W)
        eig_abs = np.abs(eigs)
        sr = float(eig_abs.max())
        pr = float(eig_abs.sum() ** 2 / (eig_abs ** 2).sum()) if eig_abs.sum() > 1e-10 else 0.0
        sorted_abs = np.sort(eig_abs)[::-1]
        gap = float((sorted_abs[0] - sorted_abs[1]) / (sorted_abs[0] + 1e-10)) if len(sorted_abs) > 1 else 0.0
        return {"spectral_radius": sr, "participation_ratio": pr, "spectral_gap_ratio": gap}
    except Exception:
        return {"spectral_radius": float("nan"), "participation_ratio": float("nan"),
                "spectral_gap_ratio": float("nan")}



def _build_low_rank_matrix(W: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
    """Reconstruct W retaining only its top-``rank`` singular value components.

    This isolates the "low-rank structure" hypothesis: does the brain's
    dynamics arise because W has a low-rank information-compressing structure
    (like a principal component filter)?  If low-rank W produces dynamics
    indistinguishable from the full W, low-rank structure suffices to explain
    the observations.  If not, higher-rank geometry matters.

    Parameters
    ----------
    W:
        ``(N, N)`` connectivity matrix.
    rank:
        Number of singular values to keep (default: max(5, N // 20) = ~5%).

    Returns
    -------
    W_lr: Rank-``rank`` approximation of W, shape ``(N, N)``.
    """
    N = W.shape[0]
    if rank is None:
        rank = max(5, N // 20)
    rank = min(rank, N)
    try:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        s_trunc = np.zeros_like(s)
        s_trunc[:rank] = s[:rank]
        W_lr = (U * s_trunc) @ Vt
    except np.linalg.LinAlgError:
        W_lr = W.copy()
    return W_lr


def _build_community_matrix(
    W: np.ndarray,
    n_communities: int = 5,
    within_strength: float = 0.8,
    between_strength: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Construct a random block-diagonal (community-structured) matrix.

    This isolates the "community structure" hypothesis: does brain dynamics
    arise purely because of modular community organisation?  The constructed
    matrix has the same spectral radius as ``W`` (set externally by the
    caller's ``_normalise_w``), random community assignments, random intra-
    and inter-community weights scaled to ``within_strength`` and
    ``between_strength``.

    Parameters
    ----------
    W:
        Original connectivity matrix (used only for N).
    n_communities:
        Number of communities.
    within_strength:
        Mean absolute weight for intra-community edges.
    between_strength:
        Mean absolute weight for inter-community edges.
    seed:
        Random seed.

    Returns
    -------
    W_comm: ``(N, N)`` community-structured matrix.
    """
    N = W.shape[0]
    rng = np.random.default_rng(seed)
    # Random community assignment
    labels = rng.integers(0, n_communities, size=N)
    same_comm = (labels[:, None] == labels[None, :]).astype(np.float64)  # [N, N]
    # Scale and add noise
    noise = rng.standard_normal((N, N)) * 0.05
    W_comm = same_comm * within_strength + (1.0 - same_comm) * between_strength + noise
    # Remove self-loops
    np.fill_diagonal(W_comm, 0.0)
    return W_comm


def _build_hub_matrix(
    W: np.ndarray,
    n_hubs: Optional[int] = None,
    hub_strength: float = 3.0,
    nonhub_strength: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Construct a hub-dominated connectivity matrix.

    This isolates the "hub topology" hypothesis: does brain dynamics arise
    because W is dominated by a few highly-connected hub nodes (star-like
    topology)?  Hub identity is preserved from the real W (top ``n_hubs``
    nodes by row-sum absolute weight), but inter-hub and hub→non-hub weights
    are randomised so only the TOPOLOGY (which nodes are hubs) is kept,
    not the specific learned weight values.

    Parameters
    ----------
    W:
        Original connectivity matrix.  Hub nodes are identified from it.
    n_hubs:
        Number of hub nodes (default: max(5, N // 20) = ~5%).
    hub_strength:
        Mean weight for connections involving hub nodes.
    nonhub_strength:
        Mean weight for connections between non-hub nodes.
    seed:
        Random seed.

    Returns
    -------
    W_hub: ``(N, N)`` hub-dominated matrix.
    """
    N = W.shape[0]
    if n_hubs is None:
        n_hubs = max(5, N // 20)
    n_hubs = min(n_hubs, N)
    rng = np.random.default_rng(seed)
    # Identify hub nodes by row-sum absolute weight
    row_sums = np.abs(W).sum(axis=1)
    hub_idx = set(np.argsort(row_sums)[-n_hubs:].tolist())
    # Build weight matrix: is_hub[i] or is_hub[j] → hub_strength, else nonhub_strength
    is_hub = np.array([1.0 if i in hub_idx else 0.0 for i in range(N)])
    hub_mask = np.maximum(is_hub[:, None], is_hub[None, :])  # 1 if either is hub
    base = hub_mask * hub_strength + (1.0 - hub_mask) * nonhub_strength
    noise = rng.standard_normal((N, N)) * 0.05
    W_hub = base + noise
    np.fill_diagonal(W_hub, 0.0)
    return W_hub



def run_graph_structure_comparison(
    W: np.ndarray,
    trajectories: np.ndarray,
    lyapunov_results: Optional[Dict] = None,
    attractor_results: Optional[Dict] = None,
    n_random: int = 5,
    n_tanh_init: int = 10,
    tanh_steps: int = 200,
    lle_steps: int = 400,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """TASK 1: Compare brain graph against random-graph baselines.

    Compares three network types:

    1. **Original brain graph** (W unchanged) — GNN dynamics metrics from
       Phase 1 trajectories + tanh(W @ x) dynamics for fair cross-network
       comparison.
    2. **Degree-preserving rewire** (Maslov-Sneppen) — same degree/weight
       distribution, edges shuffled.  Tests whether the *specific* hub-to-hub
       wiring pattern (beyond degree sequence) matters for dynamics.
    3. **Fully random Gaussian** (same spectral radius) — tests whether
       the brain graph's dynamics are explained by scale alone.

    **Dynamics rule** for baselines 2 and 3: ``x(t+1) = clip(tanh(W @ x), 0, 1)``
    (same rule used in ``_wolf_lle_random`` and ``run_random_trajectories``).
    This allows direct comparison of LLE, PCA dim, and D2 across all three
    network types using an identical dynamics model — making the scientific
    conclusion about graph structure's role unambiguous.

    For the brain graph, both the GNN-derived metrics (from Phase 1 trajectories)
    **and** the tanh(W_brain @ x) metrics are reported; ``lle_gnn``, ``d2_gnn``,
    and ``pca_dim_gnn`` come from the GNN; ``lle``, ``d2``, and ``pca_dim_90pct``
    come from the tanh rule and are directly comparable to the baselines.

    Parameters
    ----------
    W:
        Square connectivity matrix ``(N, N)`` (response matrix or FC).
    trajectories:
        GNN-generated trajectories ``(n_traj, T, N)`` from Phase 1.
    lyapunov_results:
        Phase 3 Lyapunov results dict (provides GNN model LLE).
    attractor_results:
        Phase 3 attractor analysis dict (provides n_attractors).
    n_random:
        Number of independent random realisations to average over for the
        two random baselines (LLE averaged over n_random matrices; PCA/D2
        computed on first realisation for speed).
    n_tanh_init:
        Trajectories per matrix for PCA/D2 computation (tanh dynamics).
    tanh_steps:
        Steps per trajectory (tanh dynamics).
    lle_steps:
        Wolf-Benettin steps for LLE estimation.
    seed:
        Base random seed.
    output_dir:
        Directory to save ``random_graph_comparison.csv`` and
        ``random_graph_dynamics.png``.

    Returns
    -------
    dict with keys ``'brain_graph'``, ``'degree_preserving'``,
    ``'fully_random'``, and ``'note'``.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]
    trajs = np.asarray(trajectories, dtype=np.float64)
    rng = np.random.default_rng(seed)

    logger.info(
        "Graph structure comparison: N=%d, n_random=%d, n_tanh_init=%d, "
        "tanh_steps=%d, lle_steps=%d",
        N, n_random, n_tanh_init, tanh_steps, lle_steps,
    )

    # ── Spectral-radius normalization for fair tanh-dynamics comparison ────────
    # When W is derived from FC (Pearson correlations), its spectral radius is
    # typically >> 1 (Marchenko-Pastur bulk). At such high coupling, tanh(W@x)
    # saturates immediately and all three network types converge to the same
    # trivial fixed point — making LLE comparisons meaningless (all return the
    # floor sentinel).  To produce informative comparisons we normalise ALL
    # matrices to a fixed coupling strength before computing tanh dynamics.
    #
    # _TANH_COMPARE_SR = 2.0 is known to be mildly chaotic for random N=190
    # matrices (Wolf LLE ≈ +0.06 from empirical calibration in AGENTS.md).
    # This ensures all three networks operate in an interesting dynamical regime
    # while still allowing brain-specific topology to produce visibly different LLE.
    _TANH_COMPARE_SR: float = 2.0

    def _normalise_w(mat: np.ndarray) -> np.ndarray:
        """Scale mat so that its spectral radius equals _TANH_COMPARE_SR."""
        m = np.asarray(mat, dtype=np.float64)
        eigs = np.linalg.eigvals(m)
        sr = float(np.abs(eigs).max())
        if sr > 1e-8:
            m = m * (_TANH_COMPARE_SR / sr)
        return m

    # ── 1. GNN dynamics metrics for real brain graph ──────────────────────────
    real_lle_gnn = (lyapunov_results or {}).get("primary_mean", float("nan"))
    real_n_attractors = (attractor_results or {}).get("n_attractors", float("nan"))

    X = trajs.reshape(-1, N)
    X -= X.mean(axis=0)
    try:
        _, s_pca, _ = np.linalg.svd(X, full_matrices=False)
        v = s_pca ** 2
        cumvar = np.cumsum(v) / (v.sum() + 1e-30)
        real_pca_dim_gnn = int(np.searchsorted(cumvar, 0.90) + 1)
    except Exception:
        real_pca_dim_gnn = -1

    try:
        from analysis.embedding_dimension import correlation_dimension
        flat = trajs[:, :, 0].reshape(-1)
        real_d2_gnn = float(correlation_dimension(flat, max_dim=20, n_points=2000))
    except Exception:
        real_d2_gnn = float("nan")

    # ── 2. tanh(W @ x) dynamics for fair three-way comparison ────────────────
    # Helper to average metrics over n_random matrix realisations.
    # NOTE: each matrix is normalised to _TANH_COMPARE_SR so that the
    # comparison reflects topology differences, not coupling-scale differences.
    def _avg_tanh_metrics(matrices: List[np.ndarray]) -> Dict:
        """Average LLE over all matrices; PCA/D2 from first matrix only."""
        lles = []
        norm_matrices = [_normalise_w(m) for m in matrices]
        for W_i in norm_matrices:
            x0_i = rng.random(N)
            lles.append(_wolf_lle_random(W_i, x0=x0_i, steps=lle_steps, rng=rng))
        pca_d2 = _tanh_dynamics_metrics(
            norm_matrices[0],
            n_init=n_tanh_init,
            steps=tanh_steps,
            lle_steps=lle_steps,
            seed=seed,
            rng=rng,
        )
        return {
            "lle": float(np.mean(lles)),
            "lle_std": float(np.std(lles)),
            "pca_dim_90pct": pca_d2["pca_dim_90pct"],
            "d2": pca_d2["d2"],
        }

    def _avg_spectral(matrices: List[np.ndarray]) -> Dict:
        metrics_list = [_spectral_metrics_rc(m) for m in matrices]
        keys = ["spectral_radius", "participation_ratio", "spectral_gap_ratio"]
        return {
            k: float(np.nanmean([m[k] for m in metrics_list]))
            for k in keys
        }

    # Brain graph — tanh dynamics (comparable to baselines)
    # Spectral metrics are computed on the ORIGINAL W (true topology properties).
    # LLE/PCA/D2 are computed on W normalised to _TANH_COMPARE_SR (fair comparison).
    brain_spectral = _spectral_metrics_rc(W)
    W_norm = _normalise_w(W)
    brain_tanh = _tanh_dynamics_metrics(
        W_norm,
        n_init=n_tanh_init,
        steps=tanh_steps,
        lle_steps=lle_steps,
        seed=seed,
        rng=rng,
    )
    brain_entry = {
        "network": "Brain (GNN)",
        # GNN metrics — gold standard for the trained model
        "lle_gnn": float(real_lle_gnn),
        "d2_gnn": float(real_d2_gnn),
        "pca_dim_gnn": int(real_pca_dim_gnn),
        "n_attractors": float(real_n_attractors),
        # tanh(W @ x) metrics at _TANH_COMPARE_SR — directly comparable to baselines
        "lle": brain_tanh["lle"],
        "d2": brain_tanh["d2"],
        "pca_dim_90pct": brain_tanh["pca_dim_90pct"],
        "tanh_compare_sr": _TANH_COMPARE_SR,
        **brain_spectral,
    }

    # ── 3. Degree-preserving rewire (Maslov-Sneppen) ──────────────────────────
    # n_swaps = 10% of N² — empirically sufficient to decorrelate edge positions
    # while the max_attempts guard prevents infinite loops on dense graphs.
    n_swaps = max(N * N // 10, 100)
    dp_matrices = [
        _degree_preserving_rewire(W, n_swaps=n_swaps, seed=seed + i)
        for i in range(n_random)
    ]
    dp_tanh = _avg_tanh_metrics(dp_matrices)
    dp_spectral = _avg_spectral(dp_matrices)
    dp_entry = {
        "network": "Degree-preserving random",
        "n_attractors": float("nan"),   # no GNN trajectory available
        **dp_tanh,
        **dp_spectral,
    }

    # ── 4. Fully random Gaussian (normalised to _TANH_COMPARE_SR) ─────────────
    # Previously these were matched to brain's original spectral radius, which
    # was not meaningful when brain SR >> 1 (FC matrix).  Now all three networks
    # are run at the same _TANH_COMPARE_SR, making the comparison fair.
    fr_matrices: List[np.ndarray] = []
    for i in range(n_random):
        W_fr = rng.standard_normal((N, N))
        W_fr /= np.sqrt(N)
        eigs = np.linalg.eigvals(W_fr)
        sr_fr = float(np.abs(eigs).max())
        if sr_fr > 1e-8:
            W_fr *= _TANH_COMPARE_SR / sr_fr
        fr_matrices.append(W_fr)
    fr_tanh = _avg_tanh_metrics(fr_matrices)
    fr_spectral = _avg_spectral(fr_matrices)
    fr_entry = {
        "network": "Fully random Gaussian",
        "n_attractors": float("nan"),
        **fr_tanh,
        **fr_spectral,
    }

    # ── 5. Low-rank matrix ───────────────────────────────────────────────────
    # Reconstruct W from its top-rank singular values (≈5% of N).
    # Hypothesis: brain dynamics arise because W is approximately low-rank
    # (information compression / dominant modes hypothesis).
    # Result: if LLE/PCA-dim/D2 of W_lr ≈ brain, low-rank structure suffices.
    # Otherwise, full-rank geometry is essential — low-rank alone is NOT enough.
    _lr_rank = max(5, N // 20)
    lr_matrices = []
    for i in range(n_random):
        W_lr = _build_low_rank_matrix(W, rank=_lr_rank)
        # Perturb slightly per replicate to measure sensitivity
        noise_scale = 0.02 * float(np.abs(W_lr).mean())
        W_lr_i = W_lr + rng.standard_normal((N, N)) * noise_scale
        lr_matrices.append(W_lr_i)
    lr_tanh = _avg_tanh_metrics(lr_matrices)
    lr_spectral = _avg_spectral(lr_matrices)
    lr_entry = {
        "network": f"Low-rank (rank={_lr_rank})",
        "n_attractors": float("nan"),
        **lr_tanh,
        **lr_spectral,
    }

    # ── 6. Community-structured matrix ──────────────────────────────────────
    # Random block-diagonal matrix with strong within-community weights and
    # weak between-community weights (5 communities by default for N~190).
    # Hypothesis: brain dynamics arise from modular community organisation.
    _n_comm = max(3, min(10, N // 20))
    comm_matrices = [
        _build_community_matrix(W, n_communities=_n_comm, seed=seed + i)
        for i in range(n_random)
    ]
    comm_tanh = _avg_tanh_metrics(comm_matrices)
    comm_spectral = _avg_spectral(comm_matrices)
    comm_entry = {
        "network": f"Community structure (k={_n_comm})",
        "n_attractors": float("nan"),
        **comm_tanh,
        **comm_spectral,
    }

    # ── 7. Hub-dominated matrix ──────────────────────────────────────────────
    # Matrix where hub nodes (identified from real W) have high weight but
    # all specific weight values are randomised.  Tests whether the brain's
    # dynamics are explained merely by having a few highly-connected hubs.
    _n_hubs = max(5, N // 20)
    hub_matrices = [
        _build_hub_matrix(W, n_hubs=_n_hubs, seed=seed + i)
        for i in range(n_random)
    ]
    hub_tanh = _avg_tanh_metrics(hub_matrices)
    hub_spectral = _avg_spectral(hub_matrices)
    hub_entry = {
        "network": f"Hub-dominated (n_hubs={_n_hubs})",
        "n_attractors": float("nan"),
        **hub_tanh,
        **hub_spectral,
    }

    logger.info(
        "  Brain tanh(W@x) LLE=%.4f [analytical surrogate at ρ=%.1f; "
        "lle_gnn is in brain_graph entry]",
        brain_tanh["lle"], _TANH_COMPARE_SR,
    )
    logger.info(
        "  Degree-preserving LLE=%.4f±%.4f | Fully-random LLE=%.4f±%.4f",
        dp_tanh["lle"], dp_tanh["lle_std"],
        fr_tanh["lle"], fr_tanh["lle_std"],
    )
    logger.info(
        "  [Structured] Low-rank LLE=%.4f±%.4f | Community LLE=%.4f±%.4f | Hub LLE=%.4f±%.4f",
        lr_tanh["lle"], lr_tanh["lle_std"],
        comm_tanh["lle"], comm_tanh["lle_std"],
        hub_tanh["lle"], hub_tanh["lle_std"],
    )

    all_entries = [brain_entry, dp_entry, fr_entry, lr_entry, comm_entry, hub_entry]

    results = {
        "brain_graph": brain_entry,
        "degree_preserving": dp_entry,
        "fully_random": fr_entry,
        "low_rank": lr_entry,
        "community_structure": comm_entry,
        "hub_dominated": hub_entry,
        "tanh_compare_sr": _TANH_COMPARE_SR,
        "structured_comparison_note": (
            "Three ARTIFICIALLY CONSTRUCTED structured matrices were compared against "
            "the real brain graph and random baselines, under the SAME tanh dynamics rule. "
            f"Low-rank (rank={_lr_rank}): retains only the top-{_lr_rank} singular vectors "
            "of W — tests whether information-compression structure alone drives dynamics. "
            f"Community (k={_n_comm}): random block-diagonal matrix — tests whether modular "
            "community organisation alone drives dynamics. "
            f"Hub-dominated (n_hubs={_n_hubs}): random matrix with hub topology preserved "
            "from real W — tests whether hub-star architecture alone drives dynamics. "
            "KEY FINDING: if all three structured surrogates produce dynamics that differ "
            "substantially from the real model's GNN metrics (lle_gnn, d2_gnn, pca_dim_gnn), "
            "it proves that low-rank structure, community organisation, AND hub topology "
            "are individually INSUFFICIENT to explain the model's behaviour — the real "
            "explanation lies in their specific combination and the trained temporal dynamics."
        ),
        "note": (
            f"All network types are compared under the SAME dynamics rule "
            f"x(t+1)=clip(tanh(W_norm@x),0,1) where W_norm is each matrix scaled "
            f"to spectral radius {_TANH_COMPARE_SR} (near-chaotic regime for N={N}). "
            f"Spectral-radius normalisation is required because the brain FC matrix "
            f"has ρ >> 1, which would otherwise trivially saturate tanh for all "
            f"networks, making the comparison uninformative. "
            f"Original spectral radii are preserved in each entry. "
            f"GNN dynamics metrics (lle_gnn, d2_gnn, pca_dim_gnn) are "
            f"additionally reported for the brain graph from Phase 1 trajectories."
        ),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_rg_csv(all_entries, output_dir / "random_graph_comparison.csv")
        _save_rg_plot(all_entries, output_dir / "random_graph_dynamics.png")

    return results



def _save_rg_csv(rows: List[Dict], path: Path) -> None:
    fieldnames = [
        "network",
        "lle", "lle_std", "lle_gnn",
        "d2", "d2_gnn",
        "pca_dim_90pct", "pca_dim_gnn",
        "spectral_radius", "participation_ratio", "spectral_gap_ratio",
    ]
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("  Saved %s", path)
    except Exception as exc:
        logger.warning("  CSV save failed: %s", exc)


def _save_rg_plot(rows: List[Dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except Exception:
            pass
    except ImportError:
        return

    labels = [r["network"] for r in rows]
    n = len(rows)
    x = np.arange(n)
    # Use a qualitative colormap that works for up to ~10 entries
    _cmap = plt.get_cmap("tab10")
    colors = [_cmap(i % 10) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(max(15, 3 * n), 6))
    fig.suptitle("Graph Structure Comparison (brain vs random & structured baselines)", fontsize=11)

    # Panel 1: LLE comparison (all under same tanh dynamics)
    lle_vals = [r.get("lle", float("nan")) for r in rows]
    lle_stds = [r.get("lle_std", 0.0) or 0.0 for r in rows]
    valid = [not np.isnan(v) for v in lle_vals]
    bar_vals = [v if not np.isnan(v) else 0.0 for v in lle_vals]
    axes[0].bar(x, bar_vals, color=colors, width=0.6,
                yerr=[s if v else 0.0 for s, v in zip(lle_stds, valid)],
                capsize=4)
    # Also show GNN LLE where available
    gnn_lles = [r.get("lle_gnn") for r in rows]
    for xi, gv in enumerate(gnn_lles):
        if gv is not None and not np.isnan(float(gv)):
            axes[0].scatter(xi, float(gv), marker="*", s=120, color="black", zorder=5,
                            label="GNN LLE" if xi == 0 else "")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("LLE (tanh dynamics)")
    axes[0].set_title("Largest Lyapunov Exponent\n(* = GNN model LLE)")
    axes[0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Panel 2: Participation ratio (spectral dimensionality proxy)
    pr_vals = [r.get("participation_ratio", 0.0) or 0.0 for r in rows]
    axes[1].bar(x, pr_vals, color=colors, width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Participation ratio")
    axes[1].set_title("Spectral participation ratio\n(eigenvalue dimensionality proxy)")
    axes[1].grid(True, axis="y", alpha=0.3)

    # Panel 3: PCA intrinsic dim (tanh dynamics trajectories)
    pca_vals = [max(r.get("pca_dim_90pct", -1) or -1, 0) for r in rows]
    axes[2].bar(x, pca_vals, color=colors, width=0.6)
    # Also show GNN pca dim where available
    gnn_pcas = [r.get("pca_dim_gnn") for r in rows]
    for xi, gv in enumerate(gnn_pcas):
        if gv is not None and float(gv) > 0:
            axes[2].scatter(xi, float(gv), marker="*", s=120, color="black", zorder=5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("PCA intrinsic dim (90% variance)")
    axes[2].set_title("Intrinsic dimensionality\n(* = GNN model dim)")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path)
