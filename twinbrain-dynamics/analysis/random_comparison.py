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

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

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

    def _step(x: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(W @ x), 0.0, 1.0)

    # Warm up base trajectory to avoid transient
    x_base = x0.copy()
    for _ in range(renorm_steps * 3):
        x_base = _step(x_base)

    # Random unit-norm perturbation direction
    direction = rng.standard_normal(n).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-30
    x_pert = x_base + eps * direction

    log_growths: List[float] = []
    n_periods = max(1, steps // renorm_steps)

    for _ in range(n_periods):
        for _ in range(renorm_steps):
            x_base = _step(x_base)
            x_pert = _step(x_pert)

        delta = x_pert - x_base
        r = float(np.linalg.norm(delta))
        if r > 1e-30:
            log_growths.append(np.log(r / eps))
            # Renormalize: keep separation = eps
            x_pert = x_base + eps * delta / r
        else:
            # Perturbation collapsed to near machine precision (strong contraction).
            # The exact log-growth is unreliable due to floating-point limits;
            # skip this period rather than recording a spurious large value, and
            # re-perturb to continue the Wolf traversal.
            direction = rng.standard_normal(n).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-30
            x_pert = x_base + eps * direction

    if not log_growths:
        return 0.0
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

