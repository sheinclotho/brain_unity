"""
Jacobian Spectrum Analysis
===========================

在吸引子附近估计系统的 **数值 Jacobian** J = ∂F/∂x，并对其谱结构做完整分析：

  1. 特征值分布（复平面散点图）
  2. Slow modes：Re(λ) ≈ 0 的方向 → 系统在哪些方向上"几乎不收缩也不扩张"
  3. Hopf 分岔检测：是否存在复共轭对 λ = a ± ib（ib 对应振荡频率）
  4. 空间本征模：每个主要模态对应哪些脑区（特征向量的空间权重）

方法：有限差分数值 Jacobian
---------------------------
对吸引子状态 x*，沿每个方向 eᵢ 施加小扰动：

  J[:,i] ≈ (F(x* + ε·eᵢ) − F(x*)) / ε

其中 F 为 TwinBrainDigitalTwin 的一步预测 ``rollout(x0, steps=1)``。
对 N=190 个方向共需 N+1 次模型调用（1 次基准 + N 次扰动）。

计算成本（每个吸引子状态）：
  N+1 次 ``simulator.rollout(steps=1)`` 调用。
  N=190 → 191 次调用/状态；推荐在 n_states=3–5 个吸引子状态上取平均。

输出文件
--------
  jacobian_eigenvalues.npy       — shape (n_states, N), 复数特征值（均值谱）
  jacobian_report.json           — 所有数值指标
  jacobian_complex_plane.png     — 复平面特征值散点图
  jacobian_slow_modes.png        — 慢模态空间权重（脑区 × 模态热图）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 慢模态阈值：|Re(λ)| < this → slow mode
_SLOW_MODE_RE_THRESH: float = 0.05
# Hopf 检测：振荡频率 Im(λ)/(2π) 的最小阈值（nats/step）
_HOPF_IM_THRESH: float = 0.01
# 显示的顶部空间模态数量
_N_TOP_MODES: int = 6


# ─────────────────────────────────────────────────────────────────────────────
# Core: numerical Jacobian estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_jacobian_at_point(
    simulator,
    x_star: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    在给定状态 x* 处用中心差分估计单步 Jacobian J = ∂F/∂x。

    J[:, i] ≈ (F(x* + ε·eᵢ) − F(x* − ε·eᵢ)) / (2ε)

    中心差分比前向差分精度高一阶（O(ε²) 而非 O(ε)），在 ε=1e-4 时足以
    区分局部展开与收缩方向。

    Args:
        simulator: BrainDynamicsSimulator 实例（TwinBrainDigitalTwin 模式）。
        x_star:    吸引子状态，shape (N,)。
        epsilon:   扰动幅度（推荐 1e-4 ~ 1e-3 以平衡精度和数值稳定性）。

    Returns:
        J: shape (N, N)，float64 Jacobian 矩阵。
           J[j, i] = ∂F_j / ∂x_i。
    """
    N = len(x_star)
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))

    # Baseline F(x*)
    traj_base, _ = simulator.rollout(
        x0=x_star.astype(np.float32), steps=1
    )
    f_base = traj_base[0].astype(np.float64)  # (N,)

    J = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        # Forward perturbation
        x_fwd = x_star.copy().astype(np.float64)
        x_fwd[i] += epsilon
        if _bounds is not None:
            x_fwd = np.clip(x_fwd, _bounds[0], _bounds[1])

        # Backward perturbation
        x_bwd = x_star.copy().astype(np.float64)
        x_bwd[i] -= epsilon
        if _bounds is not None:
            x_bwd = np.clip(x_bwd, _bounds[0], _bounds[1])

        traj_fwd, _ = simulator.rollout(
            x0=x_fwd.astype(np.float32), steps=1
        )
        traj_bwd, _ = simulator.rollout(
            x0=x_bwd.astype(np.float32), steps=1
        )

        f_fwd = traj_fwd[0].astype(np.float64)
        f_bwd = traj_bwd[0].astype(np.float64)

        # Effective step using signed difference (avoids masking asymmetric clipping).
        # When both fwd and bwd are clipped the same way, the sign is preserved and
        # the Jacobian estimate degrades gracefully rather than silently dividing by
        # the wrong magnitude.
        h = float(x_fwd[i] - x_bwd[i])
        if abs(h) < 1e-30:
            # Both clipped to the same boundary; skip this column
            continue
        J[:, i] = (f_fwd - f_bwd) / h

    return J


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian spectrum analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_jacobian_spectrum(
    J: np.ndarray,
    dt: float = 1.0,
) -> Dict:
    """
    对 Jacobian 矩阵 J 做完整谱分析。

    Args:
        J:  shape (N, N)，float64 Jacobian。
        dt: 时间步长（秒），用于将离散特征值 λ_discrete 转换为连续时间
            等价：λ_continuous = log(λ_discrete) / dt。
            默认 1.0（以"步"为单位，不做转换）。

    Returns:
        dict 包含:
          eigenvalues          : np.ndarray (N,) 复数，λ_discrete
          eigenvalues_ct       : np.ndarray (N,) 复数，连续时间等价 log(λ)/dt
          spectral_radius      : float，max|λ|
          n_slow_modes         : int，|Re(λ_ct)| < _SLOW_MODE_RE_THRESH 的数量
          slow_mode_indices    : List[int]，慢模态排名（按 |Re(λ_ct)| 升序）
          n_hopf_pairs         : int，复共轭对 a±ib 的数量（b > _HOPF_IM_THRESH）
          hopf_frequencies_hz  : List[float]，Hopf 频率 Im(λ_ct)/(2π)（Hz）
          dominant_oscillation_hz: float，主要振荡频率（Hz，若存在）
          eigenvectors         : np.ndarray (N, N)，右特征向量（列）
          top_mode_weights     : np.ndarray (n_slow, N)，慢模态的脑区权重
    """
    J64 = np.asarray(J, dtype=np.float64)
    N = J64.shape[0]

    # Compute eigenvalues and right eigenvectors
    eigvals, eigvecs = np.linalg.eig(J64)

    # Sort by |λ| descending
    idx_sort = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]

    rho = float(np.abs(eigvals).max()) if N > 0 else 0.0

    # Continuous-time equivalent: λ_ct = log(λ_discrete) / dt
    # Guard against zero eigenvalues
    with np.errstate(divide="ignore", invalid="ignore"):
        eigvals_ct = np.where(
            np.abs(eigvals) > 1e-30,
            np.log(eigvals.astype(np.complex128)) / dt,
            complex(-1e6, 0.0),
        )

    re_ct = eigvals_ct.real
    im_ct = eigvals_ct.imag

    # ── Slow modes: |Re(λ_ct)| < threshold ───────────────────────────────────
    slow_mask = np.abs(re_ct) < _SLOW_MODE_RE_THRESH
    slow_indices = np.where(slow_mask)[0].tolist()
    n_slow = len(slow_indices)

    # ── Hopf pairs: Im(λ_ct) > threshold (oscillatory) ───────────────────────
    # A complex conjugate pair counts as one Hopf mode
    hopf_mask = np.abs(im_ct) > _HOPF_IM_THRESH
    hopf_freqs_all = np.abs(im_ct[hopf_mask]) / (2 * np.pi)   # Hz (per step)
    # Deduplicate conjugate pairs: keep unique positive frequencies
    hopf_freqs_unique = sorted(set(np.round(hopf_freqs_all, 6)))
    # Remove near-duplicates (conjugates are equal freq)
    hopf_freqs_dedup: List[float] = []
    for f in hopf_freqs_unique:
        if not hopf_freqs_dedup or abs(f - hopf_freqs_dedup[-1]) > 1e-4:
            hopf_freqs_dedup.append(float(f))
    n_hopf = len(hopf_freqs_dedup)

    # Dominant oscillation: Hopf pair with largest |λ| (most influential)
    if n_hopf > 0:
        # Among Hopf modes, pick the one with largest spectral magnitude
        hopf_all_idx = np.where(hopf_mask)[0]
        hopf_mags = np.abs(eigvals[hopf_all_idx])
        dom_idx = hopf_all_idx[int(np.argmax(hopf_mags))]
        dominant_osc_hz = float(abs(im_ct[dom_idx]) / (2 * np.pi))
    else:
        dominant_osc_hz = 0.0

    # ── Spatial eigenmodes: |weights| per brain region ────────────────────────
    # For slow modes, the eigenvector magnitude gives brain-region participation
    n_top = min(_N_TOP_MODES, N)
    top_indices = slow_indices[:n_top] if len(slow_indices) >= n_top else list(range(n_top))
    top_mode_weights = np.abs(eigvecs[:, top_indices].real)  # (N, n_top)

    logger.info(
        "Jacobian 谱分析完成: N=%d, ρ=%.4f, "
        "n_slow=%d (|Re|<%.2f), n_Hopf=%d, dominant_osc=%.4f Hz/step",
        N, rho, n_slow, _SLOW_MODE_RE_THRESH,
        n_hopf, dominant_osc_hz,
    )

    return {
        "eigenvalues": eigvals,
        "eigenvalues_ct": eigvals_ct,
        "spectral_radius": rho,
        "n_slow_modes": n_slow,
        "slow_mode_indices": slow_indices,
        "n_hopf_pairs": n_hopf,
        "hopf_frequencies_hz": hopf_freqs_dedup,
        "dominant_oscillation_hz": dominant_osc_hz,
        "eigenvectors": eigvecs,
        "top_mode_weights": top_mode_weights,
        "n_top_modes_shown": n_top,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integration function
# ─────────────────────────────────────────────────────────────────────────────

def run_jacobian_analysis(
    simulator,
    trajectories: np.ndarray,
    tail_steps: int = 20,
    n_states: int = 3,
    epsilon: float = 1e-4,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    在多个吸引子状态上估计 Jacobian 并做谱分析。

    流程：
      1. 从 ``trajectories`` 末尾 ``tail_steps`` 步提取吸引子状态（均值）。
      2. 随机选 ``n_states`` 个状态，每个状态估计数值 Jacobian。
      3. 对所有 Jacobian 取均值，做完整谱分析。

    计算成本：n_states × (2N+1) 次 simulator.rollout(steps=1) 调用。
    N=190, n_states=3 → 1143 次调用（约 2–5 分钟 CPU）。

    Args:
        simulator:   BrainDynamicsSimulator 实例。
        trajectories: shape (n_traj, T, N)。
        tail_steps:  从末尾取多少步均值作为吸引子状态（默认 20）。
        n_states:    评估 Jacobian 的吸引子状态数量（默认 3）。
        epsilon:     有限差分步长（默认 1e-4）。
        seed:        随机种子。
        output_dir:  结果保存目录；None → 不保存。

    Returns:
        dict 包含 Jacobian 谱分析结果（同 analyze_jacobian_spectrum）
        加上 mean_J（均值 Jacobian）。
    """
    n_traj, T, N = trajectories.shape
    rng = np.random.default_rng(seed)

    # Extract attractor states from trajectory tails
    tail = trajectories[:, -tail_steps:, :].mean(axis=1)   # (n_traj, N)
    sample_idx = rng.choice(n_traj, size=min(n_states, n_traj), replace=False)

    J_list: List[np.ndarray] = []
    for i, idx in enumerate(sample_idx):
        x_star = tail[idx].astype(np.float64)
        logger.info("Jacobian 估计: 状态 %d/%d (traj=%d)", i + 1, len(sample_idx), idx)
        try:
            J_i = estimate_jacobian_at_point(simulator, x_star, epsilon=epsilon)
            J_list.append(J_i)
        except Exception as exc:
            logger.warning("  状态 %d Jacobian 估计失败: %s", idx, exc)

    if not J_list:
        raise RuntimeError("所有吸引子状态的 Jacobian 估计均失败。")

    mean_J = np.mean(np.stack(J_list, axis=0), axis=0)   # (N, N)

    dt = getattr(simulator, "dt", 1.0)
    spec_result = analyze_jacobian_spectrum(mean_J, dt=dt)
    spec_result["mean_J"] = mean_J
    spec_result["n_states_used"] = len(J_list)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "jacobian_mean.npy", mean_J)
        np.save(out / "jacobian_eigenvalues.npy",
                spec_result["eigenvalues"])

        # JSON-serialisable report (no arrays)
        report = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in spec_result.items()
            if k not in ("eigenvectors", "mean_J", "top_mode_weights")
        }
        report["dt"] = dt
        with open(out / "jacobian_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存 Jacobian 报告: %s", out / "jacobian_report.json")

        _try_plot_jacobian(spec_result, out)

    return spec_result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_jacobian(spec_result: Dict, output_dir: Path) -> None:
    """绘制复平面特征值图和慢模态空间权重热图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eigvals = spec_result["eigenvalues"]
    eigvals_ct = spec_result["eigenvalues_ct"]
    top_weights = spec_result["top_mode_weights"]   # (N, n_top)
    n_slow = spec_result["n_slow_modes"]
    hopf_hz = spec_result["hopf_frequencies_hz"]
    rho = spec_result["spectral_radius"]

    # ── Panel 1: Complex plane (discrete eigenvalues) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    re_d = eigvals.real
    im_d = eigvals.imag
    mags = np.abs(eigvals)

    sc = ax.scatter(re_d, im_d, c=mags, cmap="viridis",
                    s=18, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label="|λ|")
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "r--", lw=0.8, alpha=0.5,
            label="|λ|=1 (neutral)")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Re(λ)  [discrete]")
    ax.set_ylabel("Im(λ)  [discrete]")
    ax.set_title(f"Jacobian Eigenvalues (Discrete)\n"
                 f"ρ={rho:.4f}, n_slow={n_slow}, n_Hopf={len(hopf_hz)}")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Top slow-mode spatial weights (brain regions) ────────────────
    ax2 = axes[1]
    n_top = top_weights.shape[1]
    im = ax2.imshow(
        top_weights.T,          # (n_top, N) → rows = modes, cols = regions
        aspect="auto",
        cmap="Reds",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax2, label="|eigenvector component|")
    ax2.set_xlabel("Brain Region Index")
    ax2.set_ylabel("Mode Rank")
    ax2.set_yticks(range(n_top))
    ax2.set_yticklabels([f"Mode {i+1}" for i in range(n_top)], fontsize=8)
    title_parts = [f"Top {n_top} Slow Modes — Spatial Weights"]
    if hopf_hz:
        freqs_str = ", ".join(f"{f:.4f}" for f in hopf_hz[:3])
        title_parts.append(f"Hopf freqs (1/step): {freqs_str}")
    ax2.set_title("\n".join(title_parts), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "jacobian_spectrum.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存 Jacobian 谱图: %s", output_dir / "jacobian_spectrum.png")
