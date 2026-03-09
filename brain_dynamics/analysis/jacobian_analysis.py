"""
Jacobian Spectrum Analysis — Dynamic Mode Decomposition (DMD)
==============================================================

从自由动力学轨迹估计系统的线性化转移算子 A（DMD 算子），并对其谱结构做完整分析：

  1. 特征值分布（复平面散点图）
  2. Slow modes：|λ| ≈ 1 的方向 → 系统在哪些方向上"几乎不收缩也不扩张"
  3. Hopf 分岔检测：是否存在复共轭对 λ = |r|e^{±iθ}（θ 对应振荡频率）
  4. 空间本征模：每个主要模态对应哪些脑区（特征向量的空间权重）

为什么不用有限差分数值 Jacobian？（设计决策，AGENTS.md 记录）
------------------------------------------------------------------
TwinBrainDigitalTwin 使用上下文窗口（通常 200 步）的 Conv1d + 注意力机制编码器。
`rollout(x0, steps=1)` 将 x0 注入上下文的**最后一步**，模型编码时对全部
200 步做加权平均，单步扰动 ε=1e-4 被 199 步历史稀释：

    f_fwd - f_bwd ≈ O(ε / context_length)

在 float32 精度下（~1e-7 相对精度），对 context_length=200 的模型，ε=1e-4
产生的输出差异 ≈ 5e-7，低于 float32 的机器精度 × 输出幅值，导致所有有限差分
列为零 → J=0 → 谱半径 ρ=0（实测结果）。

正确方法：Dynamic Mode Decomposition (DMD)
------------------------------------------
从自由动力学轨迹 {x(t), x(t+1)} 构建时序对，拟合最优线性转移算子：

    A = argmin ‖X₁ − A X₀‖_F

其中 X₀, X₁ 是从已有轨迹中提取的时序对矩阵（形状 M×N，M≫N）。
A 的特征值正是系统的离散时间 DMD 模态。

优势：
  - 不需要任何额外模型调用（复用步骤 3 的轨迹）
  - 绕过上下文稀释问题（直接从真实输出序列推断）
  - 数值稳定：正规方程 + Tikhonov 正则化，避免过拟合
  - 科学标准算法（Schmid 2010 JFM；Brunton & Kutz 2022 "Data-Driven Science"）

输出文件
--------
  jacobian_dmd.npy             — shape (N, N)，DMD 算子 A
  jacobian_eigenvalues.npy     — shape (N,)，复数特征值
  jacobian_report.json         — 所有数值指标
  jacobian_spectrum.png        — 复平面特征值图 + 慢模态空间权重热图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 慢模态阈值：|Re(λ_ct)| < this → slow mode（连续时间，1/s 单位）
_SLOW_MODE_RE_THRESH: float = 0.05
# Hopf 检测：振荡频率 Im(λ_ct)/(2π) 的最小阈值（Hz）
_HOPF_IM_THRESH: float = 0.01
# 显示的顶部空间模态数量
_N_TOP_MODES: int = 6


# ─────────────────────────────────────────────────────────────────────────────
# Core: DMD-based transition operator estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_jacobian_dmd(
    trajectories: np.ndarray,
    burnin: int = 0,
    reg_alpha: float = 1e-6,
) -> np.ndarray:
    """
    从自由动力学轨迹用 DMD 估计最优线性转移算子 A（= 经验 Jacobian）。

    对所有轨迹中的连续时间步对 (x(t), x(t+1)) 求解 Tikhonov 正则化最小二乘：

        A = argmin ‖X₁ − A X₀‖_F + α ‖A‖_F

    其中：
        X₀: 所有时步 t   的状态，形状 (M, N)
        X₁: 所有时步 t+1 的状态，形状 (M, N)
        M = n_traj × (T − burnin − 1)

    正规方程求解（数值稳定，不需要 SVD 截断）：

        (X₀ᵀX₀ + αI) Aᵀ = X₀ᵀX₁
        A = solve(...).T

    Args:
        trajectories: shape (n_traj, T, N)。
        burnin:       每条轨迹跳过的初始步数（去除瞬态）。
        reg_alpha:    Tikhonov 正则化系数（相对于 trace(X₀ᵀX₀)/N）。
                      默认 1e-6，在 M≫N 时几乎不改变结果。

    Returns:
        A: shape (N, N)，float64，DMD 转移矩阵。
           A[j, i] = 状态 i 对状态 j 的下一步贡献权重。
    """
    n_traj, T, N = trajectories.shape
    T_eff = T - burnin
    if T_eff < 3:
        raise ValueError(
            f"burnin={burnin} 后有效步数 T_eff={T_eff} < 3，"
            "无法构建时序对矩阵（至少需要 2 个 (x_t, x_{t+1}) 对，即 T_eff ≥ 3）。"
            " 请减小 burnin 或增加 steps。"
        )

    # Collect consecutive-step pairs from all trajectories
    X0_list: List[np.ndarray] = []
    X1_list: List[np.ndarray] = []
    for k in range(n_traj):
        seg = trajectories[k, burnin:, :].astype(np.float64)  # (T_eff, N)
        X0_list.append(seg[:-1])   # (T_eff-1, N)
        X1_list.append(seg[1:])    # (T_eff-1, N)

    X0 = np.vstack(X0_list)  # (M, N)
    X1 = np.vstack(X1_list)  # (M, N)
    M = X0.shape[0]

    logger.info(
        "DMD 转移算子估计: M=%d 时步对, N=%d 维, burnin=%d",
        M, N, burnin,
    )

    # Normal equations: (X0.T @ X0 + alpha*I) @ A.T = X0.T @ X1
    XTX = X0.T @ X0                       # (N, N)
    XTY = X0.T @ X1                       # (N, N)
    alpha = float(np.trace(XTX)) / N * reg_alpha
    try:
        A_T = np.linalg.solve(XTX + alpha * np.eye(N), XTY)   # (N, N)
    except np.linalg.LinAlgError:
        logger.warning("DMD: np.linalg.solve 失败，改用 lstsq（可能存在奇异性）。")
        A_T, _, _, _ = np.linalg.lstsq(
            XTX + alpha * np.eye(N), XTY, rcond=None
        )

    A = A_T.T   # (N, N), convention: x1 = A @ x0
    return A


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_json_serialisable(v):
    """Convert numpy arrays (real or complex) to JSON-serialisable types."""
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"real": v.real.tolist(), "imag": v.imag.tolist()}
        return v.tolist()
    return v


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
    用 DMD（Dynamic Mode Decomposition）从自由动力学轨迹估计线性化转移算子，
    并对其谱结构做完整分析。

    流程：
      1. 计算 burnin = tail_steps（跳过每条轨迹的初始瞬态）。
      2. 从所有轨迹的 burnin 之后部分构建时序对矩阵 (X0, X1)。
      3. Tikhonov 正则化最小二乘拟合转移算子 A（= DMD 算子）。
      4. 对 A 做完整谱分析（特征值、慢模态、Hopf 对、空间模态）。

    注意：此函数不再调用 simulator.rollout()（零额外模型调用）。
    ``n_states`` 和 ``epsilon`` 参数保留但不使用，仅为向后兼容。

    Args:
        simulator:    BrainDynamicsSimulator 实例（仅用于读取 dt）。
        trajectories: shape (n_traj, T, N)。
        tail_steps:   用作 burnin 的步数（跳过初始瞬态）。
        n_states:     保留参数（兼容旧配置），不再使用。
        epsilon:      保留参数（兼容旧配置），不再使用。
        seed:         保留参数（兼容旧配置），不再使用。
        output_dir:   结果保存目录；None → 不保存。

    Returns:
        dict 包含 Jacobian 谱分析结果（同 analyze_jacobian_spectrum）
        加上 dmd_A（DMD 算子矩阵）和 method="dmd"。
    """
    n_traj, T, N = trajectories.shape
    burnin = max(0, min(tail_steps, T // 2))

    dmd_A = estimate_jacobian_dmd(trajectories, burnin=burnin)

    dt = getattr(simulator, "dt", 1.0)
    spec_result = analyze_jacobian_spectrum(dmd_A, dt=dt)
    spec_result["dmd_A"] = dmd_A
    spec_result["mean_J"] = dmd_A        # backward-compatible alias
    spec_result["n_states_used"] = n_traj
    spec_result["method"] = "dmd"
    spec_result["burnin_used"] = burnin

    # ── Supplement 2: DMD modal energy ────────────────────────────────────────
    # energy_i = ||mode_i||²  (eigenvector 2-norm squared, real part magnitude).
    # Modes are already sorted by |λ| descending in analyze_jacobian_spectrum.
    eigvecs = spec_result.get("eigenvectors")
    if eigvecs is not None:
        mode_energy = (np.abs(eigvecs) ** 2).sum(axis=0)   # (N,)
        spec_result["mode_energy"] = mode_energy

    # ── New Test 1: Dominant Hopf mode identification ─────────────────────────
    hopf_info = _extract_dominant_hopf_mode(spec_result)
    spec_result["hopf_dominant"] = hopf_info

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "jacobian_dmd.npy", dmd_A)
        np.save(out / "jacobian_eigenvalues.npy",
                spec_result["eigenvalues"])

        # Save eigenvectors (DMD modes) for downstream use.
        if eigvecs is not None:
            np.save(out / "dmd_modes.npy", eigvecs)
            logger.info("  → 保存 DMD 模态向量: %s", out / "dmd_modes.npy")

        # JSON-serialisable report (no arrays).
        # Complex arrays (eigenvalues, eigenvalues_ct) must be split into
        # separate real and imaginary lists; json.dump does not handle Python
        # complex objects.
        report = {
            k: _to_json_serialisable(v)
            for k, v in spec_result.items()
            if k not in ("eigenvectors", "dmd_A", "mean_J", "top_mode_weights",
                         "mode_energy")
        }
        report["dt"] = dt
        with open(out / "jacobian_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存 Jacobian 报告: %s", out / "jacobian_report.json")

        # Hopf modes JSON (New Test 1)
        with open(out / "hopf_modes.json", "w", encoding="utf-8") as fh:
            json.dump(hopf_info, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存 Hopf 模态: %s", out / "hopf_modes.json")

        _try_plot_jacobian(spec_result, out)
        _try_plot_hopf_eigenvalues(spec_result, out)
        _try_plot_dmd_mode_energy(spec_result, out)

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


# ─────────────────────────────────────────────────────────────────────────────
# New Test 1: Dominant Hopf Mode Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_dominant_hopf_mode(spec_result: Dict) -> Dict:
    """
    Extract the dominant Hopf bifurcation mode from a DMD spectrum result.

    Selects the Hopf pair with the largest |Im(μ_i)| (strongest oscillation)
    among those with |Re(μ_i)| closest to zero (nearest criticality).

    The "dominant Hopf mode" is defined as:
      - Among all eigenvalues with |Im(λ_ct)| > _HOPF_IM_THRESH,
        pick the one with maximum |λ| (most energetic oscillatory mode).

    Returns a dict with:
      hopf_frequency:   float — dominant Hopf frequency in units of 1/step
      hopf_growth_rate: float — Re(λ_ct) of the dominant Hopf pair
      hopf_pair_index:  int   — index in sorted eigenvalue array
      hopf_mode_vector: list  — |eigenvector| of the dominant mode (brain weights)
      n_hopf_pairs:     int   — total number of Hopf pairs found
      has_hopf:         bool  — whether any Hopf pair was found
    """
    eigvals_ct = np.asarray(spec_result.get("eigenvalues_ct", []))
    eigvals = np.asarray(spec_result.get("eigenvalues", []))
    eigvecs = spec_result.get("eigenvectors")

    if len(eigvals_ct) == 0:
        return {
            "has_hopf": False,
            "n_hopf_pairs": 0,
            "hopf_frequency": None,
            "hopf_growth_rate": None,
            "hopf_pair_index": None,
            "hopf_mode_vector": None,
        }

    im_ct = eigvals_ct.imag
    re_ct = eigvals_ct.real
    hopf_mask = np.abs(im_ct) > _HOPF_IM_THRESH
    n_hopf = int(hopf_mask.sum()) // 2  # conjugate pairs

    if not hopf_mask.any():
        return {
            "has_hopf": False,
            "n_hopf_pairs": 0,
            "hopf_frequency": None,
            "hopf_growth_rate": None,
            "hopf_pair_index": None,
            "hopf_mode_vector": None,
        }

    # Among Hopf modes, pick dominant by largest |λ_discrete|
    hopf_idx = np.where(hopf_mask)[0]
    hopf_mags = np.abs(eigvals[hopf_idx])
    dom_local = int(np.argmax(hopf_mags))
    dom_idx = int(hopf_idx[dom_local])

    freq = float(abs(im_ct[dom_idx]) / (2 * np.pi))
    growth = float(re_ct[dom_idx])

    mode_vec: Optional[List[float]] = None
    if eigvecs is not None and dom_idx < eigvecs.shape[1]:
        mode_vec = np.abs(eigvecs[:, dom_idx].real).tolist()

    return {
        "has_hopf": True,
        "n_hopf_pairs": n_hopf,
        "hopf_frequency": round(freq, 6),
        "hopf_growth_rate": round(growth, 6),
        "hopf_pair_index": dom_idx,
        "hopf_mode_vector": mode_vec,
    }


def _try_plot_hopf_eigenvalues(spec_result: Dict, output_dir: Path) -> None:
    """
    New Test 1 plot: complex plane with Hopf pairs highlighted.

    Shows all eigenvalues (discrete) and highlights the dominant Hopf pair
    with a distinctive marker.  Saves hopf_eigenvalues.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eigvals = spec_result.get("eigenvalues")
    eigvals_ct = spec_result.get("eigenvalues_ct")
    hopf_info = spec_result.get("hopf_dominant", {})

    if eigvals is None or eigvals_ct is None:
        return

    im_ct = np.asarray(eigvals_ct).imag
    hopf_mask = np.abs(im_ct) > _HOPF_IM_THRESH
    rho = spec_result.get("spectral_radius", float(np.abs(eigvals).max()))

    fig, ax = plt.subplots(figsize=(7, 6))

    # All eigenvalues (background)
    re_d = eigvals.real
    im_d = eigvals.imag
    ax.scatter(re_d[~hopf_mask], im_d[~hopf_mask], c="steelblue",
               s=16, alpha=0.6, label="Non-oscillatory modes", zorder=2)

    # Hopf modes
    if hopf_mask.any():
        ax.scatter(re_d[hopf_mask], im_d[hopf_mask], c="tomato",
                   s=28, alpha=0.9, marker="D",
                   label=f"Hopf pairs (n={hopf_mask.sum() // 2})", zorder=3)

    # Highlight dominant Hopf pair
    dom_idx = hopf_info.get("hopf_pair_index")
    if dom_idx is not None and dom_idx < len(eigvals):
        ax.scatter(re_d[dom_idx], im_d[dom_idx], c="gold", s=80,
                   edgecolors="black", lw=1.2, marker="*", zorder=5,
                   label=f"Dominant Hopf  f={hopf_info.get('hopf_frequency', 0):.4f} 1/step")

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4,
            label="|λ|=1 (neutral)")
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlabel("Re(λ)  [discrete time]")
    ax.set_ylabel("Im(λ)  [discrete time]")
    n_hopf = hopf_info.get("n_hopf_pairs", 0)
    dom_freq = hopf_info.get("hopf_frequency")
    freq_str = f"{dom_freq:.4f} 1/step" if dom_freq is not None else "none"
    ax.set_title(
        f"DMD Eigenvalues — Hopf Mode Identification\n"
        f"ρ={rho:.4f}, n_Hopf_pairs={n_hopf}, dominant_freq={freq_str}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "hopf_eigenvalues.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存 Hopf 特征值图: %s", output_dir / "hopf_eigenvalues.png")


# ─────────────────────────────────────────────────────────────────────────────
# Supplement 2: DMD Modal Energy
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_dmd_mode_energy(spec_result: Dict, output_dir: Path) -> None:
    """
    Supplement 2 plot: DMD mode energy spectrum.

    energy_i = ||mode_i||² = sum of squared eigenvector components.
    Shows top-10 modes ranked by energy.  Saves dmd_mode_energy.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    mode_energy = spec_result.get("mode_energy")
    eigvals = spec_result.get("eigenvalues")
    if mode_energy is None or eigvals is None:
        return

    mode_energy = np.asarray(mode_energy)
    # Eigenvalues are already sorted by |λ| descending in analyze_jacobian_spectrum.
    N = len(mode_energy)
    n_show = min(10, N)
    top_idx = np.arange(n_show)          # already sorted descending
    top_energy = mode_energy[top_idx]
    top_mags = np.abs(eigvals[top_idx])

    total_energy = mode_energy.sum()
    top_frac = top_energy.sum() / max(total_energy, 1e-30) * 100

    # Determine Hopf membership for colouring.
    # eigenvalues_ct may be absent (e.g., dt=None path); fall back to all-False.
    eigvals_ct_raw = spec_result.get("eigenvalues_ct")
    if eigvals_ct_raw is not None:
        im_ct = np.asarray(eigvals_ct_raw).imag
        is_hopf = np.abs(im_ct[top_idx]) > _HOPF_IM_THRESH
    else:
        is_hopf = np.zeros(n_show, dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart of top-10 mode energies
    ax = axes[0]
    colors = ["tomato" if h else "steelblue" for h in is_hopf]
    bars = ax.bar(np.arange(n_show) + 1, top_energy, color=colors,
                  edgecolor="k", lw=0.5, alpha=0.85)
    ax.set_xlabel("Mode Rank  (sorted by |λ| descending)")
    ax.set_ylabel("Mode Energy  ||mode||²")
    ax.set_title(
        f"DMD Mode Energy Spectrum (top {n_show})\n"
        f"Top-{n_show} capture {top_frac:.1f}% of total energy\n"
        f"[red = Hopf (oscillatory),  blue = non-oscillatory]"
    )
    ax.set_xticks(np.arange(n_show) + 1)
    ax.set_xticklabels([f"M{i+1}" for i in range(n_show)], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: |λ| vs mode energy scatter
    ax2 = axes[1]
    sc = ax2.scatter(top_mags, top_energy,
                     c=["tomato" if h else "steelblue" for h in is_hopf],
                     s=60, edgecolors="k", lw=0.5, alpha=0.9, zorder=3)
    for i in range(n_show):
        ax2.annotate(f"M{i+1}", (top_mags[i], top_energy[i]),
                     fontsize=7, ha="left", xytext=(3, 2),
                     textcoords="offset points")
    ax2.set_xlabel("|λ|  (spectral magnitude)")
    ax2.set_ylabel("Mode Energy  ||mode||²")
    ax2.set_title("Mode Energy vs Spectral Magnitude\n"
                  "Red = Hopf pair,  Blue = decaying/neutral mode")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "dmd_mode_energy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存 DMD 模态能量图: %s", output_dir / "dmd_mode_energy.png")
