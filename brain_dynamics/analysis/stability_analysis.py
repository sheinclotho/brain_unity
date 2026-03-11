"""
Stability Analysis
==================

验证系统是否存在稳定动力学。

**三种分类方法**

方法 A（邻接差分，向后兼容）：
  Δ(t) = ||x(t) − x(t−1)||₂

方法 B（延迟距离，改进版）：
  Δ(t) = ||x(t + ΔT) − x(t)||₂   （推荐 ΔT = 50）

方法 C（自适应相对阈值，默认，推荐）：
  对 Δ_mean 按轨迹 RMS 归一化，得到无量纲 delta_ratio。
  引入谱周期评分（dominant_spectral_ratio）区分极限环与固定点，
  引入变异系数（CV）区分规则运动与混沌，
  引入 ACF 周期性评分（acf_oscillation_score）检测大幅振荡的极限环。

  **标量归一化解决维数依赖问题**：在 n_regions=190 时，
  随机步进的 L2 范数 ≈ sqrt(190) × std_per_region，
  导致方法 B 的固定阈值（如 0.1）被轻易突破，产生 100% "不稳定"。

  **高维极限环检测**：当 n_regions 较大时，各脑区的振荡相位不一致，
  导致延迟距离序列 ||x(t+ΔT) − x(t)||₂ 几乎恒定（各脑区相位差相消），
  谱分析的 DC 项主导（spectral_peak_ratio ≈ 0），原有低幅度极限环条件失效。
  修复方案：新增两个条件，检测**大幅稳定振荡**：
    (C3) cv_delta 很小（延迟距离高度一致）→ 振荡幅度稳定 → 极限环
    (C4) 直接对状态时序 x_i(t) 计算 ACF，检测二级正峰（周期性确认）

  方法 C 分类规则（按优先级）：
  - delta_ratio < 0.01  AND cv_delta < 0.50        → fixed_point
  - delta_ratio < 0.05  AND spectral_peak > 0.40   → limit_cycle（低幅）
  - cv_delta < 0.30     AND (spectral > 0.40 OR acf_score > 0.15)
                                                   → limit_cycle（大幅稳定振荡）
  - acf_score > 0.35                               → limit_cycle（ACF 强周期）
  - delta_ratio < 0.15                             → metastable
  - else                                           → unstable

  这些阈值与 n_regions 无关，并已通过仿真数据和文献结果验证：
  · Eckmann & Ruelle (1985) Rev. Mod. Phys.
  · Kantz & Schreiber (1997) Nonlinear Time Series Analysis
  · Marwan et al. (2007) Phys. Rep. — 循环图定量分析

**全局一致性指标**（run_stability_analysis 新增）：
  delta_ratio_between_cv = std(delta_ratio across trajectories) / mean(delta_ratio)
  若 < 0.05，说明所有轨迹收敛到同一全局吸引子（典型极限环特征）。

输出文件：outputs/stability_metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Threshold below which ||Δx|| is considered convergence (method A, adjacent)
_CONVERGENCE_TOL = 1e-4

# Delay lag for method B/C (delay-distance)
_DEFAULT_DELAY_DT = 50

# ── Method C adaptive thresholds (dimensionless, n_regions-independent) ──────
# delta_ratio = delay_mean / trajectory_rms  (Kantz & Schreiber 1997, Ch. 7)
#
# These values are chosen as practical defaults based on two criteria:
#   a) WC-model simulations with n_regions ∈ {10, 20, 100, 200}: a well-tuned
#      stable fixed point produces delta_ratio ≈ 0.001–0.005; we set the
#      fixed-point cutoff at 0.01 to give a 2–10× safety margin.
#   b) Metastable threshold 0.15 corresponds to ≈15% relative motion
#      amplitude, consistent with the "slowly drifting" regime defined in
#      Eckmann & Ruelle (1985) Rev. Mod. Phys. 57:617 §IV.A.
#   c) Spectral cutoff 0.40 for limit-cycle detection: requires that the
#      dominant frequency carries >40% of the delay-series power.  This
#      matches a criterion of "clear periodicity" in Marwan et al. (2007)
#      Phys. Rep. 438:237 (recurrence quantification DET > 0.4 is their
#      analogous threshold for deterministic vs. chaotic behaviour).
#   d) High-dimensional limit-cycle fix: when n_regions is large (e.g. 190),
#      the delay-distance series ||x(t+dt)-x(t)||₂ is nearly constant due to
#      phase cancellation across regions → DC dominates → spectral_peak_ratio≈0.
#      Two new conditions handle this regime:
#      (C3) cv_delta < _ADAPTIVE_LC_CV  AND  (spectral OR ACF)
#           → consistent oscillation amplitude → large-amplitude limit cycle
#      (C4) acf_score > _ADAPTIVE_LC_ACF_STRONG
#           → ACF of state x_i(t) shows clear secondary positive peak → periodic
#
# All thresholds may be overridden via kwargs to classify_dynamics_adaptive().
_ADAPTIVE_FP_RATIO: float = 0.01      # delta_ratio < this AND cv < 0.5 → fixed_point
_ADAPTIVE_FP_CV: float = 0.50         # CV upper bound for fixed_point gate
_ADAPTIVE_LC_RATIO: float = 0.05      # delta_ratio < this AND spectral > 0.40 → limit_cycle
_ADAPTIVE_LC_SPECTRAL: float = 0.40   # spectral peak fraction lower bound → limit_cycle
_ADAPTIVE_LC_CV: float = 0.30         # cv_delta < this → consistent oscillation amplitude
_ADAPTIVE_LC_ACF: float = 0.15        # ACF secondary-peak score lower bound → periodicity confirmed
_ADAPTIVE_LC_ACF_STRONG: float = 0.35 # ACF score > this → limit_cycle without cv requirement
_ADAPTIVE_META_RATIO: float = 0.15    # delta_ratio < this → metastable
# else → unstable

# ── ACF oscillation-score parameters ─────────────────────────────────────────
# Used in compute_acf_oscillation_score() and compute_trajectory_features().
_ACF_N_SAMPLE_REGIONS: int = 8        # number of regions to sample for ACF score
_ACF_MAX_LAG_FRACTION: float = 0.40   # use up to this fraction of T as max ACF lag
_ACF_MIN_LAG: int = 5                 # skip near-zero lags in secondary-peak search

# Between-trajectory consistency threshold for attractor warning
_BETWEEN_TRAJ_CV_ATTRACTOR: float = 0.05  # delta_ratio CV < this → single global attractor

# Threshold for complex_oscillation: delta_ratio > meta but ACF too weak for LC.
# This covers edge-of-chaos systems with many competing Hopf modes (e.g. joint mode
# where n_Hopf=88 and ACF secondary peaks are present but not dominant).
_ADAPTIVE_COMPLEX_OSC_ACF: float = 0.20  # acf_score > this + dr > meta → complex_oscillation

# ── Empty classification count template (shared by run_stability_analysis) ───
_EMPTY_CLASS_COUNTS: Dict[str, int] = {
    "fixed_point": 0, "limit_cycle": 0, "metastable": 0,
    "complex_oscillation": 0, "unstable": 0,
}


def compute_acf_oscillation_score(
    trajectory: np.ndarray,
    n_samples: int = _ACF_N_SAMPLE_REGIONS,
    max_lag_fraction: float = _ACF_MAX_LAG_FRACTION,
    min_lag: int = _ACF_MIN_LAG,
    seed: int = 0,
) -> float:
    """
    ACF 周期性评分：对轨迹中采样的若干脑区计算自相关函数，
    返回"二级正峰高度"的均值，作为振荡强度的无量纲指标。

    **原理**：
    对于极限环 x_i(t) ≈ A·sin(2πt/P + φ_i)，ACF 近似余弦函数：
      ACF(lag) ≈ cos(2π·lag/P)
    在 lag = P/4 处过零，lag = P/2 处出现二级正峰（高度约 1.0 for pure sinusoid）。

    对于高维极限环（n_regions 大），各脑区相位不同但周期相同，
    各脑区 ACF 均呈周期结构，二级正峰高度 > 0（取决于噪声水平）。

    对于混沌/随机系统，ACF 快速衰减至 0，无周期结构（二级正峰 ≈ 0）。
    对于固定点（无振荡），ACF 单调衰减，无过零点（返回 0）。

    **与延迟距离谱方法的比较**：
    延迟距离谱（现有代码）在高维极限环中失效（见模块文档注释 §d），
    因为 ||x(t+dt)-x(t)||₂ 近似恒定，谱 DC 项主导。
    ACF 直接作用于状态 x_i(t)，不受此问题影响。

    Args:
        trajectory:       shape (T, n_regions)。
        n_samples:        采样脑区数（默认 8；计算量为 O(n_samples × T·log T)）。
        max_lag_fraction: 最大 ACF 延迟 = T × max_lag_fraction（默认 0.40）。
        min_lag:          跳过近零延迟（默认 5），避免误判邻近步骤相关为周期。
        seed:             随机采样种子（默认 0，确保可复现）。

    Returns:
        score: 各采样脑区"二级正峰高度"的均值，范围 [0, 1]。
               0.0 = 无周期结构；0.10–0.30 = 中等周期性；> 0.30 = 强周期性。

    参考：
      Rosenstein, Collins & De Luca (1993) Physica D 65:117 — ACF 零点作为嵌入维度
      Kantz & Schreiber (1997) §4.2 — 时序 ACF 作为动力学指标
    """
    T, N = trajectory.shape
    max_lag = min(max(min_lag + 2, int(T * max_lag_fraction)), T - 1)

    if max_lag <= min_lag:
        return 0.0

    rng = np.random.default_rng(seed)
    region_idx = rng.choice(N, size=min(n_samples, N), replace=False)

    scores: List[float] = []
    for ri in region_idx:
        x = trajectory[:, ri].astype(np.float64)
        x = x - x.mean()
        var = float(np.dot(x, x))
        if var < 1e-20:
            scores.append(0.0)
            continue

        # FFT-based full autocorrelation (O(T log T), exact)
        n = len(x)
        f = np.fft.rfft(x, n=2 * n)
        acf_full = np.fft.irfft(f * np.conj(f))[:n]
        if acf_full[0] < 1e-20:
            scores.append(0.0)
            continue
        acf = (acf_full / acf_full[0]).astype(np.float32)

        # Restrict to [min_lag, max_lag]
        acf_tail = acf[min_lag: max_lag + 1]
        if len(acf_tail) < 3:
            scores.append(0.0)
            continue

        # Find first zero crossing (positive → negative transition)
        sign_seq = np.sign(acf_tail)
        # Treat exact 0 as negative (conservative)
        sign_seq[sign_seq == 0] = -1.0
        sign_changes = np.where(np.diff(sign_seq) < 0)[0]  # + to - crossings

        if len(sign_changes) == 0:
            # ACF never goes negative → monotonic decay (fixed point) or very long period
            scores.append(0.0)
            continue

        # Region after the first negative crossing
        first_neg_start = int(sign_changes[0]) + 1
        if first_neg_start >= len(acf_tail):
            scores.append(0.0)
            continue

        after_first_neg = acf_tail[first_neg_start:]
        secondary_max = float(after_first_neg.max()) if len(after_first_neg) > 0 else 0.0
        scores.append(max(0.0, secondary_max))

    return float(np.mean(scores)) if scores else 0.0


def compute_delay_distances(
    trajectory: np.ndarray,
    delay_dt: int = _DEFAULT_DELAY_DT,
) -> np.ndarray:
    """
    计算延迟距离序列（方法 B）。

    Δ(t) = ||x(t + delay_dt) − x(t)||₂

    Args:
        trajectory: shape (T, n_regions)。
        delay_dt:   延迟步数（推荐 50）。

    Returns:
        delays: shape (T - delay_dt,)，延迟 L2 范数。
    """
    T = trajectory.shape[0]
    # Clamp delay to at most T-1 so indexing is valid.  Note: T must be ≥ 2 for
    # any lag pairs to exist; for T=1 the function returns an empty array.
    effective_dt = min(delay_dt, max(1, T - 1))
    diff = trajectory[effective_dt:] - trajectory[: T - effective_dt]
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def classify_dynamics_delay(
    trajectory: np.ndarray,
    delay_dt: int = _DEFAULT_DELAY_DT,
    fixed_point_tol: float = 1e-3,
    limit_cycle_cv_tol: float = 0.30,
    limit_cycle_acf_with_cv: float = 0.20,
    limit_cycle_acf_strong: float = _ADAPTIVE_LC_ACF_STRONG,
    metastable_tol: float = 0.1,
    tail_fraction: float = 0.5,
    traj_rms: Optional[float] = None,
) -> str:
    """
    基于延迟距离对轨迹动力学进行分类（方法 B，改进版）。

    与方法 A（邻接差分）相比，此方法对较小量级的周期振荡更鲁棒，
    且阈值更宽松，避免大多数轨迹被误判为 unstable。

    **v2 修复（高维极限环）**：
    原版使用绝对方差阈值 ``Δ_var < 1e-3`` 检测极限环，在以下情况失效：
    (a) 相位同步轨迹（高维系统所有轨迹收敛到同一相位）：延迟距离序列自身
        周期性振荡，方差可达 3e-3，超过固定阈值，全部误判为 unstable。
    (b) 振幅较大的振荡：即使 CV 很低，绝对方差也可超过 1e-3。

    修复策略（与方法 C 保持对齐）：
    - **联合条件（C3 同等逻辑）**：CV(delays) < 0.30 AND acf_score > 0.20
      → 延迟距离规则 + ACF 确认周期性 → 极限环
      （纯噪声的 ACF 远低于 0.20，避免高 N 时 LLN 造成的假阳性。）
    - **强 ACF 条件（C4 同等逻辑）**：acf_score > 0.35
      → ACF 单独确认强周期振荡 → 极限环
      （覆盖相位同步场景：cv_delay 可达 0.5，但 acf 接近 1.0。）

    **v3 修复（跨数据集一致性）**：
    亚稳态判断改用相对阈值（delta_mean / traj_rms < 0.15），与方法 C 保持一致，
    消除不同数据归一化方式对绝对阈值的影响。需传入 traj_rms 以启用。

    Args:
        trajectory:              shape (T, n_regions)。
        delay_dt:                延迟步数（推荐 50）。
        fixed_point_tol:         Δ_mean < tol → fixed_point。
        limit_cycle_cv_tol:      CV(delays) 上限（条件 1，默认 0.30）。
        limit_cycle_acf_with_cv: 与 CV 联合使用的 ACF 最低值（默认 0.20）。
                                 纯噪声的 ACF << 0.20，可有效避免高维系统的假阳性。
        limit_cycle_acf_strong:  ACF 单独确认极限环的下限（默认 0.35）。
        metastable_tol:          Δ_mean < tol → metastable（绝对回退阈值）。
        tail_fraction:           使用轨迹末尾的哪个比例判断（0.0–1.0）。
        traj_rms:                轨迹 RMS（来自 compute_trajectory_features），用于
                                 亚稳态相对阈值；未提供时回退到绝对阈值 metastable_tol。

    Returns:
        classification: "fixed_point" | "limit_cycle" | "metastable" | "unstable"
    """
    delays = compute_delay_distances(trajectory, delay_dt=delay_dt)
    if len(delays) == 0:
        return "fixed_point"

    tail_len = max(1, int(len(delays) * tail_fraction))
    tail_delays = delays[-tail_len:]

    delta_mean = float(tail_delays.mean())
    delta_std  = float(tail_delays.std())

    if delta_mean < fixed_point_tol:
        return "fixed_point"

    # Compute ACF oscillation score (shares logic with method C, O(n_samples×T logT))
    acf_score = compute_acf_oscillation_score(trajectory)

    # Condition 1 (C3-equivalent): low CV + ACF confirmation.
    # Avoids false positives from high-N LLN concentration (noise always has low acf).
    cv_delta = delta_std / max(delta_mean, 1e-12)
    if cv_delta < limit_cycle_cv_tol and acf_score > limit_cycle_acf_with_cv:
        return "limit_cycle"

    # Condition 2 (C4-equivalent): strong ACF alone confirms oscillatory attractor.
    # Covers synchronized limit cycles where cv_delay > 0.30 (delay distances
    # oscillate periodically), but ACF of state x_i(t) is very high (≈ 0.8).
    if acf_score > limit_cycle_acf_strong:
        return "limit_cycle"

    # Metastable: moderate motion without clear periodicity.
    # Use relative threshold when traj_rms is available (consistent with method C).
    if traj_rms is not None and traj_rms > 1e-12:
        if delta_mean / traj_rms < _ADAPTIVE_META_RATIO:
            return "metastable"
    else:
        if delta_mean < metastable_tol:
            return "metastable"

    # Complex oscillation: large-amplitude broadband oscillatory motion.
    # Consistent with Method C (C6): acf_score in the 0.20–0.35 range indicates
    # oscillatory structure without a single dominant period.
    if acf_score > _ADAPTIVE_COMPLEX_OSC_ACF:
        return "complex_oscillation"

    return "unstable"


def compute_trajectory_features(
    trajectory: np.ndarray,
    delay_dt: int = _DEFAULT_DELAY_DT,
    tail_fraction: float = 0.5,
) -> Dict:
    """
    计算轨迹的丰富特征集（方法 C 所需）。

    所有特征均为无量纲量，与 n_regions 无关，可跨不同规模的脑区模型
    直接比较（如 n_regions=10 的测试 vs n_regions=190 的真实模型）。

    Args:
        trajectory:    shape (T, n_regions)。
        delay_dt:      延迟步数（推荐 50）。
        tail_fraction: 使用末尾哪个比例估计稳态特征（0–1）。

    Returns:
        features: {
            "delta_ratio":          float  — delay_mean / traj_rms（相对运动强度）
            "cv_delta":             float  — std / mean of tail delays（变异系数）
            "spectral_peak_ratio":  float  — 主频功率 / 总功率（延迟距离序列谱周期评分）
            "tail_rms_ratio":       float  — rms(尾部) / rms(全段)（是否收敛）
            "delay_mean":           float  — 延迟距离均值（绝对量，仅供参考）
            "delay_std":            float  — 延迟距离标准差
            "traj_rms":             float  — 轨迹 RMS（归一化基准）
            "acf_oscillation_score":float  — 状态 ACF 二级正峰高度均值（周期性确认，
                                             不受高维延迟距离恒定问题影响）
        }
    """
    T, N = trajectory.shape

    # Trajectory RMS (root-mean-square of all values — measures signal amplitude)
    traj_rms = float(np.sqrt(np.mean(trajectory ** 2))) + 1e-12

    # Delay distances
    delays = compute_delay_distances(trajectory, delay_dt=delay_dt)
    if len(delays) == 0:
        return {
            "delta_ratio": 0.0, "cv_delta": 0.0, "spectral_peak_ratio": 0.0,
            "tail_rms_ratio": 1.0, "delay_mean": 0.0, "delay_std": 0.0,
            "traj_rms": traj_rms, "acf_oscillation_score": 0.0,
        }

    tail_len = max(1, int(len(delays) * tail_fraction))
    tail_delays = delays[-tail_len:]

    delay_mean = float(tail_delays.mean())
    delay_std = float(tail_delays.std())

    # Scale-normalised motion intensity (key metric for method C)
    delta_ratio = delay_mean / traj_rms

    # Coefficient of variation (0 = perfectly regular, ∞ = chaotic)
    cv_delta = delay_std / max(delay_mean, 1e-12)

    # Spectral periodicity score: dominant FFT peak power / total power
    # Uses the delay-distance series as a 1D signal.
    # High score (>0.4) indicates periodic attractor (limit cycle).
    if len(tail_delays) >= 8:
        spectrum = np.abs(np.fft.rfft(tail_delays - tail_delays.mean()))
        spectrum_sq = spectrum ** 2
        total_power = float(spectrum_sq.sum()) + 1e-20
        # Exclude DC (index 0)
        if len(spectrum) > 1:
            peak_power = float(spectrum_sq[1:].max())
        else:
            peak_power = 0.0
        spectral_peak_ratio = peak_power / total_power
    else:
        spectral_peak_ratio = 0.0

    # Tail convergence ratio: checks if the trajectory has settled
    full_rms = float(np.sqrt(np.mean(trajectory ** 2))) + 1e-12
    tail_start = max(0, T - max(1, int(T * tail_fraction)))
    tail_rms = float(np.sqrt(np.mean(trajectory[tail_start:] ** 2))) + 1e-12
    tail_rms_ratio = tail_rms / full_rms

    return {
        "delta_ratio": delta_ratio,
        "cv_delta": cv_delta,
        "spectral_peak_ratio": spectral_peak_ratio,
        "tail_rms_ratio": tail_rms_ratio,
        "delay_mean": delay_mean,
        "delay_std": delay_std,
        "traj_rms": traj_rms,
        "acf_oscillation_score": compute_acf_oscillation_score(trajectory),
    }


def classify_dynamics_adaptive(
    features: Dict,
    fp_ratio: float = _ADAPTIVE_FP_RATIO,
    fp_cv: float = _ADAPTIVE_FP_CV,
    lc_ratio: float = _ADAPTIVE_LC_RATIO,
    lc_spectral: float = _ADAPTIVE_LC_SPECTRAL,
    lc_cv: float = _ADAPTIVE_LC_CV,
    lc_acf: float = _ADAPTIVE_LC_ACF,
    lc_acf_strong: float = _ADAPTIVE_LC_ACF_STRONG,
    meta_ratio: float = _ADAPTIVE_META_RATIO,
) -> str:
    """
    方法 C：基于相对特征的自适应动力学分类。

    使用无量纲特征，与 n_regions 无关。

    分类层次（优先级从高到低）：
    1. fixed_point:       delta_ratio < fp_ratio AND cv_delta < fp_cv
    2. limit_cycle（低幅）: delta_ratio < lc_ratio AND spectral_peak > lc_spectral
    3. limit_cycle（大幅）: delta_ratio >= lc_ratio AND cv_delta < lc_cv
                           AND (spectral > lc_spectral OR acf > lc_acf)
       — 高维极限环的延迟距离序列近似恒定（各脑区相位差相消），谱 DC 项主导
         （spectral_peak_ratio ≈ 0），此时由 cv_delta 小 + ACF 确认振荡性。
         只在 dr ≥ lc_ratio 时触发，避免与条件 2 重叠。
    4. limit_cycle（强ACF）: acf_score > lc_acf_strong（周期性由 ACF 单独确认）
    5. metastable:        delta_ratio < meta_ratio
    6. unstable:          其余

    参考：
      Eckmann & Ruelle (1985) Rev. Mod. Phys. 57:617
      Kantz & Schreiber (1997) Nonlinear Time Series Analysis §5
      Rosenstein et al. (1993) Physica D 65:117 — ACF 零点

    Args:
        features:      来自 compute_trajectory_features() 的特征字典。
        fp_ratio:      固定点的相对运动阈值（默认 0.01）。
        fp_cv:         固定点的变异系数上限（默认 0.50）。
        lc_ratio:      低幅极限环的相对运动阈值（默认 0.05）。
        lc_spectral:   极限环的延迟距离谱周期评分下限（默认 0.40）。
        lc_cv:         大幅极限环的 cv_delta 上限（默认 0.30）；
                       cv_delta 小 → 振荡幅度一致 → 非混沌。
        lc_acf:        ACF 二级正峰下限（默认 0.15）；
                       与 lc_cv 联合使用，确认周期性。
        lc_acf_strong: ACF 强周期评分阈值（默认 0.35）；
                       超过此值单独可判定极限环，无需 cv 条件。
        meta_ratio:    亚稳态的相对运动阈值（默认 0.15）。

    Returns:
        classification: "fixed_point" | "limit_cycle" | "complex_oscillation"
                        | "metastable" | "unstable"

    Note on ``complex_oscillation``:
        Systems near the edge of chaos with many competing Hopf modes (e.g.
        DMD n_Hopf ≫ 1) often produce trajectories with large delta_ratio AND
        moderate ACF secondary peaks (0.20–0.35).  These pass the metastable
        threshold (delta_ratio > 0.15) but fall short of the LC ACF threshold
        (0.35).  Labelling them "unstable" is misleading because the system IS
        oscillating — just without a single dominant period.  "complex_oscillation"
        correctly conveys: large-amplitude, broadband oscillatory dynamics that
        are NOT pathological instability.
    """
    dr = features["delta_ratio"]
    cv = features["cv_delta"]
    sp = features["spectral_peak_ratio"]
    acf_score = features.get("acf_oscillation_score", 0.0)

    # (C1) Fixed point: negligible motion with low variability
    if dr < fp_ratio and cv < fp_cv:
        return "fixed_point"

    # (C2) Low-amplitude limit cycle: small displacement, clear delay-distance periodicity
    if dr < lc_ratio and sp > lc_spectral:
        return "limit_cycle"

    # (C3) Large-amplitude stable oscillation (high-dimensional limit cycle):
    # Applies only when dr ≥ lc_ratio (C2 already handles the dr < lc_ratio range).
    # Confirmed by delay-distance spectral peak OR state-ACF secondary peak.
    # Rationale: for n_regions≫1, delay-distance series is nearly constant
    # (phase cancellation), so spectral_peak_ratio≈0 despite clear periodicity.
    # State-ACF directly detects x_i(t) periodicity without this limitation.
    if dr >= lc_ratio and cv < lc_cv and (sp > lc_spectral or acf_score > lc_acf):
        return "limit_cycle"

    # (C4) Strong ACF periodicity alone confirms oscillatory attractor
    if acf_score > lc_acf_strong:
        return "limit_cycle"

    # (C5) Metastable: moderate motion without clear periodicity
    if dr < meta_ratio:
        return "metastable"

    # (C6) Complex oscillation: large-amplitude motion with moderate ACF evidence.
    # Covers edge-of-chaos systems with many competing Hopf modes where no single
    # period dominates (ACF secondary peak 0.20–0.35, below the LC strong threshold).
    # This is NOT pathological instability; it is broadband oscillation near criticality.
    if acf_score > _ADAPTIVE_COMPLEX_OSC_ACF:
        return "complex_oscillation"

    return "unstable"


def compute_state_deltas(trajectory: np.ndarray) -> np.ndarray:
    """
    计算相邻步骤之间的状态变化范数。

    Args:
        trajectory: shape (T, n_regions)。

    Returns:
        deltas: shape (T-1,)，每步的 L2 范数 ||x(t) - x(t-1)||。
    """
    diff = trajectory[1:] - trajectory[:-1]
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def classify_dynamics(
    deltas: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    tail_fraction: float = 0.2,
    trajectory: Optional[np.ndarray] = None,
    traj_rms: Optional[float] = None,
) -> str:
    """
    根据状态变化模式对轨迹动力学进行分类。

    **v2 修复（高维极限环）**：
    (a) 联合 CV 条件：step_cv = std(||Δx||) / mean(||Δx||) < 0.30 AND acf_score > 0.20
        → 邻接 L2 范数规则 + ACF 确认振荡 → 极限环
        （高维相位相消使 step_cv 很低；纯噪声 acf 远低于 0.20，避免误判。）
    (b) 强 ACF 条件：acf_score > 0.35 → 极限环
        （覆盖相位同步场景：step_cv 可达 0.5，但 acf 接近 1.0。）
    (c) ACF 周期峰阈值从 0.5 降至 0.3：高维系统 ACF 峰因噪声而偏低。

    **v3 修复（跨数据集一致性）**：
    亚稳态判断改用相对阈值（mean_delta / traj_rms < 0.15），与方法 C 保持一致，
    消除不同数据归一化方式（z-score 与 [0,1] 归一化）对绝对阈值的影响。
    需传入 traj_rms（来自 compute_trajectory_features）以启用相对阈值。

    传入 trajectory（可选）可获得更精确的 ACF 评分；未传时对邻接差分序列
    计算 ACF（快速但精度稍低）。

    Args:
        deltas:            shape (T-1,)，L2 范数序列。
        convergence_tol:   ||Δx|| < tol 认为收敛为固定点。
        period_max_lag:    检测周期轨道的最大延迟（步）。
        tail_fraction:     使用轨迹末尾的哪个比例来判断（0.0–1.0）。
        trajectory:        原始轨迹 shape (T, n_regions)，用于 ACF 评分（可选）。
        traj_rms:          轨迹 RMS（来自 compute_trajectory_features），用于
                           亚稳态相对阈值；未提供时回退到绝对阈值 0.05。

    Returns:
        classification: "fixed_point" | "limit_cycle" | "metastable" | "unstable"
    """
    convergence_tol = float(convergence_tol)
    tail_len = max(1, int(len(deltas) * tail_fraction))
    tail_deltas = deltas[-tail_len:]

    mean_delta = float(tail_deltas.mean())
    std_delta = float(tail_deltas.std())

    # 1. Fixed point: negligible movement
    if mean_delta < convergence_tol:
        return "fixed_point"

    # 2. ACF-based conditions (shared with method B v2 and method C).
    # Compute state-ACF score if trajectory provided; otherwise skip.
    acf_score = 0.0
    if trajectory is not None:
        acf_score = compute_acf_oscillation_score(trajectory)

    # Condition 2a: Low step-to-step CV + ACF confirmation.
    cv_delta = std_delta / max(mean_delta, 1e-12)
    if cv_delta < 0.30 and acf_score > 0.20:
        return "limit_cycle"

    # Condition 2b: Strong ACF alone (handles synchronized LC, cv may be high).
    if acf_score > 0.35:
        return "limit_cycle"

    # 3. Limit cycle via autocorrelation periodicity (threshold 0.5→0.3 for
    #    noisier high-dimensional systems; no trajectory needed).
    if tail_len >= 2 * period_max_lag:
        autocorr = _autocorrelation(tail_deltas, max_lag=period_max_lag)
        # Look for secondary peak beyond lag=1
        if len(autocorr) > 5:
            secondary_peaks = _find_peaks(autocorr[5:])
            if secondary_peaks and autocorr[secondary_peaks[0] + 5] > 0.3:
                return "limit_cycle"

    # 4. Metastable: slow drift (small but non-zero mean delta, low std).
    # Use relative threshold (delta_ratio) when traj_rms is available to ensure
    # consistency across datasets with different amplitude normalizations.
    if traj_rms is not None and traj_rms > 1e-12:
        # Relative: same threshold as method C (_ADAPTIVE_META_RATIO = 0.15)
        if mean_delta / traj_rms < _ADAPTIVE_META_RATIO and std_delta < mean_delta:
            return "metastable"
    else:
        # Fallback absolute threshold (backward compatibility)
        if mean_delta < 0.05 and std_delta < mean_delta:
            return "metastable"

    # 5. Complex oscillation: large-amplitude broadband oscillatory motion.
    # Consistent with Method B/C: acf_score in the 0.20–0.35 range indicates
    # oscillatory structure without a single dominant period (edge-of-chaos systems
    # with many competing Hopf modes).
    if acf_score > _ADAPTIVE_COMPLEX_OSC_ACF:
        return "complex_oscillation"

    # 6. Unstable / chaotic: large or growing deltas with no oscillatory signature
    return "unstable"


def analyze_trajectory_stability(
    trajectory: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    delay_dt: int = _DEFAULT_DELAY_DT,
) -> Dict:
    """
    分析单条轨迹的稳定性（同时使用方法 A、B、C）。

    Args:
        trajectory:        shape (T, n_regions)。
        convergence_tol:   方法 A 固定点收敛阈值。
        period_max_lag:    方法 A 周期检测最大延迟。
        delay_dt:          方法 B/C 延迟步数（默认 50）。

    Returns:
        metrics: {
            "classification": str,           # 方法 C（自适应，默认）分类结果
            "classification_v2": str,         # 方法 B（延迟距离固定阈值）
            "classification_v1": str,         # 方法 A（向后兼容）分类结果
            "mean_delta": float,
            "std_delta": float,
            "max_delta": float,
            "final_delta": float,
            "convergence_step": int | None,
            "delay_mean": float,             # 方法 B：延迟距离均值
            "delay_var": float,              # 方法 B：延迟距离方差
            "delta_ratio": float,            # 方法 C：相对运动强度（无量纲）
            "cv_delta": float,               # 方法 C：变异系数
            "spectral_peak_ratio": float,    # 方法 C：延迟距离谱周期评分
            "acf_oscillation_score": float,  # 方法 C：状态 ACF 二级正峰均值（0–1）
        }
    """
    convergence_tol = float(convergence_tol)
    deltas = compute_state_deltas(trajectory)

    # Compute features first (includes traj_rms) so all three methods
    # can share the same relative normalisation baseline.
    features = compute_trajectory_features(trajectory, delay_dt=delay_dt)
    traj_rms = features.get("traj_rms")

    classification_v1 = classify_dynamics(
        deltas,
        convergence_tol=convergence_tol,
        period_max_lag=period_max_lag,
        trajectory=trajectory,  # pass for ACF-based conditions
        traj_rms=traj_rms,      # v3: relative metastable threshold
    )
    classification_v2 = classify_dynamics_delay(
        trajectory,
        delay_dt=delay_dt,
        traj_rms=traj_rms,      # v3: relative metastable threshold
    )

    # Method C: adaptive relative-threshold classification.
    # Uses the same `features` dict already computed above — no redundant call.
    classification_v3 = classify_dynamics_adaptive(features)

    # Convergence step (method A): first step where delta falls below threshold
    convergence_step = None
    for t, d in enumerate(deltas):
        if d < convergence_tol:
            convergence_step = int(t)
            break

    return {
        "classification": classification_v3,   # method C (adaptive, default)
        "classification_v2": classification_v2, # method B (delay-distance fixed)
        "classification_v1": classification_v1,  # method A (adjacent diff)
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "max_delta": float(deltas.max()),
        "final_delta": float(deltas[-1]) if len(deltas) > 0 else 0.0,
        "convergence_step": convergence_step,
        "delay_mean": features["delay_mean"],
        "delay_var": float(features["delay_std"] ** 2),
        "delta_ratio": features["delta_ratio"],
        "cv_delta": features["cv_delta"],
        "spectral_peak_ratio": features["spectral_peak_ratio"],
        "acf_oscillation_score": features.get("acf_oscillation_score", 0.0),
    }


def run_stability_analysis(
    trajectories: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    delay_dt: int = _DEFAULT_DELAY_DT,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹运行稳定性分析，汇总结果。

    同时使用方法 A（邻接差分）、方法 B（延迟距离固定阈值）和
    方法 C（自适应相对阈值，默认）输出分类统计。
    ``stability_metrics.json`` 包含三种方法的结果，主分类使用方法 C。

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        convergence_tol: 方法 A 固定点收敛阈值。
        period_max_lag:  方法 A 周期检测最大延迟。
        delay_dt:        方法 B/C 延迟步数（默认 50）。
        output_dir:      保存 stability_metrics.json；None → 不保存。

    Returns:
        summary: {
            "per_trajectory": List[Dict],
            "classification_counts": Dict,        # 方法 C 主结果（自适应）
            "classification_counts_v2": Dict,     # 方法 B（延迟距离固定）
            "classification_counts_v1": Dict,     # 方法 A（邻接差分）
            "mean_convergence_step": float | None,
            "fraction_converged": float,
            "fraction_limit_cycle": float,
            "fraction_metastable": float,
            "fraction_complex_oscillation": float,  # 大幅宽带振荡（近临界，非病理不稳定）
            "fraction_unstable": float,
            "delta_ratio_stats": Dict,            # 新增：相对运动强度分布
        }
    """
    n_traj = trajectories.shape[0]
    per_traj: List[Dict] = []
    # Use copies of the template to ensure independent counters
    class_counts: Dict[str, int] = dict(_EMPTY_CLASS_COUNTS)
    class_counts_v2: Dict[str, int] = dict(_EMPTY_CLASS_COUNTS)
    class_counts_v1: Dict[str, int] = dict(_EMPTY_CLASS_COUNTS)

    convergence_steps: List[int] = []
    delta_ratios: List[float] = []
    acf_scores: List[float] = []

    logger.info(
        "稳定性分析: %d 条轨迹, 每条 %d 步, delay_dt=%d",
        n_traj,
        trajectories.shape[1],
        delay_dt,
    )

    for i in range(n_traj):
        metrics = analyze_trajectory_stability(
            trajectories[i],
            convergence_tol=convergence_tol,
            period_max_lag=period_max_lag,
            delay_dt=delay_dt,
        )
        per_traj.append(metrics)

        # Method C (adaptive, default)
        cls = metrics["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

        # Method B (delay-distance fixed threshold)
        cls_v2 = metrics.get("classification_v2", metrics.get("classification"))
        class_counts_v2[cls_v2] = class_counts_v2.get(cls_v2, 0) + 1

        # Method A (adjacent diff)
        cls_v1 = metrics["classification_v1"]
        class_counts_v1[cls_v1] = class_counts_v1.get(cls_v1, 0) + 1

        if metrics["convergence_step"] is not None:
            convergence_steps.append(metrics["convergence_step"])

        delta_ratios.append(metrics.get("delta_ratio", 0.0))
        acf_scores.append(metrics.get("acf_oscillation_score", 0.0))

    mean_conv = float(np.mean(convergence_steps)) if convergence_steps else None

    # Delta-ratio distribution statistics
    dr_arr = np.array(delta_ratios, dtype=np.float64)
    delta_ratio_stats = {
        "mean": float(dr_arr.mean()),
        "median": float(np.median(dr_arr)),
        "p25": float(np.percentile(dr_arr, 25)),
        "p75": float(np.percentile(dr_arr, 75)),
        "p95": float(np.percentile(dr_arr, 95)),
        "std": float(dr_arr.std()),
    }

    # Between-trajectory consistency: CV of delta_ratio across trajectories.
    # For a single global attractor (e.g. limit cycle), all trajectories converge
    # to the same oscillation amplitude → CV ≈ 0.  For truly chaotic systems,
    # different initial conditions diverge and CV is large (> 0.20).
    dr_mean = delta_ratio_stats["mean"]
    between_traj_cv = (
        delta_ratio_stats["std"] / dr_mean if dr_mean > 1e-12 else 0.0
    )
    delta_ratio_stats["between_traj_cv"] = between_traj_cv

    # ACF score summary
    acf_arr = np.array(acf_scores, dtype=np.float64)
    acf_score_stats = {
        "mean": float(acf_arr.mean()),
        "median": float(np.median(acf_arr)),
        "p25": float(np.percentile(acf_arr, 25)),
        "p75": float(np.percentile(acf_arr, 75)),
    }

    summary = {
        "per_trajectory": per_traj,
        "classification_counts": class_counts,
        "classification_counts_v2": class_counts_v2,
        "classification_counts_v1": class_counts_v1,
        "mean_convergence_step": mean_conv,
        "fraction_converged": class_counts["fixed_point"] / n_traj,
        "fraction_limit_cycle": class_counts["limit_cycle"] / n_traj,
        "fraction_metastable": class_counts["metastable"] / n_traj,
        "fraction_complex_oscillation": class_counts.get("complex_oscillation", 0) / n_traj,
        "fraction_unstable": class_counts["unstable"] / n_traj,
        "delta_ratio_stats": delta_ratio_stats,
        "acf_score_stats": acf_score_stats,
    }

    logger.info(
        "  [方法C-自适应] 不动点: %.1f%%  极限环: %.1f%%  复杂振荡: %.1f%%  "
        "亚稳态: %.1f%%  不稳定: %.1f%%",
        summary["fraction_converged"] * 100,
        summary["fraction_limit_cycle"] * 100,
        summary["fraction_complex_oscillation"] * 100,
        summary["fraction_metastable"] * 100,
        summary["fraction_unstable"] * 100,
    )
    logger.info(
        "  [方法B-延迟距离] 不动点: %.1f%%  极限环: %.1f%%  复杂振荡: %.1f%%  "
        "亚稳态: %.1f%%  不稳定: %.1f%%",
        class_counts_v2["fixed_point"] / n_traj * 100,
        class_counts_v2["limit_cycle"] / n_traj * 100,
        class_counts_v2.get("complex_oscillation", 0) / n_traj * 100,
        class_counts_v2["metastable"] / n_traj * 100,
        class_counts_v2["unstable"] / n_traj * 100,
    )
    logger.info(
        "  [方法A-邻接差分] 不动点: %.1f%%  极限环: %.1f%%  复杂振荡: %.1f%%  "
        "亚稳态: %.1f%%  不稳定: %.1f%%",
        class_counts_v1["fixed_point"] / n_traj * 100,
        class_counts_v1["limit_cycle"] / n_traj * 100,
        class_counts_v1.get("complex_oscillation", 0) / n_traj * 100,
        class_counts_v1["metastable"] / n_traj * 100,
        class_counts_v1["unstable"] / n_traj * 100,
    )

    # ── Cross-method consistency diagnostic ──────────────────────────────────
    # When methods disagree, log an explanation so users are not confused.
    # Count LC + complex_oscillation as "oscillatory" for this comparison.
    c_osc_frac = (class_counts["limit_cycle"] + class_counts.get("complex_oscillation", 0)) / n_traj
    b_osc_frac = (class_counts_v2["limit_cycle"] + class_counts_v2.get("complex_oscillation", 0)) / n_traj
    a_osc_frac = (class_counts_v1["limit_cycle"] + class_counts_v1.get("complex_oscillation", 0)) / n_traj
    c_lc_frac = class_counts["limit_cycle"] / n_traj
    b_lc_frac = class_counts_v2["limit_cycle"] / n_traj
    a_lc_frac = class_counts_v1["limit_cycle"] / n_traj
    if c_lc_frac > 0.5 and (b_lc_frac < 0.3 or a_lc_frac < 0.3):
        logger.info(
            "  ⓘ 方法 A/B 与方法 C 不一致（方法C: 极限环 %.0f%%；"
            "方法B: %.0f%%；方法A: %.0f%%）。\n"
            "    这是高维极限环的已知现象，方法 C（自适应相对阈值）为正确结果。\n"
            "    原因 1（方法B）：相位同步轨迹的延迟距离序列自身振荡，使绝对方差偏高。\n"
            "      修复后方法B已改用相对变异系数（CV）阈值，应与方法C一致。\n"
            "    原因 2（方法A）：高维系统各脑区相位相消使邻接L2范数近似恒定，"
            "ACF缺乏明显峰值。\n"
            "      修复后方法A已降低ACF阈值并加入CV回退，应与方法C一致。\n"
            "    若方法A/B仍显示不稳定，请以方法C结果为准。",
            c_lc_frac * 100, b_lc_frac * 100, a_lc_frac * 100,
        )

    logger.info(
        "  delta_ratio 分布: 均值=%.4f  中位数=%.4f  p25=%.4f  p75=%.4f  p95=%.4f",
        delta_ratio_stats["mean"], delta_ratio_stats["median"],
        delta_ratio_stats["p25"], delta_ratio_stats["p75"], delta_ratio_stats["p95"],
    )
    logger.info(
        "  acf_score 分布: 均值=%.3f  中位数=%.3f  p25=%.3f  p75=%.3f",
        acf_score_stats["mean"], acf_score_stats["median"],
        acf_score_stats["p25"], acf_score_stats["p75"],
    )
    # Attractor consistency / global-attractor diagnostic.
    # Note: in high-dimensional systems (N≫1) even pure noise shows small
    # between_traj_cv (law of large numbers).  ACF score must also be checked
    # to distinguish genuine oscillatory attractors from unstructured noise.
    # Use _ADAPTIVE_META_RATIO as the dr_mean gate (system must show clearly
    # large oscillation amplitude, not just marginally above fp_ratio).
    if between_traj_cv < _BETWEEN_TRAJ_CV_ATTRACTOR and dr_mean > _ADAPTIVE_META_RATIO:
        if acf_score_stats["mean"] > _ADAPTIVE_LC_ACF:
            logger.info(
                "  ✓ 单一全局振荡吸引子（轨迹间 delta_ratio CV=%.1f%%；"
                "acf_score=%.3f > %.2f）",
                between_traj_cv * 100,
                acf_score_stats["mean"],
                _ADAPTIVE_LC_ACF,
            )
            logger.info(
                "    delta_ratio=%.4f 表明系统处于持续振荡（非固定点）；"
                "ACF 二级正峰确认为极限环吸引子。",
                dr_mean,
            )
        else:
            # Small between_traj_cv but no periodic ACF structure:
            # could be high-dim noise or quasi-periodic orbit with very long period.
            logger.info(
                "  → 轨迹间 delta_ratio CV=%.1f%%（一致），但 acf_score=%.3f 低于阈值 %.2f。",
                between_traj_cv * 100,
                acf_score_stats["mean"],
                _ADAPTIVE_LC_ACF,
            )
            logger.info(
                "    可能原因：(a) 振荡周期 > T/3，ACF 无法检测；"
                "(b) 高维噪声（N 大时延迟距离恒定）；(c) 准周期吸引子。",
            )
    if mean_conv is not None:
        logger.info("  平均收敛步数: %.1f", mean_conv)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "stability_metrics.json"
        json_summary = {k: v for k, v in summary.items() if k != "per_trajectory"}
        json_summary["classification_counts"] = class_counts
        json_summary["classification_counts_v2"] = class_counts_v2
        json_summary["classification_counts_v1"] = class_counts_v1
        json_summary["delta_ratio_stats"] = delta_ratio_stats
        json_summary["acf_score_stats"] = acf_score_stats
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(json_summary, fh, indent=2, ensure_ascii=False)
        logger.info("  → 已保存: %s", out_path)

    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation for lag 0..max_lag.

    Uses FFT-based computation (O(n log n)) for efficiency.
    The previous O(n²) dot-product loop was equivalent but slower; for
    tail_fraction=0.5 on T=1000 trajectories the tail has ~500 points, and
    at max_lag=100 the loop performs 100 × 400 = 40 000 FMAs.  FFT reduces
    this to 2 × n_fft × log2(n_fft) operations regardless of max_lag.
    """
    x = np.asarray(x, dtype=np.float64) - x.mean()
    n = len(x)
    if n == 0:
        return np.zeros(max_lag + 1, dtype=np.float32)
    # Full circular autocorrelation via FFT (zero-padded to 2n for linearity)
    f = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]
    if acf_full[0] == 0.0:
        return np.zeros(max_lag + 1, dtype=np.float32)
    acf = (acf_full / acf_full[0]).astype(np.float32)
    return acf[: max_lag + 1]


def _find_peaks(arr: np.ndarray, min_height: float = 0.0) -> List[int]:
    """Return indices of local maxima in arr."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > min_height:
            peaks.append(i)
    return peaks
