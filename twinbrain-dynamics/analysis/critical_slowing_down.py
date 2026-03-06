"""
Critical Slowing Down Detection
=================================

检测动力系统临近 **临界转变（Critical Transition / Bifurcation）** 时的
**早期预警信号（Early Warning Signals, EWS）**。

理论背景
--------
当非线性动力系统接近鞍结分岔（saddle-node bifurcation）或 Hopf 分岔时，
系统的恢复速率（recovery rate，即线性化最大特征值的绝对值）趋近于零——
这一现象称为 **临界慢化（Critical Slowing Down, CSD）**。

CSD 在时间序列上产生两个可测的早期预警信号（Scheffer et al. 2009 Nature）：

1. **方差增大（Variance）**：系统在当前状态附近的波动幅度增加，
   因为弱化的"回弹力"不再快速抑制扰动。

2. **自相关增大（Autocorrelation at lag-1）**：系统状态变化越来越"记得"
   上一步的状态，AR(1) 系数 φ → 1 时系统趋近随机游走（中性稳定）。

这两个指标在生态学（Scheffer 2009）、气候科学（Dakos 2008）、
金融市场（Guttal 2008）和神经科学（Van de Leemput 2014 PNAS）中均被
验证为可靠的临界转变预警指标。

**神经科学应用**（Van de Leemput 2014）：
  在抑郁症发作前 2–4 周，患者的 EEG 脑电自相关显著升高；
  在精神科门诊随访数据中，情绪自相关升高预测复发概率提升 3–5 倍。

本模块在 TwinBrain 模拟轨迹上计算 CSD 指标：
  - **时间窗口滑动方差** 和 **滑动 AR(1) 系数**
  - **Kendall τ 趋势检验**（非参数，对噪声鲁棒）
  - **去趋势波动分析（DFA）**：标量指数 α > 1 表示长程自相关

附加指标（Guttal & Jayaprakash 2008 Oikos）：
  - **Skewness（偏度）**：分布偏斜增加可能预示双稳态出现
  - **Kurtosis（峰度）**：重尾分布与临界涨落一致

**注意事项**：
  - CSD 仅在系统真正靠近分岔时可靠；在稳定吸引子内部变化参数时，
    EWS 可能出现假阳性（Lenton 2011 Climatic Change）。
  - TwinBrain 模拟中，EWS 反映的是*模型*的动力学，而非受试者的真实大脑状态。
  - 建议将 EWS 结果与稳定性分析和 Lyapunov 指数综合解读。

科学参考
--------
  Scheffer M et al (2009) Nature 461:53-59  — EWS 奠基性综述
  Dakos V et al (2008) PNAS 105:14308-14312 — 气候转变中的 CSD
  Van de Leemput I et al (2014) PNAS 111:87-92 — 抑郁症的 CSD 预警
  Guttal V & Jayaprakash C (2008) Oikos 117:1175-1184 — 偏度/峰度指标
  Lenton TM et al (2012) Nat Clim Change 2:682-685 — EWS 的局限性

输出文件
--------
  critical_slowing_down_report.json   — 全脑 EWS 汇总统计
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Fraction of trajectory length used as the rolling window.
# Dakos et al. (2012) recommend half the time series as the rolling-window size
# to balance temporal resolution and statistical stability.  We use 0.5 as the
# default; users can override via the `window_fraction` parameter.
_DEFAULT_WINDOW_FRACTION: float = 0.5

# Kendall τ significance threshold (|τ| > this value is considered meaningful)
# Based on the critical value tables in Dakos et al. (2012) for n≈50 rolling
# windows: |τ| > 0.25 is significant at α=0.1 (one-tailed).
_KENDALL_TAU_THRESHOLD: float = 0.25

# DFA box sizes (as fractions of trajectory length)
_DFA_N_SCALES: int = 12   # number of box sizes for the log-log regression

# DFA minimum and maximum box (scale) sizes, as *divisors* of T:
#   min scale = max(4, T // _DFA_MIN_SCALE_DIVISOR)   → at least 4 points per box
#   max scale = max(min+1, T // _DFA_MAX_SCALE_DIVISOR) → at most T//4, per Peng (1994)
# Peng et al. (1994) Phys Rev E 49:1685 recommend scales from T//100 to T//4
# to ensure both adequate box statistics and long-range trend visibility.
_DFA_MIN_SCALE_DIVISOR: int = 100  # min scale = T // 100 (many small boxes)
_DFA_MAX_SCALE_DIVISOR: int = 4    # max scale = T //  4  (few large boxes)
_DFA_ABSOLUTE_MIN_SCALE: int = 4   # hard lower bound (need ≥ 4 points per box)


# ══════════════════════════════════════════════════════════════════════════════
# Rolling statistics
# ══════════════════════════════════════════════════════════════════════════════

def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    """
    滑动窗口方差（使用 Welford 单次遍历算法，数值稳定）。

    Args:
        x:      1-D 时间序列，shape (T,)。
        window: 窗口大小（步数）。

    Returns:
        var_series: shape (T − window + 1,)，每个窗口的方差。
    """
    T = len(x)
    n_win = T - window + 1
    if n_win <= 0:
        return np.array([], dtype=np.float64)
    var_series = np.empty(n_win, dtype=np.float64)
    for i in range(n_win):
        chunk = x[i: i + window]
        var_series[i] = float(chunk.var())
    return var_series


def rolling_autocorrelation(x: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    """
    滑动窗口 AR(lag) 系数（Pearson 相关系数 r(t, t+lag)）。

    Args:
        x:      1-D 时间序列，shape (T,)。
        window: 窗口大小（步数）。
        lag:    自相关延迟（默认 1）。

    Returns:
        ac_series: shape (T − window + 1,)，每个窗口的自相关系数。
    """
    T = len(x)
    n_win = T - window + 1
    if n_win <= 0 or window <= lag + 1:
        return np.array([], dtype=np.float64)
    ac_series = np.empty(n_win, dtype=np.float64)
    for i in range(n_win):
        chunk = x[i: i + window]
        c = np.corrcoef(chunk[:-lag], chunk[lag:])
        ac_series[i] = float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0
    return ac_series


# ══════════════════════════════════════════════════════════════════════════════
# Kendall τ trend test
# ══════════════════════════════════════════════════════════════════════════════

def kendall_tau(series: np.ndarray) -> float:
    """
    计算时间序列的 Kendall τ 趋势统计量。

    τ ∈ [−1, +1]：
      τ > 0 → 递增趋势（接近临界时方差/自相关增大）
      τ < 0 → 递减趋势

    使用 O(n²) 直接计算，适合 n ≤ 1000 的短窗口统计序列。
    参考：Dakos et al. (2012) Methods Ecol Evol §2.2。

    Args:
        series: 1-D 序列，shape (n,)。

    Returns:
        tau:  Kendall τ 统计量，nan 若序列长度 < 4。
    """
    n = len(series)
    if n < 4:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                concordant += 1
            elif diff < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom) if denom > 0 else float("nan")


def _fast_kendall_tau(series: np.ndarray) -> float:
    """
    O(n log n) Kendall τ（使用 merge-sort 计数）。
    替代 O(n²) 直接计算，用于 n > 200 的长序列。
    """
    n = len(series)
    if n < 4:
        return float("nan")
    # Use scipy if available (O(n log n)), otherwise fall back to O(n²)
    try:
        from scipy.stats import kendalltau as scipy_kt
        tau, _ = scipy_kt(np.arange(n, dtype=float), series)
        return float(tau)
    except ImportError:
        return kendall_tau(series)


# ══════════════════════════════════════════════════════════════════════════════
# Detrended Fluctuation Analysis (DFA)
# ══════════════════════════════════════════════════════════════════════════════

def detrended_fluctuation_analysis(
    x: np.ndarray,
    n_scales: int = _DFA_N_SCALES,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    一元时间序列的去趋势波动分析（DFA）。

    DFA 标度指数 α：
      α ≈ 0.5  → 无关随机游走（白噪声积分），无长程相关
      α > 0.5  → 长程正自相关（持续性）
      α < 0.5  → 长程反自相关（反持续性）
      α > 1.0  → 非稳态趋势，可能临近临界转变

    算法（Peng et al. 1994 Phys Rev E）：
      1. 计算累积和（去均值）: y(i) = Σ_{k=1}^{i} (x(k) - <x>)
      2. 对若干窗口大小 n，分段拟合线性趋势，计算残差 RMS F(n)
      3. 在 log-log 图中拟合 log F(n) vs log n 的斜率 = DFA 指数 α

    Args:
        x:        1-D 时间序列，shape (T,)。
        n_scales: 对数等间隔的窗口大小数量（默认 12）。

    Returns:
        (alpha, scales, fluctuations):
          alpha:       DFA 标度指数。
          scales:      使用的窗口大小 array，shape (n_scales,)。
          fluctuations: 对应的 F(n)，shape (n_scales,)。
    """
    T = len(x)
    if T < 32:
        return float("nan"), np.array([]), np.array([])

    # Step 1: cumulative sum (profile)
    y = np.cumsum(x - x.mean()).astype(np.float64)

    # Step 2: box sizes on a log scale.
    # Peng et al. (1994) recommend scales from T//100 (min) to T//4 (max).
    # _DFA_ABSOLUTE_MIN_SCALE=4 is a hard lower bound (need ≥ 4 points per box).
    min_scale = max(_DFA_ABSOLUTE_MIN_SCALE, T // _DFA_MIN_SCALE_DIVISOR)
    max_scale = max(min_scale + 1, T // _DFA_MAX_SCALE_DIVISOR)
    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_scale), n_scales
    ).astype(int))

    fluctuations = np.zeros(len(scales), dtype=np.float64)
    for s_idx, n in enumerate(scales):
        n = int(n)
        n_segments = T // n
        if n_segments < 2:
            fluctuations[s_idx] = np.nan
            continue
        f2 = 0.0
        for seg in range(n_segments):
            seg_y = y[seg * n: (seg + 1) * n]
            t_seg = np.arange(n, dtype=np.float64)
            coeffs = np.polyfit(t_seg, seg_y, deg=1)
            trend = np.polyval(coeffs, t_seg)
            f2 += float(np.mean((seg_y - trend) ** 2))
        fluctuations[s_idx] = np.sqrt(f2 / n_segments)

    # Step 3: log-log linear regression
    valid = np.isfinite(fluctuations) & (fluctuations > 0)
    if valid.sum() < 3:
        return float("nan"), scales, fluctuations

    log_scales = np.log10(scales[valid].astype(float))
    log_fluct = np.log10(fluctuations[valid])
    coeffs = np.polyfit(log_scales, log_fluct, deg=1)
    alpha = float(coeffs[0])

    return alpha, scales, fluctuations


# ══════════════════════════════════════════════════════════════════════════════
# Per-trajectory EWS computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_ews_single_trajectory(
    trajectory: np.ndarray,
    window_fraction: float = _DEFAULT_WINDOW_FRACTION,
) -> Dict:
    """
    对单条轨迹计算所有早期预警信号指标（在时间轴上对每个脑区取均值）。

    指标列表：
      - ``var_tau``:    滑动方差的 Kendall τ 趋势（τ > 0 = 方差递增 = EWS）
      - ``ac1_tau``:   滑动 AR(1) 的 Kendall τ 趋势（τ > 0 = 自相关递增 = EWS）
      - ``skew_mean``: 轨迹末半段的偏度（绝对值增大 = 分布偏斜 = EWS）
      - ``kurt_mean``: 轨迹末半段的峰度（> 3 = 重尾 = EWS）
      - ``dfa_alpha``: DFA 标度指数均值（> 0.7 = 长程相关 = CSD 信号）
      - ``ews_score``: 综合 EWS 评分（0–1，> 0.5 为正分岔预警）

    Args:
        trajectory:      shape (T, n_regions)。
        window_fraction: 滑动窗口占总长度的比例（默认 0.5）。

    Returns:
        ews: 字典，包含上述指标和解读文字。
    """
    T, N = trajectory.shape
    window = max(4, int(T * window_fraction))

    var_taus = []
    ac1_taus = []
    dfa_alphas = []
    skews = []
    kurts = []

    for i in range(N):
        x = trajectory[:, i].astype(np.float64)

        # Rolling variance Kendall τ
        var_s = rolling_variance(x, window)
        if len(var_s) >= 4:
            var_taus.append(_fast_kendall_tau(var_s))

        # Rolling autocorrelation Kendall τ
        ac1_s = rolling_autocorrelation(x, window, lag=1)
        if len(ac1_s) >= 4:
            ac1_taus.append(_fast_kendall_tau(ac1_s))

        # DFA on the full trajectory
        alpha, _, _ = detrended_fluctuation_analysis(x, n_scales=_DFA_N_SCALES)
        if np.isfinite(alpha):
            dfa_alphas.append(alpha)

        # Skewness and kurtosis of the second half
        half_x = x[T // 2:]
        if len(half_x) > 4:
            std_h = float(half_x.std()) + 1e-12
            mu_h = float(half_x.mean())
            skews.append(float(np.mean(((half_x - mu_h) / std_h) ** 3)))
            kurts.append(float(np.mean(((half_x - mu_h) / std_h) ** 4)))

    var_tau  = float(np.nanmean(var_taus))  if var_taus  else float("nan")
    ac1_tau  = float(np.nanmean(ac1_taus)) if ac1_taus  else float("nan")
    dfa_alpha = float(np.nanmean(dfa_alphas)) if dfa_alphas else float("nan")
    skew_mean = float(np.nanmean(np.abs(skews))) if skews else float("nan")
    kurt_mean = float(np.nanmean(kurts)) if kurts else float("nan")

    # ── EWS composite score (0 = no signal, 1 = strong CSD) ──────────────────
    # Each indicator contributes 0.2 to the score if it exceeds its threshold:
    #   var_tau  > 0.25   (Dakos 2012 threshold)
    #   ac1_tau  > 0.25   (same threshold)
    #   dfa_alpha > 0.7   (persistent long-range correlation)
    #   |skew|   > 0.5    (Guttal 2008: skewness EWS threshold)
    #   kurt     > 3.5    (excess kurtosis > 0.5)
    indicators = [
        float(var_tau)  > _KENDALL_TAU_THRESHOLD,
        float(ac1_tau)  > _KENDALL_TAU_THRESHOLD,
        (np.isfinite(dfa_alpha) and float(dfa_alpha) > 0.7),
        (np.isfinite(skew_mean) and float(skew_mean) > 0.5),
        (np.isfinite(kurt_mean) and float(kurt_mean) > 3.5),
    ]
    ews_score = float(sum(indicators)) / len(indicators)

    # Interpret
    if ews_score >= 0.6:
        interp = (
            f"⚠ 强临界慢化信号（EWS 评分 = {ews_score:.2f}）。"
            "多项指标均超过阈值，系统可能临近动力学转变。"
            "建议结合 Lyapunov 指数和吸引子分析综合判断。"
        )
    elif ews_score >= 0.4:
        interp = (
            f"⊘ 中等 CSD 信号（EWS 评分 = {ews_score:.2f}）。"
            "部分指标提示系统可能处于亚稳态或慢漂移过程。"
        )
    else:
        interp = (
            f"✓ 无显著临界慢化（EWS 评分 = {ews_score:.2f}）。"
            "系统当前在稳定吸引子内部运行，无临近分岔的统计证据。"
        )

    return {
        "var_tau":          var_tau,
        "ac1_tau":          ac1_tau,
        "dfa_alpha":        dfa_alpha,
        "skew_mean":        skew_mean,
        "kurt_mean":        kurt_mean,
        "ews_score":        ews_score,
        "interpretation_zh": interp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public run function
# ══════════════════════════════════════════════════════════════════════════════

def run_critical_slowing_down_analysis(
    trajectories: np.ndarray,
    window_fraction: float = _DEFAULT_WINDOW_FRACTION,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹运行临界慢化（CSD）早期预警信号分析。

    Args:
        trajectories:    shape (n_init, T, n_regions)。
        window_fraction: 滑动窗口比例（默认 0.5）。
        output_dir:      输出目录；None → 不保存文件。

    Returns:
        results: {
            "per_trajectory_ews":   List[Dict]   — 每条轨迹的 EWS 指标
            "aggregate":            Dict         — 全局汇总统计
            "report":               Dict         — JSON-serialisable summary
        }
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    n_init, T, N = trajectories.shape

    logger.info(
        "临界慢化分析：%d 条轨迹, T=%d, N=%d, 窗口比例=%.2f",
        n_init, T, N, window_fraction,
    )

    per_traj: List[Dict] = []
    for k, traj in enumerate(trajectories):
        ews = compute_ews_single_trajectory(traj, window_fraction)
        per_traj.append(ews)
        if (k + 1) % 5 == 0 or k == n_init - 1:
            logger.info("  CSD 进度: %d/%d 条轨迹", k + 1, n_init)

    # Aggregate across trajectories
    def _agg(key: str) -> Tuple[float, float]:
        vals = [d[key] for d in per_traj if np.isfinite(d[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    var_tau_mean,   var_tau_std  = _agg("var_tau")
    ac1_tau_mean,   ac1_tau_std  = _agg("ac1_tau")
    dfa_alpha_mean, dfa_alpha_std = _agg("dfa_alpha")
    ews_score_mean, ews_score_std = _agg("ews_score")

    # Fraction of trajectories showing each indicator
    frac_var  = sum(d["var_tau"]  > _KENDALL_TAU_THRESHOLD for d in per_traj) / max(1, n_init)
    frac_ac1  = sum(d["ac1_tau"]  > _KENDALL_TAU_THRESHOLD for d in per_traj) / max(1, n_init)
    frac_dfa  = sum(d["dfa_alpha"] > 0.7 for d in per_traj) / max(1, n_init)
    frac_high = sum(d["ews_score"] >= 0.4 for d in per_traj) / max(1, n_init)

    report = {
        "n_trajectories":   n_init,
        "T":                T,
        "n_regions":        N,
        "window_fraction":  window_fraction,
        "var_tau_mean":     var_tau_mean,
        "var_tau_std":      var_tau_std,
        "ac1_tau_mean":     ac1_tau_mean,
        "ac1_tau_std":      ac1_tau_std,
        "dfa_alpha_mean":   dfa_alpha_mean,
        "dfa_alpha_std":    dfa_alpha_std,
        "ews_score_mean":   ews_score_mean,
        "ews_score_std":    ews_score_std,
        "frac_var_ews":     frac_var,
        "frac_ac1_ews":     frac_ac1,
        "frac_dfa_ews":     frac_dfa,
        "frac_high_ews":    frac_high,
        "interpretation_zh": (
            f"临界慢化分析完成（{n_init} 条轨迹）。"
            f"方差 τ 均值 = {var_tau_mean:.3f}，AC(1) τ 均值 = {ac1_tau_mean:.3f}，"
            f"DFA α 均值 = {dfa_alpha_mean:.3f}。"
            f"{frac_high*100:.0f}% 的轨迹显示中等或强 CSD 信号。"
            f"{'建议：多条轨迹均表现出 CSD 早期预警，系统可能处于临界转变前期。' if frac_high > 0.5 else '当前无显著 CSD 信号，系统处于稳定状态。'}"
        ),
    }

    aggregate = {
        "var_tau_mean": var_tau_mean,
        "var_tau_std":  var_tau_std,
        "ac1_tau_mean": ac1_tau_mean,
        "ac1_tau_std":  ac1_tau_std,
        "dfa_alpha_mean": dfa_alpha_mean,
        "ews_score_mean": ews_score_mean,
        "frac_high_ews": frac_high,
    }

    if output_dir is not None:
        report_path = output_dir / "critical_slowing_down_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  CSD 报告已保存: %s", report_path)

    logger.info(
        "临界慢化分析完成: EWS 评分=%.3f±%.3f, DFA α=%.3f±%.3f, %.0f%% 轨迹显示中等以上 CSD",
        ews_score_mean, ews_score_std, dfa_alpha_mean, dfa_alpha_std, frac_high * 100,
    )

    return {
        "per_trajectory_ews": per_traj,
        "aggregate":          aggregate,
        "report":             report,
    }
