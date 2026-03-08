"""
Power Spectrum & Oscillation Frequency Analysis
=================================================

对自由动力学轨迹做频谱分析，回答：

  系统振荡的主要时间尺度是什么？哪些脑区在哪个频段最活跃？

分析内容
--------
1. **功率谱密度（PSD）**：对每条轨迹每个脑区做 FFT，得到 N×T/2+1 功率矩阵，
   再对轨迹和脑区取均值得到全局平均 PSD。

2. **主导频率**：从均值 PSD 中识别峰值频率（排除 DC 分量）。

3. **频段标注**（fMRI TR=2 s，Nyquist = 0.25 Hz）：
   ┌──────────────┬────────────────┬───────────────────────────────────┐
   │ 频段         │ 频率范围        │ 生理参考                           │
   ├──────────────┼────────────────┼───────────────────────────────────┤
   │ infraslow    │ < 0.027 Hz     │ resting-state BOLD 慢涨落           │
   │ very_low     │ 0.027–0.073 Hz │ DMN/SN 静息态网络                  │
   │ low          │ 0.073–0.15 Hz  │ 高频 BOLD 涨落                     │
   │ high         │ 0.15–0.25 Hz   │ 呼吸/心跳混叠（TR=2s 可见）        │
   └──────────────┴────────────────┴───────────────────────────────────┘
   对更短 dt（EEG 级别）则对应 delta/theta/alpha/beta/gamma 节律。

4. **空间振荡模态**：对每个频段，计算各脑区在该频段的平均功率，
   得到"频段地形图"（topography per frequency band）。

输出文件
--------
  power_spectrum.npy          — shape (n_freqs,)，全局平均 PSD
  power_spectrum_regional.npy — shape (N, n_freqs)，每个脑区的平均 PSD
  power_spectrum_report.json  — 数值摘要（主导频率、频段功率比等）
  power_spectrum.png          — 全局 PSD + 频段标注
  spatial_spectral_modes.png  — 频段地形图（脑区 × 频段功率热图）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import scipy.signal.detrend once at module load time.
# When available it provides a correct O(N·T) per-channel linear detrend;
# the vectorised NumPy fallback below is equivalent but kept as a safeguard.
try:
    from scipy.signal import detrend as _scipy_detrend
    _HAS_SCIPY_DETREND: bool = True
except ImportError:
    _HAS_SCIPY_DETREND = False

# fMRI 频段定义（Hz）—— 当 dt≈2s（TR=2s）时适用
_FMRI_BANDS: List[Tuple[str, float, float]] = [
    ("infraslow", 0.0,   0.027),
    ("very_low",  0.027, 0.073),
    ("low",       0.073, 0.15),
    ("high",      0.15,  0.25),
]

# EEG 频段定义（Hz）—— 当 dt≈0.004s（250 Hz）时适用
_EEG_BANDS: List[Tuple[str, float, float]] = [
    ("delta",  0.5,   4.0),
    ("theta",  4.0,   8.0),
    ("alpha",  8.0,  13.0),
    ("beta",  13.0,  30.0),
    ("gamma", 30.0, 100.0),
]

# 选择频段集合的 Nyquist 分界（Hz）
_EEG_NYQUIST_THRESHOLD: float = 5.0   # Nyquist > 5 Hz → use EEG bands


# ─────────────────────────────────────────────────────────────────────────────
# Core PSD computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_trajectory_psd(
    trajectories: np.ndarray,
    dt: float = 1.0,
    burnin: int = 10,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算所有轨迹的功率谱密度（PSD）。

    对每条轨迹的每个脑区应用窗函数后做 FFT，取振幅平方作为功率谱，
    最后对所有轨迹和脑区取均值。

    Args:
        trajectories: shape (n_traj, T, N)。
        dt:           时间步长（秒）；用于换算频率轴为 Hz。
        burnin:       跳过每条轨迹前 burnin 步（去除瞬态）。
        window:       窗函数 ("hann" / "none")；Hann 窗减少频谱泄漏。

    Returns:
        freqs:      shape (n_freqs,)，频率轴（Hz），n_freqs = T_use//2 + 1。
        mean_psd:   shape (n_freqs,)，对所有轨迹+所有脑区平均的 PSD。
        region_psd: shape (N, n_freqs)，对所有轨迹平均、保留脑区维度的 PSD。
    """
    n_traj, T, N = trajectories.shape
    T_use = T - burnin
    if T_use < 4:
        raise ValueError(
            f"去掉 burnin={burnin} 后时序太短（T_use={T_use}）；"
            f"请减小 burnin 或增加 steps。"
        )

    n_freqs = T_use // 2 + 1
    freqs = np.fft.rfftfreq(T_use, d=dt)   # (n_freqs,)

    # Build window
    if window == "hann":
        win = np.hanning(T_use).astype(np.float32)   # (T_use,)
        win_norm = float(np.sum(win ** 2))
    else:
        win = np.ones(T_use, dtype=np.float32)
        win_norm = float(T_use)

    # Accumulate PSD across trajectories
    region_psd_acc = np.zeros((N, n_freqs), dtype=np.float64)

    for i in range(n_traj):
        seg = trajectories[i, burnin:, :].astype(np.float64)   # (T_use, N)
        # Detrend: remove per-channel linear trend (not just mean).
        # Reason: trajectories initialised from random states converge to the
        # attractor over hundreds of steps; the resulting slow ramp dominates
        # the FFT and pushes the dominant frequency to near-DC (infraslow)
        # even when the true oscillatory dynamics are at much higher frequencies.
        # Standard practice (scipy.signal.detrend, type='linear') removes the
        # best-fit affine function from each channel before windowing + FFT.
        if _HAS_SCIPY_DETREND:
            seg = _scipy_detrend(seg, axis=0, type="linear")
        else:
            # Vectorised fallback: subtract per-channel linear fit
            t_c = np.arange(T_use, dtype=np.float64) - (T_use - 1) / 2.0
            t_var = float((t_c ** 2).sum())
            slopes = (t_c @ seg) / t_var          # (N,)
            seg -= t_c[:, None] * slopes[None, :]  # remove slope
            seg -= seg.mean(axis=0, keepdims=True)  # remove mean
        # Apply window: (T_use, N) * (T_use, 1)
        seg_w = seg * win[:, None]
        # FFT: shape (n_freqs, N)
        fft_vals = np.fft.rfft(seg_w, axis=0)
        psd_i = (np.abs(fft_vals) ** 2) / win_norm      # (n_freqs, N)
        # Double non-DC/Nyquist bins (one-sided spectrum):
        # For even T_use, Nyquist (last bin) should NOT be doubled.
        # For odd T_use, there is no Nyquist bin — all non-DC bins are doubled.
        if T_use % 2 == 0:
            psd_i[1:-1] *= 2
        else:
            psd_i[1:] *= 2
        region_psd_acc += psd_i.T   # (N, n_freqs)

    region_psd = region_psd_acc / n_traj                # (N, n_freqs)
    mean_psd = region_psd.mean(axis=0)                  # (n_freqs,)

    return freqs, mean_psd, region_psd


# ─────────────────────────────────────────────────────────────────────────────
# Band analysis
# ─────────────────────────────────────────────────────────────────────────────

def identify_brain_frequency_bands(
    freqs: np.ndarray,
    psd: np.ndarray,
    dt: float = 1.0,
) -> Dict:
    """
    从均值 PSD 识别主导频率并按脑节律频段统计功率。

    Args:
        freqs:  shape (n_freqs,)，频率轴（Hz）。
        psd:    shape (n_freqs,)，全局均值 PSD。
        dt:     时间步长（秒），用于判断选用 EEG 还是 fMRI 频段。

    Returns:
        dict 包含:
          dominant_freq_hz      : float，主峰频率
          dominant_freq_band    : str，对应的频段名称
          total_power           : float
          band_powers           : Dict[str, float]，各频段功率
          band_power_fractions  : Dict[str, float]，各频段功率占比
          nyquist_hz            : float
          bands_used            : str，"fmri" 或 "eeg"
    """
    nyquist = freqs[-1]
    bands = _EEG_BANDS if nyquist > _EEG_NYQUIST_THRESHOLD else _FMRI_BANDS
    bands_label = "eeg" if nyquist > _EEG_NYQUIST_THRESHOLD else "fmri"

    total_power = float(psd.sum())

    # Dominant frequency (exclude DC bin 0)
    if len(psd) > 1:
        peak_idx = int(np.argmax(psd[1:])) + 1
        dominant_freq = float(freqs[peak_idx])
    else:
        dominant_freq = 0.0

    # Band powers
    band_powers: Dict[str, float] = {}
    for bname, blow, bhigh in bands:
        mask = (freqs >= blow) & (freqs < bhigh)
        band_powers[bname] = float(psd[mask].sum())

    # Normalise
    band_fracs: Dict[str, float] = {
        k: round(v / (total_power + 1e-30), 6)
        for k, v in band_powers.items()
    }

    # Which band contains the dominant frequency?
    dom_band = "unknown"
    for bname, blow, bhigh in bands:
        if blow <= dominant_freq < bhigh:
            dom_band = bname
            break

    return {
        "dominant_freq_hz": round(dominant_freq, 6),
        "dominant_freq_band": dom_band,
        "total_power": round(total_power, 6),
        "band_powers": {k: round(v, 6) for k, v in band_powers.items()},
        "band_power_fractions": band_fracs,
        "nyquist_hz": round(float(nyquist), 4),
        "bands_used": bands_label,
    }


def compute_spatial_spectral_modes(
    freqs: np.ndarray,
    region_psd: np.ndarray,
    dt: float = 1.0,
) -> Dict:
    """
    计算每个频段的空间功率地形图（各脑区在该频段的功率）。

    Args:
        freqs:      shape (n_freqs,)，频率轴（Hz）。
        region_psd: shape (N, n_freqs)，每个脑区的 PSD。
        dt:         时间步长（秒）。

    Returns:
        dict 包含:
          band_topographies: Dict[str, np.ndarray (N,)]，各频段脑区功率向量
          peak_region_per_band: Dict[str, int]，各频段最强脑区的索引
    """
    nyquist = freqs[-1]
    bands = _EEG_BANDS if nyquist > _EEG_NYQUIST_THRESHOLD else _FMRI_BANDS

    band_topo: Dict[str, np.ndarray] = {}
    peak_region: Dict[str, int] = {}

    for bname, blow, bhigh in bands:
        mask = (freqs >= blow) & (freqs < bhigh)
        if mask.any():
            topo = region_psd[:, mask].sum(axis=1)   # (N,)
        else:
            topo = np.zeros(region_psd.shape[0])
        band_topo[bname] = topo
        peak_region[bname] = int(np.argmax(topo))

    return {
        "band_topographies": band_topo,
        "peak_region_per_band": peak_region,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integration function
# ─────────────────────────────────────────────────────────────────────────────

def run_power_spectrum_analysis(
    trajectories: np.ndarray,
    dt: float = 1.0,
    burnin: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    完整功率谱分析流程：PSD 计算 → 频段分析 → 空间模态。

    Args:
        trajectories: shape (n_traj, T, N)。
        dt:           时间步长（秒）。
        burnin:       跳过每条轨迹开头的步数。
        output_dir:   结果保存目录；None → 不保存。

    Returns:
        dict 包含:
          freqs               : np.ndarray (n_freqs,)
          mean_psd            : np.ndarray (n_freqs,)
          region_psd          : np.ndarray (N, n_freqs)
          band_analysis       : dict（identify_brain_frequency_bands 输出）
          spatial_modes       : dict（compute_spatial_spectral_modes 输出）
    """
    freqs, mean_psd, region_psd = compute_trajectory_psd(
        trajectories, dt=dt, burnin=burnin
    )
    band_analysis = identify_brain_frequency_bands(freqs, mean_psd, dt=dt)
    spatial_modes = compute_spatial_spectral_modes(freqs, region_psd, dt=dt)

    logger.info(
        "功率谱分析完成: dominant_freq=%.4f Hz [%s], "
        "Nyquist=%.3f Hz, 最活跃脑区（very_low/low）= 区域 %s",
        band_analysis["dominant_freq_hz"],
        band_analysis["dominant_freq_band"],
        band_analysis["nyquist_hz"],
        ", ".join(
            str(spatial_modes["peak_region_per_band"].get(b, "N/A"))
            for b in ["very_low", "low"]
            if b in spatial_modes["peak_region_per_band"]
        ),
    )

    result = {
        "freqs": freqs,
        "mean_psd": mean_psd,
        "region_psd": region_psd,
        "band_analysis": band_analysis,
        "spatial_modes": spatial_modes,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "power_spectrum.npy", mean_psd)
        np.save(out / "power_spectrum_regional.npy", region_psd)
        np.save(out / "power_spectrum_freqs.npy", freqs)

        # Save band topographies
        for bname, topo in spatial_modes["band_topographies"].items():
            np.save(out / f"spatial_mode_{bname}.npy", topo)

        # JSON report
        report = {
            "band_analysis": band_analysis,
            "peak_region_per_band": spatial_modes["peak_region_per_band"],
            "dt": dt,
            "n_freqs": int(len(freqs)),
            "n_trajectories": int(trajectories.shape[0]),
            "steps": int(trajectories.shape[1]),
            "n_regions": int(trajectories.shape[2]),
        }
        with open(out / "power_spectrum_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存功率谱报告: %s", out / "power_spectrum_report.json")

        _try_plot_psd(freqs, mean_psd, region_psd, band_analysis, spatial_modes, out)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_psd(
    freqs: np.ndarray,
    mean_psd: np.ndarray,
    region_psd: np.ndarray,
    band_analysis: Dict,
    spatial_modes: Dict,
    output_dir: Path,
) -> None:
    """绘制全局 PSD + 频段标注 + 空间功率地形图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    nyquist = band_analysis["nyquist_hz"]
    bands_used = band_analysis["bands_used"]
    bands = _EEG_BANDS if bands_used == "eeg" else _FMRI_BANDS
    band_colors = plt.get_cmap("tab10", len(bands))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: Global PSD ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(freqs, mean_psd + 1e-30, "k-", lw=1.5, label="Mean PSD")

    # Shade frequency bands
    patches = []
    for j, (bname, blow, bhigh) in enumerate(bands):
        bhigh_clip = min(bhigh, nyquist)
        if blow >= nyquist:
            continue
        ax.axvspan(blow, bhigh_clip, alpha=0.12, color=band_colors(j))
        patches.append(mpatches.Patch(color=band_colors(j), alpha=0.5,
                                      label=f"{bname} ({blow:.3f}–{bhigh_clip:.3f} Hz)"))

    # Mark dominant frequency
    dom_f = band_analysis["dominant_freq_hz"]
    if dom_f > freqs[0]:
        ax.axvline(dom_f, color="red", ls="--", lw=1.2,
                   label=f"Peak {dom_f:.4f} Hz [{band_analysis['dominant_freq_band']}]")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title(f"Global Mean Power Spectrum\n"
                 f"Nyquist={nyquist:.3f} Hz, bands={bands_used}")
    ax.set_xlim(0, nyquist)
    ax.legend(handles=[ax.lines[0]] + [ax.lines[-1]] + patches,
              fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    # ── Right: Spatial topographies per band ──────────────────────────────────
    ax2 = axes[1]
    band_topos = spatial_modes["band_topographies"]
    valid_bands = [(bname, topo)
                   for bname, topo in band_topos.items()
                   if topo.max() > 0]
    if valid_bands:
        topo_matrix = np.vstack([t for _, t in valid_bands])   # (n_bands, N)
        # Normalise each band row to [0, 1]
        row_max = topo_matrix.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        topo_norm = topo_matrix / row_max

        im = ax2.imshow(topo_norm, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax2, label="Normalised band power")
        ax2.set_xlabel("Brain Region Index")
        ax2.set_ylabel("Frequency Band")
        ax2.set_yticks(range(len(valid_bands)))
        ax2.set_yticklabels([b for b, _ in valid_bands], fontsize=8)
        # Annotate peak regions
        for row, (bname, _) in enumerate(valid_bands):
            pk = spatial_modes["peak_region_per_band"].get(bname, -1)
            if pk >= 0:
                ax2.scatter(pk, row, color="blue", s=60, marker="v",
                            zorder=5, label=f"Peak" if row == 0 else "")
        ax2.set_title("Spatial Spectral Topography\n"
                      "(normalised power per band per region)\n"
                      "▼ = peak region per band", fontsize=9)
        if any(b == "infraslow" for b, _ in valid_bands):
            ax2.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "power_spectrum.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存功率谱图: %s", output_dir / "power_spectrum.png")
