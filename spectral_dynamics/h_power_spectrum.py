"""
H: Power Spectrum & Spatial Oscillation Modes
===============================================

对自由动力学轨迹做频谱分析，验证假设 H4：

  H4  网络振荡对应已知脑节律（delta/theta/alpha/beta），
      且振荡模态的空间分布与经典脑网络（DMN/SN/FPN）一致。

实现
----
从 twinbrain-dynamics 管线的预计算轨迹直接计算功率谱，无需任何模型调用：

  1. 对每条轨迹的每个脑区做 Hann 窗 + FFT
  2. 取全局均值 PSD（用于主导频率识别）
  3. 计算每个频段的脑区功率地形图（topography）
  4. 优先调用 ``twinbrain-dynamics`` 的 ``run_power_spectrum_analysis``；
     若无法导入则使用本模块内置的轻量版实现。

与 twinbrain-dynamics 的关系
-----------------------------
``twinbrain-dynamics/analysis/power_spectrum.py`` 是"权威实现"，本模块
是适配 spectral_dynamics 管线（从 .npy 文件加载，无 simulator 依赖）的封装层。

输出文件（同 twinbrain-dynamics）
-----------------------------------
  power_spectrum_h.json      — 频段分析摘要
  power_spectrum_h.png       — 全局 PSD + 频段标注
  spatial_modes_h.png        — 频段地形图
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to reuse twinbrain-dynamics implementation
_TD_DIR = Path(__file__).parent.parent / "twinbrain-dynamics"
if _TD_DIR.exists() and str(_TD_DIR) not in sys.path:
    sys.path.insert(0, str(_TD_DIR))

try:
    from analysis.power_spectrum import (
        compute_trajectory_psd,
        identify_brain_frequency_bands,
        compute_spatial_spectral_modes,
        run_power_spectrum_analysis as _td_run_psd,
    )
    _TD_PSD_AVAILABLE = True
except ImportError:
    _TD_PSD_AVAILABLE = False
    logger.debug("twinbrain-dynamics power_spectrum 未找到，使用内置实现。")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_power_spectrum(
    trajectories: np.ndarray,
    dt: float = 1.0,
    burnin: int = 10,
    output_dir: Optional[Path] = None,
    label: str = "traj",
) -> Dict:
    """
    完整功率谱分析：PSD → 频段 → 空间模态。

    优先调用 ``twinbrain-dynamics`` 的实现（若可用），否则使用内置轻量版。

    Args:
        trajectories: shape (n_traj, T, N)。
        dt:           时间步长（秒）；fMRI TR 通常为 2.0s。
        burnin:       跳过前几步（去除瞬态）。
        output_dir:   结果保存目录；None → 不保存。
        label:        输出文件名标签。

    Returns:
        dict 包含 freqs, mean_psd, region_psd, band_analysis, spatial_modes。
    """
    if _TD_PSD_AVAILABLE:
        # Delegate to canonical implementation
        result = _td_run_psd(
            trajectories=trajectories,
            dt=dt,
            burnin=burnin,
            output_dir=output_dir / f"h_{label}" if output_dir else None,
        )
    else:
        result = _run_power_spectrum_builtin(
            trajectories, dt=dt, burnin=burnin,
            output_dir=output_dir, label=label,
        )

    # Save JSON summary to output_dir directly (for spectral_dynamics index)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"power_spectrum_{label}.json"
        band_info = result.get("band_analysis", {})
        summary = {
            "dominant_freq_hz": band_info.get("dominant_freq_hz"),
            "dominant_freq_band": band_info.get("dominant_freq_band"),
            "band_power_fractions": band_info.get("band_power_fractions"),
            "nyquist_hz": band_info.get("nyquist_hz"),
            "bands_used": band_info.get("bands_used"),
            "peak_region_per_band": result.get("spatial_modes", {}).get(
                "peak_region_per_band", {}
            ),
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info("H: 保存功率谱摘要: %s", json_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Built-in lightweight implementation (no twinbrain-dynamics dependency)
# ─────────────────────────────────────────────────────────────────────────────

_FMRI_BANDS = [
    ("infraslow", 0.0,   0.027),
    ("very_low",  0.027, 0.073),
    ("low",       0.073, 0.15),
    ("high",      0.15,  0.25),
]
_EEG_BANDS = [
    ("delta",  0.5,   4.0),
    ("theta",  4.0,   8.0),
    ("alpha",  8.0,  13.0),
    ("beta",  13.0,  30.0),
    ("gamma", 30.0, 100.0),
]


def _run_power_spectrum_builtin(
    trajectories: np.ndarray,
    dt: float = 1.0,
    burnin: int = 10,
    output_dir: Optional[Path] = None,
    label: str = "traj",
) -> Dict:
    """内置轻量版功率谱分析（当 twinbrain-dynamics 不可用时使用）。"""
    n_traj, T, N = trajectories.shape
    T_use = T - burnin
    if T_use < 4:
        raise ValueError(f"T_use={T_use} 太短（burnin={burnin}）")

    freqs = np.fft.rfftfreq(T_use, d=dt)
    n_freqs = len(freqs)
    win = np.hanning(T_use).astype(np.float32)
    win_norm = float(np.sum(win ** 2))

    region_psd_acc = np.zeros((N, n_freqs), dtype=np.float64)
    for i in range(n_traj):
        seg = trajectories[i, burnin:, :].astype(np.float64)
        seg -= seg.mean(axis=0, keepdims=True)
        seg_w = seg * win[:, None]
        fft_v = np.fft.rfft(seg_w, axis=0)
        psd_i = (np.abs(fft_v) ** 2) / win_norm
        if T_use % 2 == 0:
            psd_i[1:-1] *= 2
        else:
            psd_i[1:] *= 2
        region_psd_acc += psd_i.T
    region_psd = region_psd_acc / n_traj
    mean_psd = region_psd.mean(axis=0)

    nyquist = float(freqs[-1])
    bands = _EEG_BANDS if nyquist > 5.0 else _FMRI_BANDS
    bands_label = "eeg" if nyquist > 5.0 else "fmri"

    total_power = float(mean_psd.sum())
    peak_idx = int(np.argmax(mean_psd[1:])) + 1 if len(mean_psd) > 1 else 0
    dom_freq = float(freqs[peak_idx])

    band_powers = {}
    band_fracs = {}
    for bname, blow, bhigh in bands:
        mask = (freqs >= blow) & (freqs < bhigh)
        bp = float(mean_psd[mask].sum())
        band_powers[bname] = round(bp, 6)
        band_fracs[bname] = round(bp / (total_power + 1e-30), 6)

    dom_band = next(
        (b for b, lo, hi in bands if lo <= dom_freq < hi), "unknown"
    )

    band_topos = {}
    peak_region = {}
    for bname, blow, bhigh in bands:
        mask = (freqs >= blow) & (freqs < bhigh)
        topo = region_psd[:, mask].sum(axis=1) if mask.any() else np.zeros(N)
        band_topos[bname] = topo
        peak_region[bname] = int(np.argmax(topo))

    band_analysis = {
        "dominant_freq_hz": round(dom_freq, 6),
        "dominant_freq_band": dom_band,
        "total_power": round(total_power, 6),
        "band_powers": band_powers,
        "band_power_fractions": band_fracs,
        "nyquist_hz": round(nyquist, 4),
        "bands_used": bands_label,
    }
    spatial_modes = {
        "band_topographies": band_topos,
        "peak_region_per_band": peak_region,
    }

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
        np.save(out / f"power_spectrum_{label}.npy", mean_psd)
        np.save(out / f"power_spectrum_regional_{label}.npy", region_psd)
        _try_plot_psd_builtin(freqs, mean_psd, region_psd, band_analysis,
                              spatial_modes, bands, out, label)

    return result


def _try_plot_psd_builtin(freqs, mean_psd, region_psd, band_analysis,
                           spatial_modes, bands, output_dir, label):
    """绘制 PSD 图（内置版）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    nyquist = band_analysis["nyquist_hz"]
    band_colors = plt.get_cmap("tab10", len(bands))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.semilogy(freqs, mean_psd + 1e-30, "k-", lw=1.5, label="Mean PSD")
    patches = []
    for j, (bname, blow, bhigh) in enumerate(bands):
        bh = min(bhigh, nyquist)
        if blow >= nyquist:
            continue
        ax.axvspan(blow, bh, alpha=0.12, color=band_colors(j))
        patches.append(mpatches.Patch(
            color=band_colors(j), alpha=0.5,
            label=f"{bname} ({blow:.3f}–{bh:.3f} Hz)"))
    dom_f = band_analysis["dominant_freq_hz"]
    if dom_f > freqs[0]:
        ax.axvline(dom_f, color="red", ls="--", lw=1.2,
                   label=f"Peak {dom_f:.4f} Hz [{band_analysis['dominant_freq_band']}]")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log)")
    ax.set_title(f"Global Mean PSD [{label}]\nNyquist={nyquist:.3f} Hz")
    ax.set_xlim(0, nyquist)
    ax.legend(handles=[ax.lines[0], ax.lines[-1]] + patches, fontsize=7)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    band_topos = spatial_modes["band_topographies"]
    valid = [(b, t) for b, t in band_topos.items() if t.max() > 0]
    if valid:
        mat = np.vstack([t for _, t in valid])
        row_max = mat.max(axis=1, keepdims=True)
        mat_norm = mat / np.where(row_max > 0, row_max, 1)
        im = ax2.imshow(mat_norm, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax2, label="Normalised band power")
        ax2.set_xlabel("Brain Region Index")
        ax2.set_yticks(range(len(valid)))
        ax2.set_yticklabels([b for b, _ in valid], fontsize=8)
        ax2.set_title("Spatial Spectral Topography\n(per band, per region)")
    fig.tight_layout()
    fig.savefig(output_dir / f"power_spectrum_{label}.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("H: 保存功率谱图: %s", output_dir / f"power_spectrum_{label}.png")
