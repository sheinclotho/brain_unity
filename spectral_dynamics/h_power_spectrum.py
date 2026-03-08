"""
H: Power Spectrum & Spatial Oscillation Modes
===============================================

对自由动力学轨迹做频谱分析，验证假设 H4：

  H4  网络振荡对应已知脑节律（delta/theta/alpha/beta），
      且振荡模态的空间分布与经典脑网络（DMN/SN/FPN）一致。

实现
----
从 twinbrain-dynamics 管线的预计算轨迹直接计算功率谱，无需任何模型调用：

  1. 对每条轨迹的每个脑区做 Hann 窗 + 线性去趋势 + FFT
  2. 取全局均值 PSD（用于主导频率识别）
  3. 计算每个频段的脑区功率地形图（topography）
  4. 委托给 ``twinbrain-dynamics/analysis/power_spectrum.run_power_spectrum_analysis``。
     twinbrain-dynamics 是权威实现；本模块是适配 spectral_dynamics 管线的封装层。

与 twinbrain-dynamics 的关系
-----------------------------
``twinbrain-dynamics/analysis/power_spectrum.py`` 是 **唯一实现**。
合并前本模块有自己的内置 FFT 实现（``_run_power_spectrum_builtin``），
已于合并时删除（100+ 行冗余代码），现在直接委托给权威实现。
如果 twinbrain-dynamics 不可用，模块会抛出带说明的 ImportError。

输出文件（同 twinbrain-dynamics）
-----------------------------------
  power_spectrum_h.json      — 频段分析摘要
  power_spectrum_h.png       — 全局 PSD + 频段标注
  spatial_modes_h.png        — 频段地形图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical power spectrum implementation — required (no fallback)
try:
    from analysis.power_spectrum import run_power_spectrum_analysis as _td_run_psd
except ImportError as _err:
    raise ImportError(
        "spectral_dynamics.h_power_spectrum requires twinbrain-dynamics to be on "
        "sys.path.  The simplest fix is to import the spectral_dynamics package "
        "at the top level (which registers the path automatically):\n\n"
        "    import spectral_dynamics\n"
        "    from spectral_dynamics.h_power_spectrum import run_power_spectrum\n\n"
        "Alternatively, add the twinbrain-dynamics directory to sys.path manually "
        "before importing this module."
    ) from _err


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

    委托给 ``twinbrain-dynamics/analysis/power_spectrum.run_power_spectrum_analysis``。

    Args:
        trajectories: shape (n_traj, T, N)。
        dt:           时间步长（秒）；fMRI TR 通常为 2.0s。
        burnin:       跳过前几步（去除瞬态）。
        output_dir:   结果保存目录；None → 不保存。
        label:        输出文件名标签。

    Returns:
        dict 包含 freqs, mean_psd, region_psd, band_analysis, spatial_modes。
    """
    result = _td_run_psd(
        trajectories=trajectories,
        dt=dt,
        burnin=burnin,
        output_dir=output_dir / f"h_{label}" if output_dir else None,
    )

    # Save JSON summary directly in output_dir (for spectral_dynamics index)
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

