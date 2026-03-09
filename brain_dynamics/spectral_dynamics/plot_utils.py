"""
plot_utils — shared matplotlib configuration for spectral_dynamics
==================================================================

Provides :func:`configure_matplotlib` which should be called once at
module import time in any spectral_dynamics file that produces plots.
The function tries to configure a CJK-compatible font so that Chinese
axis labels render correctly on Windows/Linux systems where such fonts
are present; on systems where no CJK font is found it falls back to
standard ASCII-safe fonts without raising errors.
"""
from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure_matplotlib() -> None:
    """
    Configure matplotlib for consistent, warning-free rendering.

    Sets ``rcParams['font.sans-serif']`` to a font list that includes
    CJK-capable fonts (when available) so that Chinese axis labels are
    rendered without "Glyph NNN missing from font(s) DejaVu Sans"
    UserWarnings.  Also sets ``axes.unicode_minus = False`` to ensure
    minus signs render correctly.

    This function is idempotent; subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    try:
        import matplotlib
        import matplotlib.font_manager as fm

        # Fonts tried in order: common CJK-capable fonts on Windows/Linux/macOS.
        _CJK_FONT_CANDIDATES = [
            "Microsoft YaHei",       # Windows
            "SimHei",                # Windows (simplified Chinese)
            "SimSun",                # Windows
            "PingFang SC",           # macOS
            "Hiragino Sans GB",      # macOS
            "Noto Sans CJK SC",      # Linux (noto-cjk package)
            "Noto Sans CJK",         # Linux variant
            "WenQuanYi Micro Hei",   # Linux
            "WenQuanYi Zen Hei",     # Linux
            "AR PL UMing CN",        # Linux
            "Source Han Sans SC",    # Adobe
            "DejaVu Sans",           # fallback (no CJK, but safe)
        ]

        available = {f.name for f in fm.fontManager.ttflist}
        ordered = [f for f in _CJK_FONT_CANDIDATES if f in available]

        if not ordered:
            ordered = ["DejaVu Sans", "sans-serif"]

        matplotlib.rcParams["font.sans-serif"] = ordered
        matplotlib.rcParams["axes.unicode_minus"] = False

        has_cjk = any(
            f in available
            for f in _CJK_FONT_CANDIDATES
            if f not in ("DejaVu Sans", "sans-serif")
        )
        if has_cjk:
            logger.debug("configure_matplotlib: CJK font '%s' activated.", ordered[0])
        else:
            logger.debug(
                "configure_matplotlib: no CJK font found; "
                "Chinese labels will use ASCII-safe English equivalents."
            )

    except ImportError as exc:
        logger.warning(
            "configure_matplotlib: matplotlib not available (%s). "
            "CJK font setup skipped.", exc
        )

    _CONFIGURED = True
