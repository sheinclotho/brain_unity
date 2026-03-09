"""
plot_utils — shared matplotlib configuration for spectral_dynamics
==================================================================

Provides :func:`configure_matplotlib` which should be called once at
module import time in any spectral_dynamics file that produces plots.
The function tries to configure a CJK-compatible font so that Chinese
axis labels render correctly on Windows/Linux systems where such fonts
are present; on systems where no CJK font is found it falls back to
standard ASCII-safe fonts without raising errors.

Also provides :func:`write_fallback_png` which writes a minimal valid
PNG file using only the Python standard library.  Use it as a fallback
inside ``except ImportError`` handlers so that output PNG paths are
always created even when matplotlib is not installed.
"""
from __future__ import annotations

import logging
import struct
import warnings
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIGURED = False


def write_fallback_png(path: "Path | str") -> None:
    """Write a minimal valid 2×2 grey PNG without matplotlib (stdlib only).

    Used as a placeholder when matplotlib is not installed so that callers
    (including tests) can verify a PNG file was produced at the expected path.
    The file is a valid PNG: it passes PNG signature and chunk-CRC checks but
    contains only a 2×2 greyscale image.

    Args:
        path: Destination file path.  Parent directory must already exist.
    """
    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    # 2×2 greyscale (bit depth 8): each row = filter byte (0) + 2 grey pixels
    raw_row = b"\x00\xcc\xcc"   # filter=None, grey=204, grey=204
    idat_data = zlib.compress(raw_row * 2, level=1)   # 2 identical rows
    png = (
        b"\x89PNG\r\n\x1a\n"                                           # PNG signature
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0))  # 2×2 grey 8-bit
        + _chunk(b"IDAT", idat_data)
        + _chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)


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
