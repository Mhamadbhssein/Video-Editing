"""Small FFmpeg helpers shared by render.py, qc.py and review.py.

esc(path)      -> a path escaped for an FFmpeg *filtergraph* option value.  Use it WITHOUT
                  surrounding quotes:  f"lut3d=file={esc(lut)}:interp=tetrahedral".
fps_rate(fps)  -> the exact rational FFmpeg expects for a frame rate (29.97 -> "30000/1001").
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

_NTSC = {23.976: "24000/1001", 29.97: "30000/1001", 47.952: "48000/1001",
         59.94: "60000/1001", 119.88: "120000/1001"}


def esc(path: Path | str) -> str:
    """Two-level escape: first for the filter's option parser (\\ ' :), then for the
    filtergraph parser (\\ ' [ ] , ;).  Works for apostrophes, colons, commas, brackets,
    spaces and Windows drive letters (C:\\...)."""
    s = str(path)
    for c in "\\':":
        s = s.replace(c, "\\" + c)
    for c in "\\'[],;":
        s = s.replace(c, "\\" + c)
    return s


def fps_rate(fps: float | int | str) -> str:
    """Exact frame-rate string for fps=/framerate=/minterpolate/-r (NTSC rates as x/1001)."""
    if isinstance(fps, str):
        if "/" in fps:
            return fps
        fps = float(fps)
    key = round(float(fps), 3)
    if key in _NTSC:
        return _NTSC[key]
    f = Fraction(float(fps)).limit_denominator(1001)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def fps_fraction(fps: float | int | str) -> Fraction:
    """The same rate as an exact Fraction (for frame <-> time arithmetic)."""
    return Fraction(fps_rate(fps))
