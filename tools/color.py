"""Colour science for the Madinah reel pipeline.

Every shot is graded through ONE 3D LUT that is generated from parameters, so the
whole grade is reproducible from edit/edl.json:

    camera R'G'B'  --input transform-->  display-linear BT.709 (SDR-relative)
                   --shot balance----->  exposure (stops) + white balance (linear gains)
                   --look------------->  contrast, highlight shoulder, warmth, saturation
                   --output encode---->  BT.709 / BT.1886 R'G'B' (gamma 1/2.4)

Working domain ("SDR display-linear"): 1.0 = SDR peak (100 %), diffuse white of the
scene sits at REF_WHITE_LIN (= 90 % signal), exactly as an SDR camera would put it.

Supported inputs:
  sdr709   BT.709 / sRGB-like camera footage (decoded with the BT.1886 EOTF, gamma 2.4)
  hlg      BT.2100 HLG, BT.2020 primaries (iPhone HDR/Dolby Vision 8.4 base layer, Android HLG)
  pq       BT.2100 PQ (HDR10 / Dolby Vision 8.1 base layer), BT.2020 primaries

HDR conversion follows ITU-R BT.2100 / BT.2408 (HDR reference white = 203 cd/m2),
gamut-maps BT.2020 -> BT.709 with soft desaturation instead of hard clipping and
rolls highlights off with a C1-continuous shoulder so bright marble, sky and lamps
keep their texture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
REF_WHITE_NITS = 203.0                 # BT.2408 HDR reference (diffuse) white
REF_WHITE_SIGNAL = 0.90                # where diffuse white lands in SDR signal
REF_WHITE_LIN = REF_WHITE_SIGNAL ** 2.4  # ~0.776 SDR display-linear
SDR_GAMMA = 2.4                        # BT.1886
HDR_KNEE = 0.70                        # HDR->SDR shoulder start (SDR-linear; ~86 % signal)

# HLG (BT.2100)
HLG_A = 0.17883277
HLG_B = 1.0 - 4.0 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4.0 * HLG_A)

# PQ (SMPTE ST 2084)
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0

LUMA_2020 = np.array([0.2627, 0.6780, 0.0593])
LUMA_709 = np.array([0.2126, 0.7152, 0.0722])

# Linear-light BT.2020 RGB -> BT.709 RGB (same D65 white)
M_2020_TO_709 = np.array([
    [1.660491, -0.587641, -0.072850],
    [-0.124550, 1.132900, -0.008349],
    [-0.018151, -0.100579, 1.118730],
])


# ---------------------------------------------------------------- transfer functions
def hlg_inverse_oetf(ep: np.ndarray) -> np.ndarray:
    ep = np.clip(ep, 0.0, 1.0)
    low = ep * ep / 3.0
    high = (np.exp((ep - HLG_C) / HLG_A) + HLG_B) / 12.0
    return np.where(ep <= 0.5, low, high)


def hlg_to_display_nits(rgb_signal: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    """HLG R'G'B' (BT.2020) -> display light in cd/m2, BT.2020 primaries (BT.2100 OOTF)."""
    scene = hlg_inverse_oetf(rgb_signal)
    gamma = 1.2 + 0.42 * math.log10(peak_nits / 1000.0)
    ys = np.tensordot(scene, LUMA_2020, axes=([-1], [0]))[..., None]
    return peak_nits * np.power(np.maximum(ys, 1e-12), gamma - 1.0) * scene


def pq_to_display_nits(rgb_signal: np.ndarray) -> np.ndarray:
    ep = np.clip(rgb_signal, 0.0, 1.0)
    p = np.power(ep, 1.0 / PQ_M2)
    return 10000.0 * np.power(np.maximum(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), 1.0 / PQ_M1)


# ---------------------------------------------------------------- tone + gamut
def shoulder(x: np.ndarray, knee: float, x_max: float) -> np.ndarray:
    """Identity below `knee`; above it a C1-continuous rational roll-off that maps
    x_max exactly to 1.0 (and keeps rising gently, asymptotically, beyond)."""
    t_max = max(x_max - knee, 1e-6)
    room = 1.0 - knee
    if t_max <= room:            # nothing to compress
        return np.minimum(x, 1.0)
    a = room * t_max / (t_max - room)
    t = np.maximum(x - knee, 0.0)
    rolled = knee + room * (t / (t + a)) * ((t_max + a) / t_max)
    return np.where(x <= knee, x, rolled)


def tone_map_rgb(rgb: np.ndarray, knee: float, x_max: float, hue_preserve: float = 0.6) -> np.ndarray:
    """Roll highlights off.  Mix of max-RGB (hue/saturation preserving) and per-channel
    (natural highlight desaturation, like film)."""
    mx = np.max(rgb, axis=-1, keepdims=True)
    mx_t = shoulder(mx, knee, x_max)
    ratio = np.where(mx > 1e-9, mx_t / np.maximum(mx, 1e-9), 1.0)
    by_max = rgb * ratio
    per_ch = shoulder(rgb, knee, x_max)
    return hue_preserve * by_max + (1.0 - hue_preserve) * per_ch


def soft_gamut_clip(rgb: np.ndarray, luma=LUMA_709) -> np.ndarray:
    """Pull out-of-gamut (negative) components back toward the pixel's luminance
    instead of hard clipping (avoids hue shifts / banding in saturated lights)."""
    y = np.tensordot(rgb, luma, axes=([-1], [0]))[..., None]
    y = np.maximum(y, 0.0)
    mn = np.min(rgb, axis=-1, keepdims=True)
    # fraction toward grey needed so the minimum component becomes >= 0
    need = np.where(mn < 0.0, (0.0 - mn) / np.maximum(y - mn, 1e-9), 0.0)
    need = np.clip(need, 0.0, 1.0)
    return np.maximum(y + (rgb - y) * (1.0 - need), 0.0)


# ---------------------------------------------------------------- input transforms
def input_to_working(rgb_signal: np.ndarray, kind: str, hdr_peak_nits: float = 1000.0,
                     hlg_display_nits: float = 1000.0) -> np.ndarray:
    """Camera R'G'B' in [0,1] -> SDR display-linear BT.709 working space (unclipped above 1).

    hdr_peak_nits: PQ content peak (MaxCLL / mastering max), used only to place the HDR tone-map.
    hlg_display_nits: nominal HLG display peak Lw for the BT.2100 OOTF (BT.2408 reference = 1000).
    """
    if kind == "sdr709":
        return np.power(np.clip(rgb_signal, 0.0, 1.0), SDR_GAMMA)
    if kind == "hlg":
        nits = hlg_to_display_nits(rgb_signal, hlg_display_nits)
    elif kind == "pq":
        nits = pq_to_display_nits(rgb_signal)
    else:
        raise ValueError(f"unknown input transform {kind!r}")
    rel = nits / REF_WHITE_NITS * REF_WHITE_LIN          # diffuse white -> 90 % SDR signal
    rgb709 = np.tensordot(rel, M_2020_TO_709.T, axes=([-1], [0]))
    rgb709 = soft_gamut_clip(rgb709)
    return rgb709


# ---------------------------------------------------------------- grade parameters
@dataclass
class ShotGrade:
    exposure: float = 0.0     # stops, applied in linear light
    temperature: float = 0.0  # -1 (cooler) .. +1 (warmer); ~0.1 = subtle
    tint: float = 0.0         # -1 (green) .. +1 (magenta)
    saturation: float = 1.0   # per-shot multiplier on top of the look
    contrast: float = 0.0     # per-shot extra contrast (-0.5 .. 0.5)
    black_crush: float = 0.0  # deepen the toe (SDR-linear units x100, e.g. 0.05)
    black_lift: float = 0.0   # lift the toe (SDR-linear units x100, e.g. 0.05)
    highlights: float = 0.0   # -1 recover .. +1 brighten (acts above mid-grey)


@dataclass
class Look:
    contrast: float = 0.14          # S-curve strength around mid-grey (log domain)
    pivot: float = 0.18             # mid-grey in working linear
    knee: float = 0.62              # highlight shoulder start (working linear, SDR)
    warmth: float = 0.035           # mid/shadow warm split-tone, highlights stay neutral
    shadow_tint: float = 0.0        # +: teal shadows, -: warm shadows (kept ~0 for realism)
    saturation: float = 1.06        # global saturation
    vibrance: float = 0.12          # extra saturation for muted colours
    skin_protect: float = 0.7       # 0..1, how much vibrance/saturation spares skin hues
    black_floor: float = 0.0004     # smooth toe so blacks are deep but not crushed


def _wb_gains(temperature: float, tint: float) -> np.ndarray:
    # Small-signal approximation of a CCT/tint shift as von Kries gains in linear RGB.
    r = 1.0 + 0.30 * temperature + 0.05 * tint
    g = 1.0 - 0.18 * tint
    b = 1.0 - 0.30 * temperature + 0.05 * tint
    gains = np.array([r, g, b])
    return gains / float(np.dot(gains, LUMA_709))     # keep luminance constant


def _hue_sat(rgb: np.ndarray):
    y = np.tensordot(rgb, LUMA_709, axes=([-1], [0]))[..., None]
    chroma = rgb - y
    mx = np.max(rgb, axis=-1, keepdims=True)
    mn = np.min(rgb, axis=-1, keepdims=True)
    sat = (mx - mn) / np.maximum(mx, 1e-9)
    return y, chroma, sat


def _skin_weight(rgb: np.ndarray) -> np.ndarray:
    """~1 for typical skin hues (orange-ish, moderate saturation), 0 elsewhere."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = np.maximum(mx - mn, 1e-9)
    # hue in degrees (HSV)
    h = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    s = (mx - mn) / np.maximum(mx, 1e-9)
    hue_w = np.exp(-0.5 * ((h - 25.0) / 12.0) ** 2)
    sat_w = np.exp(-0.5 * ((s - 0.38) / 0.22) ** 2)
    return (hue_w * sat_w)[..., None]


def apply_grade(lin: np.ndarray, shot: ShotGrade, look: Look) -> np.ndarray:
    """Working linear -> graded working linear (still linear, may exceed 1)."""
    x = lin * (2.0 ** shot.exposure)
    x = x * _wb_gains(shot.temperature, shot.tint)

    # --- tone: log-domain S-curve around the pivot, plus per-shot contrast & highlights
    c = look.contrast + shot.contrast
    eps = 1e-5
    y = np.tensordot(x, LUMA_709, axes=([-1], [0]))[..., None]
    ly = np.log2(np.maximum(y, eps) / look.pivot)
    # smooth S: slope (1+c) at the pivot, easing back toward 1 far from it
    ly2 = ly * (1.0 + c / (1.0 + (ly / 3.5) ** 2))
    if shot.highlights:
        ly2 = ly2 + shot.highlights * 0.35 * np.clip(ly / 2.5, 0.0, 1.0) ** 1.5
    y2 = look.pivot * np.power(2.0, ly2)
    gain = np.where(y > eps, y2 / np.maximum(y, eps), 1.0)
    x = x * gain

    # --- toe: tiny black floor / per-shot lift-crush (in linear)
    floor = look.black_floor + shot.black_crush / 100.0
    if floor > 0:      # smooth toe: ~x - floor for bright values, gently to 0 at black
        x = x * x / (x + floor)
    lift = shot.black_lift / 100.0
    if lift > 0:
        x = x + lift * (1.0 - np.minimum(x, 1.0))

    # --- warm split-tone: affects shadows/mids, fades out in highlights (clean whites)
    yv = np.tensordot(x, LUMA_709, axes=([-1], [0]))[..., None]
    mid_w = np.clip(1.0 - (yv / REF_WHITE_LIN) ** 1.5, 0.0, 1.0)
    warm = look.warmth * mid_w
    x = x * np.concatenate([1.0 + warm, np.ones_like(warm) + 0.15 * warm, 1.0 - warm * 1.2], axis=-1)
    if look.shadow_tint:
        sh_w = np.clip(1.0 - yv / 0.05, 0.0, 1.0) * look.shadow_tint
        x = x * np.concatenate([1.0 - sh_w, np.ones_like(sh_w), 1.0 + sh_w], axis=-1)

    # --- saturation + vibrance with skin protection (done in gamma domain for perceptual evenness)
    g = np.power(np.maximum(x, 0.0), 1.0 / SDR_GAMMA)
    yg, chroma, sat = _hue_sat(g)
    skin = _skin_weight(g) * look.skin_protect
    vib = look.vibrance * (1.0 - np.clip(sat, 0.0, 1.0)) * (1.0 - skin)
    s_total = (look.saturation * shot.saturation - 1.0) * (1.0 - 0.5 * skin) + 1.0 + vib
    g = yg + chroma * s_total
    x = np.power(np.maximum(g, 0.0), SDR_GAMMA)

    # --- highlight shoulder: keep marble/sky detail, never hard-clip
    x = tone_map_rgb(x, look.knee, x_max=max(1.05, float(np.max(x))))
    return np.clip(x, 0.0, 1.0)


def working_to_output(lin: np.ndarray) -> np.ndarray:
    return np.power(np.clip(lin, 0.0, 1.0), 1.0 / SDR_GAMMA)


# ---------------------------------------------------------------- LUT generation
def build_lut(kind: str, shot: ShotGrade | None = None, look: Look | None = None,
              size: int = 65, hdr_peak_nits: float = 1000.0, neutral: bool = False,
              hlg_display_nits: float = 1000.0) -> np.ndarray:
    """Return a (size, size, size, 3) LUT indexed [b, g, r] (cube file order).

    hdr_peak_nits applies to PQ input only; hlg_display_nits to HLG input only."""
    shot = shot or ShotGrade()
    look = look or Look()
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    lin = input_to_working(rgb, kind, hdr_peak_nits, hlg_display_nits)
    if kind != "sdr709":
        peak = hlg_display_nits if kind == "hlg" else hdr_peak_nits
        # HDR: compress highlights above the knee into SDR range before grading
        lin = tone_map_rgb(lin, knee=HDR_KNEE, x_max=(peak / REF_WHITE_NITS) * REF_WHITE_LIN)
    if not neutral:
        lin = apply_grade(lin, shot, look)
    out = working_to_output(lin)
    return out.reshape(size, size, size, 3)


def write_cube(path: Path, lut: np.ndarray, title: str = "madinah") -> Path:
    size = lut.shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = lut.reshape(-1, 3)
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        np.savetxt(f, flat, fmt="%.6f")
    return path


def shot_grade_from_dict(d: dict | None) -> ShotGrade:
    d = d or {}
    return ShotGrade(**{k: v for k, v in d.items() if k in ShotGrade.__dataclass_fields__})


def look_from_dict(d: dict | None) -> Look:
    d = d or {}
    return Look(**{k: v for k, v in d.items() if k in Look.__dataclass_fields__})


if __name__ == "__main__":
    # Self-check: key reference points.
    hlg75 = np.array([[0.75, 0.75, 0.75]])
    nits = hlg_to_display_nits(hlg75)[0, 0]
    print(f"HLG 75% grey -> {nits:.1f} nits (expect ~203)")
    for kind in ("hlg", "pq"):
        lut = build_lut(kind, neutral=True, size=33)
    sig = {"hlg": 0.75, "pq": 0.5807}  # PQ 203 nits ~ 58 %
    for kind, v in sig.items():
        lin = input_to_working(np.array([[v, v, v]]), kind)
        lin = tone_map_rgb(lin, HDR_KNEE, (1000 / REF_WHITE_NITS) * REF_WHITE_LIN)
        print(f"{kind} ref-white -> SDR signal {working_to_output(lin)[0, 0]:.3f} (expect ~0.90)")
    peak = input_to_working(np.array([[1.0, 1.0, 1.0]]), "hlg")
    peak = tone_map_rgb(peak, HDR_KNEE, (1000 / REF_WHITE_NITS) * REF_WHITE_LIN)
    print(f"HLG peak -> SDR signal {working_to_output(peak)[0, 0]:.3f} (expect 1.000)")
    ident = build_lut("sdr709", neutral=True, size=17)
    ax = np.linspace(0, 1, 17)
    print("SDR neutral LUT max identity error:", float(np.abs(ident[..., 0] - ax[None, None, :]).max()))
    graded = build_lut("sdr709", size=17)
    print("graded mid-grey (0.46 signal) ->", graded[8, 8, 8].round(3), " white ->", graded[-1, -1, -1].round(3))
