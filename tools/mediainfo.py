"""ffprobe helpers: orientation, frame rate, HDR/colour classification, audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mkv", ".mts", ".m2ts", ".avi", ".hevc", ".webm", ".3gp", ".mxf"}
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".aif", ".aiff", ".caf", ".wma"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng", ".tif", ".tiff", ".webp"}

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "work", "output", "assets", "tools", "node_modules", "__pycache__"}


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise SystemExit(f"'{binary}' not found on PATH — install FFmpeg (with libass, libzimg, libx264).")
    return path


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def is_lfs_pointer(path: Path) -> bool:
    """True for a Git LFS pointer file (a ~130-byte text stub left when `git lfs pull` was not run)."""
    try:
        path = Path(path)
        return path.stat().st_size < 1024 and path.read_bytes()[:len(LFS_POINTER)] == LFS_POINTER
    except OSError:
        return False


def ffprobe(path: Path) -> dict:
    """ffprobe JSON (format + streams + stream side data).  Raises ValueError with a readable
    message (ffprobe's own error text, or a Git LFS hint) when the file cannot be probed."""
    require("ffprobe")
    path = Path(path)
    if is_lfs_pointer(path):
        raise ValueError(f"{path.name} is a Git LFS pointer, not the media itself — run `git lfs pull`")
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
             "-show_entries", "stream_side_data", str(path)], check=False)
    if p.returncode != 0:
        msg = (p.stderr or "").strip() or f"ffprobe exited with code {p.returncode}"
        raise ValueError(f"ffprobe cannot read {path.name}: {msg}")
    return json.loads(p.stdout)


def first_frame_side_data(path: Path) -> list[dict]:
    """Side data of the first video frame (HDR10 mastering display / content light level are
    often only in the bitstream SEI, not in the container, so -show_streams misses them)."""
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
             "-show_frames", "-print_format", "json", str(path)], check=False)
    try:
        frames = json.loads(p.stdout or "{}").get("frames") or []
    except json.JSONDecodeError:
        return []
    return list(frames[0].get("side_data_list") or []) if frames else []


def _frac(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    rot = (stream.get("tags") or {}).get("rotate")
    return int(rot) if rot and rot.lstrip("-").isdigit() else 0


def _dovi(stream: dict) -> dict | None:
    for sd in stream.get("side_data_list", []) or []:
        if "dv_profile" in sd or sd.get("side_data_type", "").lower().startswith("dovi"):
            return {k: sd.get(k) for k in ("dv_version_major", "dv_profile", "dv_level",
                                            "rpu_present_flag", "el_present_flag", "bl_present_flag",
                                            "dv_bl_signal_compatibility_id")}
    return None


def _num(v) -> float | None:
    try:
        return float(Fraction(str(v)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def hdr10_metadata(side_data: list[dict] | None) -> dict | None:
    """Mastering display max/min luminance and MaxCLL/MaxFALL (cd/m2) from ffprobe side data."""
    md: dict = {}
    for sd in side_data or []:
        kind = (sd.get("side_data_type") or "").lower()
        if kind == "mastering display metadata":
            md["mastering_max_nits"] = _num(sd.get("max_luminance"))
            md["mastering_min_nits"] = _num(sd.get("min_luminance"))
        elif kind == "content light level metadata":
            md["max_cll"] = _num(sd.get("max_content"))
            md["max_fall"] = _num(sd.get("max_average"))
    md = {k: v for k, v in md.items() if v}
    return md or None


def pq_peak_nits(md: dict | None) -> float | None:
    """Content peak for the PQ tone-map: MaxCLL (the brightest pixel actually in the content),
    else the mastering display's max luminance; implausible values are ignored, capped at 4000."""
    for key in ("max_cll", "mastering_max_nits"):
        v = (md or {}).get(key)
        if v and 100.0 <= v <= 10000.0:
            return round(min(v, 4000.0), 1)
    return None


# VUI transfers that carry no HDR information (a Dolby Vision compatibility id then decides)
_NO_HDR_TRC = ("", "unknown", "unspecified", "reserved", "bt2020-10", "bt2020-12")


def classify_color(stream: dict) -> dict:
    """Decide how the shot must be converted to SDR Rec.709.

    Dolby Vision is decided first: a profile 5 / compatibility-id 0 stream has an IPTPQc2 base
    layer that is not displayable as Y'CbCr, whatever its VUI transfer says.  For PQ, the
    content peak from HDR10 metadata in the stream's side_data_list is returned as 'peak_nits'
    (absent when there is none; render then assumes 1000)."""
    trc = (stream.get("color_transfer") or "").lower()
    prim = (stream.get("color_primaries") or "").lower()
    space = (stream.get("color_space") or "").lower()
    rng = (stream.get("color_range") or "").lower()
    pix = stream.get("pix_fmt") or ""
    codec_tag = (stream.get("codec_tag_string") or "").lower()
    dovi = _dovi(stream)
    compat = dovi.get("dv_bl_signal_compatibility_id") if dovi else None
    notes = []
    if dovi and (dovi.get("dv_profile") == 5 or compat == 0):
        transform = "dovi5"
        notes.append(f"Dolby Vision profile {dovi.get('dv_profile')}.{compat} has no backward-compatible base "
                     "layer (IPTPQc2) — needs a DV-aware decoder; export HLG/SDR from the phone or editor first.")
    elif trc == "arib-std-b67":
        transform = "hlg"
    elif trc == "smpte2084":
        transform = "pq"
    elif dovi and compat == 4 and trc in _NO_HDR_TRC:
        transform = "hlg"
        notes.append("No HDR transfer in the VUI; Dolby Vision compatibility id 4 says the base layer is HLG.")
    elif dovi and compat in (1, 6) and trc in _NO_HDR_TRC:
        transform = "pq"
        notes.append(f"No HDR transfer in the VUI; Dolby Vision compatibility id {compat} says the base layer "
                     "is HDR10 (PQ).")
    else:
        transform = "sdr709"
        if prim == "bt2020" and trc in _NO_HDR_TRC:
            notes.append("BT.2020 primaries without an HDR transfer: possibly Apple Log / camera log — verify visually.")
    if dovi and transform != "dovi5":
        notes.append(f"Dolby Vision profile {dovi.get('dv_profile')}.{compat} — using the "
                     f"{'HLG' if compat == 4 else 'HDR10' if compat in (1, 6) else 'base'} layer; RPU ignored.")
    md = hdr10_metadata(stream.get("side_data_list"))
    peak = pq_peak_nits(md) if transform == "pq" else None
    if transform == "pq":
        notes.append(f"PQ content peak {peak:g} nits from HDR10 metadata (tone-map white point)." if peak else
                     "PQ without usable HDR10 metadata — tone-mapping as if mastered to 1000 nits "
                     "(set the shot's hdr_peak_nits to override).")
    bit_depth = 10 if "10" in pix else 12 if "12" in pix else 8
    out = {
        "transform": transform,
        "hdr": transform in ("hlg", "pq", "dovi5"),
        "transfer": trc or "unknown",
        "primaries": prim or "unknown",
        "matrix": space or "unknown",
        "range": rng or "unknown",
        "pix_fmt": pix,
        "bit_depth": bit_depth,
        "dolby_vision": dovi,
        "hdr_metadata": md,
        "codec_tag": codec_tag,
        "notes": notes,
    }
    if peak:
        out["peak_nits"] = peak
    return out


def describe(path: Path) -> dict:
    """Condensed, decision-ready description of a media file."""
    path = Path(path)
    info = ffprobe(path)
    fmt = info.get("format", {})
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic")]
    astreams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    d: dict = {
        "path": str(path.relative_to(REPO)) if path.is_absolute() and REPO in path.parents else str(path),
        "size_mb": round(int(fmt.get("size", 0)) / 1e6, 1),
        "duration": round(float(fmt.get("duration", 0) or 0), 3),
        "container": fmt.get("format_name"),
        "creation_time": tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate"),
        "device": tags.get("com.apple.quicktime.model") or tags.get("model"),
        "video": None,
        "audio": None,
    }
    if vstreams:
        v = vstreams[0]
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = _rotation(v)
        dw, dh = (h, w) if abs(rot) % 180 == 90 else (w, h)
        r_fps = _frac(v.get("r_frame_rate"))
        avg_fps = _frac(v.get("avg_frame_rate"))
        nb = int(v.get("nb_frames") or 0)
        vdur = float(v.get("duration") or d["duration"] or 0)
        # avg_frame_rate and the stream duration honour a MOV/MP4 edit list; nb_frames counts every
        # sample in the track (pre-roll of a trimmed / passthrough export too), so derive from avg.
        est = int(round(avg_fps * vdur)) if avg_fps and vdur else 0
        frames = nb if nb and (not est or abs(nb - est) <= 1) else (est or nb)
        vfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) / r_fps > 0.01)
        color = classify_color(v)
        if color["transform"] == "pq" and not color["hdr_metadata"]:     # HDR10 metadata only in the SEI?
            extra = first_frame_side_data(path)
            if extra:
                color = classify_color(dict(v, side_data_list=list(v.get("side_data_list") or []) + extra))
        d["video"] = {
            "codec": v.get("codec_name"), "profile": v.get("profile"),
            "coded_size": [w, h], "rotation": rot, "display_size": [dw, dh],
            "orientation": "vertical" if dh > dw else "horizontal" if dw > dh else "square",
            "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
            "fps": round(avg_fps or r_fps, 3), "fps_nominal": fps_nominal(r_fps, avg_fps),
            "frames": frames, "duration": round(vdur, 3), "vfr": vfr,
            "bitrate_mbps": round(int(v.get("bit_rate", 0) or 0) / 1e6, 1),
            "sar": v.get("sample_aspect_ratio"), "field_order": v.get("field_order"),
            "color": color,
        }
    if astreams:
        a = astreams[0]
        d["audio"] = {
            "codec": a.get("codec_name"), "channels": a.get("channels"),
            "layout": a.get("channel_layout"), "sample_rate": int(a.get("sample_rate", 0) or 0),
            "bitrate_kbps": round(int(a.get("bit_rate", 0) or 0) / 1e3),
            "duration": round(float(a.get("duration") or d["duration"] or 0), 3),
            "streams": len(astreams),
        }
    return d


NTSC_FPS = (23.976, 29.97, 47.952, 59.94, 119.88, 239.76)
INTEGER_FPS = (24, 25, 30, 48, 50, 60, 100, 120, 240)


def _exact(n: float) -> float:
    return float(n) if n in INTEGER_FPS else round(n * 1.001) / 1.001     # 29.97 -> 30000/1001


def nominal_fps(fps: float) -> float:
    """Snap a measured rate to the standard rate it was recorded at.  NTSC (x/1001) only when the
    rate matches it to 0.02 % (cameras that record 29.97 are CFR and exact); anything else within
    2 % goes to the integer rate, so a 30p phone clip averaging 29.95 after dropped frames is 30."""
    if not fps:
        return 0.0
    for n in NTSC_FPS:
        if abs(fps - _exact(n)) <= 2e-4 * fps:
            return n
    best = min(INTEGER_FPS, key=lambda n: abs(fps - n))
    return best if abs(fps - best) < 0.02 * best else round(fps, 3)


def fps_nominal(r_fps: float, avg_fps: float) -> float:
    """Nominal rate of a stream: r_frame_rate when it is exactly a standard rate and the average
    is within 2 % of it (a VFR phone clip keeps its camera rate), else the snapped average."""
    if r_fps and avg_fps and abs(avg_fps - r_fps) < 0.02 * r_fps:
        n = nominal_fps(r_fps)
        if (n in INTEGER_FPS or n in NTSC_FPS) and abs(r_fps - _exact(n)) <= 2e-4 * r_fps:
            return n
    return nominal_fps(avg_fps or r_fps)


def find_media(roots: list[Path] | None = None) -> dict[str, list[Path]]:
    """Media files by extension only; callers that probe them (inventory) move an audio-only
    file in a video container (.mp4/.mov nasheed) to audio."""
    roots = roots or [REPO]
    found = {"video": [], "audio": [], "image": []}
    for root in roots:
        for p in sorted(Path(root).rglob("*")):
            if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
                continue
            if p.name.startswith("._"):
                continue
            ext = p.suffix.lower()
            if ext in VIDEO_EXT:
                found["video"].append(p)
            elif ext in AUDIO_EXT:
                found["audio"].append(p)
            elif ext in IMAGE_EXT:
                found["image"].append(p)
    return found


def to_rgb_filter(color: dict, out_fmt: str = "gbrpf32le") -> str:
    """zscale step: camera Y'CbCr -> R'G'B' with NO transfer/primaries change (LUT does the rest)."""
    m = {"bt709": "709", "bt2020nc": "2020_ncl", "bt2020c": "2020_cl", "smpte170m": "170m",
         "bt470bg": "470bg"}.get(color.get("matrix"), None)
    if m is None:
        m = "2020_ncl" if color["transform"] in ("hlg", "pq") else "709"
    rin = "full" if color.get("range") == "pc" else "limited"
    # Tag the frame as linear-agnostic: transfer/primaries "input"->"input" means no conversion.
    return f"zscale=matrixin={m}:rangein={rin}:filter=spline36,format={out_fmt}"


def from_rgb_filter(out_pix: str = "yuv422p10le") -> str:
    """R'G'B' already BT.709-encoded by the LUT -> Y'CbCr BT.709 limited range, tagged."""
    return ("zscale=transferin=709:primariesin=709:rangein=full:"
            "transfer=709:primaries=709:matrix=709:range=limited:dither=error_diffusion,"
            f"format={out_pix},setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(json.dumps(describe(Path(arg)), indent=2, ensure_ascii=False))
