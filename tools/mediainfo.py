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


def ffprobe(path: Path) -> dict:
    require("ffprobe")
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
               "-show_entries", "stream_side_data", str(path)]).stdout
    return json.loads(out)


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


def classify_color(stream: dict) -> dict:
    """Decide how the shot must be converted to SDR Rec.709."""
    trc = (stream.get("color_transfer") or "").lower()
    prim = (stream.get("color_primaries") or "").lower()
    space = (stream.get("color_space") or "").lower()
    rng = (stream.get("color_range") or "").lower()
    pix = stream.get("pix_fmt") or ""
    codec_tag = (stream.get("codec_tag_string") or "").lower()
    dovi = _dovi(stream)
    notes = []
    if trc in ("arib-std-b67",):
        transform = "hlg"
    elif trc in ("smpte2084",):
        transform = "pq"
    elif dovi and dovi.get("dv_profile") == 5:
        transform = "dovi5"
        notes.append("Dolby Vision profile 5 (no backward-compatible base layer) — needs a DV-aware decoder.")
    else:
        transform = "sdr709"
        if prim == "bt2020" and trc in ("", "unknown", "bt2020-10", "bt2020-12"):
            notes.append("BT.2020 primaries without an HDR transfer: possibly Apple Log / camera log — verify visually.")
    if dovi:
        compat = dovi.get("dv_bl_signal_compatibility_id")
        notes.append(f"Dolby Vision profile {dovi.get('dv_profile')}.{compat} — using the "
                     f"{'HLG' if compat == 4 else 'HDR10' if compat == 1 else 'base'} layer; RPU ignored.")
    bit_depth = 10 if "10" in pix else 12 if "12" in pix else 8
    return {
        "transform": transform,
        "hdr": transform in ("hlg", "pq", "dovi5"),
        "transfer": trc or "unknown",
        "primaries": prim or "unknown",
        "matrix": space or "unknown",
        "range": rng or "unknown",
        "pix_fmt": pix,
        "bit_depth": bit_depth,
        "dolby_vision": dovi,
        "codec_tag": codec_tag,
        "notes": notes,
    }


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
        measured = nb / vdur if nb and vdur else avg_fps
        vfr = bool(r_fps and measured and abs(r_fps - measured) / r_fps > 0.01)
        d["video"] = {
            "codec": v.get("codec_name"), "profile": v.get("profile"),
            "coded_size": [w, h], "rotation": rot, "display_size": [dw, dh],
            "orientation": "vertical" if dh > dw else "horizontal" if dw > dh else "square",
            "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
            "fps": round(avg_fps or r_fps, 3), "fps_nominal": nominal_fps(avg_fps or r_fps),
            "frames": nb, "duration": round(vdur, 3), "vfr": vfr,
            "bitrate_mbps": round(int(v.get("bit_rate", 0) or 0) / 1e6, 1),
            "sar": v.get("sample_aspect_ratio"), "field_order": v.get("field_order"),
            "color": classify_color(v),
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


def nominal_fps(fps: float) -> float:
    cands = (23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 100, 119.88, 120, 239.76, 240)
    best = min(cands, key=lambda n: abs(fps - n))
    return best if abs(fps - best) < 0.02 * best else round(fps, 3)


def find_media(roots: list[Path] | None = None) -> dict[str, list[Path]]:
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
