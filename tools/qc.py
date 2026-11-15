#!/usr/bin/env python3
"""Step 5 — quality control of the exported reel.

    python3 tools/qc.py [VIDEO] [--edl edit/edl.json]

VIDEO defaults to the first file in work/render/timeline.json "outputs" (what render.py just wrote),
else output/<export.name>.  tools/run_all.sh checks every output, the 4K master included.

Checks (report in work/qc/report.md + report.json, or report_<name>.* for the other outputs such as
the master; non-zero exit on hard failures):
  * delivery spec: H.264 High, level vs. size/fps, yuv420p, size, CFR fps, frame count, BT.709 limited
    tags in the SPS VUI *and* the MP4 colr atom, AAC 48 kHz stereo >= 128 kb/s, faststart,
    audio vs. video duration and start time
  * black frames (single flash frames too), freezes and duplicate-frame stutter inside moving shots
    (the repeats a real-frame conform makes by design are allowed), isolated stray/flash frames,
    unexpected or misplaced cuts vs. the timeline
  * A/V offset (the reel's audio cross-correlated with work/render/mix.wav), clicks at the audio
    edges, EBU R128 loudness vs. the EDL target, true peak, full-scale samples, hard-clipped sources,
    unintended silences, music ending inside a recited/sung phrase (work/audio/<name>.json)
  * title boxes inside the Instagram-safe area (titles rebuilt from the EDL being checked; a title
    that cannot be rendered is a failure)
and writes inspection sheets (cuts, titles + safe-zone overlay, slow-motion frame strips,
first/last frame) to work/qc/ for visual review.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediainfo import REPO, ffprobe, require  # noqa: E402
from ffutil import esc  # noqa: E402
from render import FONTS, SAFE, build_ass, h264_level  # noqa: E402
from review import font, tile  # noqa: E402

WORK = REPO / "work"
QC = WORK / "qc"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"      # FFmpeg logs %g: 7.28202e-05 is a number too
SR = 48000


def moov_first(path: Path) -> bool:
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return False
            size, typ = struct.unpack(">I4s", hdr)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                f.seek(size - 16, 1)
            elif size == 0:
                return False
            else:
                f.seek(size - 8, 1)
            if typ == b"moov":
                return True
            if typ == b"mdat":
                return False


def read_moov(path: Path) -> bytes:
    """The raw moov box, wherever it sits in the file (b'' if there is none)."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return b""
            size, typ = struct.unpack(">I4s", hdr)
            hl = 8
            if size == 1:
                size, hl = struct.unpack(">Q", f.read(8))[0], 16
            elif size == 0:
                return f.read() if typ == b"moov" else b""
            if typ == b"moov":
                return f.read(size - hl)
            f.seek(size - hl, 1)


def colr_nclx(path: Path) -> list[int] | None:
    """[primaries, transfer, matrix, full_range] of the video sample entry's colr/nclx atom, or None.
    ffprobe reports a merged colour (the VUI when it is set, else this atom), so it sees neither alone."""
    moov = read_moov(path)
    i = moov.find(b"colrnclx")
    if i < 0 or len(moov) < i + 15:
        return None
    p, t, m = struct.unpack(">HHH", moov[i + 8:i + 14])
    return [p, t, m, moov[i + 14] >> 7]


def sps_vui(path: Path) -> dict:
    """Fields of the first H.264 SPS as trace_headers prints them (level_idc,
    video_signal_type_present_flag, video_full_range_flag, colour_description_present_flag,
    colour_primaries, transfer_characteristics, matrix_coefficients, ...)."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-v", "info", "-i", str(path), "-map", "0:v:0", "-c:v", "copy",
                        "-bsf:v", "trace_headers", "-frames:v", "1", "-f", "null", "-"],
                       stderr=subprocess.PIPE, text=True)
    out: dict = {}
    for k, val in re.findall(r"\]\s+\d+\s+(\w+)\s+[01]+ = (-?\d+)", p.stderr):
        out.setdefault(k, int(val))
    return out


def decode_audio(path: Path, seconds: float | None = None) -> np.ndarray:
    """First audio stream as float32 stereo at 48 kHz, shape (n, 2).  An MP4 without an edit list
    decodes with the AAC encoder priming in front; render pre-advances the mix by exactly that, so
    sample n of the decoded reel lines up with sample n of work/render/mix.wav."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)] + (["-t", f"{seconds:.3f}"] if seconds else [])
    cmd += ["-map", "0:a:0", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw[:len(raw) // 8 * 8], np.float32).reshape(-1, 2)


def av_lag(reel: np.ndarray, ref: np.ndarray, max_lag: int = SR // 2) -> tuple[int, float] | None:
    """Lag in samples (+ = the reel's audio is late) and normalised correlation of the reel's audio
    against the mix it was encoded from, searched over +-max_lag.  None if the reference is silent."""
    n = min(len(reel), len(ref))
    a = reel[:n].astype(np.float64)
    b = ref[:n].astype(np.float64)
    if n < SR // 10 or np.sqrt(np.mean(b * b)) < 1e-4:
        return None
    max_lag = min(max_lag, n // 4)
    nfft = 1 << (2 * n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    lags = np.r_[0:max_lag + 1, -max_lag:0]
    vals = np.r_[c[:max_lag + 1], c[nfft - max_lag:]]
    k = int(np.argmax(vals))
    return int(lags[k]), float(vals[k] / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def flat_peak_runs(x: np.ndarray, min_run: int = 3) -> int:
    """Plateaus at a channel's own peak level: runs of >= min_run identical consecutive samples within
    0.01 dB of it (astats' Peak_count/Flat_factor idea).  The signature of a hard-clipped lossless
    source, even after it was turned down.  (Lossy sources smear the plateaus; they go unseen.)"""
    runs = 0
    for ch in np.atleast_2d(x.T):
        pk = float(np.abs(ch).max()) if ch.size else 0.0
        if pk <= 0:
            continue
        m = (np.abs(ch[1:]) >= pk * 0.99885) & (ch[1:] == ch[:-1])
        d = np.diff(np.r_[0, m.astype(np.int8), 0])
        runs += int(((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) >= min_run - 1).sum())
    return runs


def dup_frames(seg: np.ndarray, W: int = 3, k: float = 0.05, floor: float = 0.4) -> list[int]:
    """Repeated frames inside a shot: a near-zero YDIF between frames that move on both sides.  The
    reference is local (the smaller of the largest YDIF among the W frames before and the W after),
    so a 3x stutter (m,0,0,m,0,0,...) is caught whatever the shot median, and a locked-off stretch
    with no motion around it is never flagged.  Holds longer than ~2W frames are freezedetect's job."""
    out = []
    for i in range(1, len(seg) - 1):
        ref = min(seg[max(0, i - W):i].max(), seg[i + 1:i + 1 + W].max())
        if ref > floor and seg[i] < k * ref:
            out.append(i)
    return out


def flash_frames(seg: np.ndarray) -> list[tuple[int, int]]:
    """Isolated intrusions of 1-2 frames (a flash, a black frame, a stray frame from another shot):
    YDIF jumps far above the shot's own motion going in and coming out, with ordinary motion on both
    sides.  A whip pan or a burst of fast motion is several high values in a row and is not flagged.
    Returns (index of the first foreign frame, number of foreign frames)."""
    if seg.size < 6:
        return []
    p90 = float(np.percentile(seg, 90))
    hi, lo = max(8.0, 3.0 * p90), 1.5 * p90
    out, i = [], 1
    while i < len(seg) - 2:
        if seg[i] > hi and seg[i - 1] <= lo:
            if seg[i + 1] > hi and seg[i + 2] <= lo:
                out.append((i, 1))
                i += 3
                continue
            if i + 3 < len(seg) and seg[i + 1] <= hi and seg[i + 2] > hi and seg[i + 3] <= lo:
                out.append((i, 2))
                i += 4
                continue
        i += 1
    return out


def lavfi_log(path: Path, vf: str | None = None, af: str | None = None) -> str:
    cmd = ["ffmpeg", "-v", "info", "-nostats", "-i", str(path)]
    if vf:
        cmd += ["-vf", vf]
    else:
        cmd += ["-vn"]
    if af:
        cmd += ["-af", af]
    else:
        cmd += ["-an"]
    cmd += ["-f", "null", "-"]
    return subprocess.run(cmd, stderr=subprocess.PIPE, text=True).stderr


def frames_at(path: Path, idx: list[int], w: int, h: int) -> dict[int, np.ndarray]:
    idx = sorted(set(i for i in idx if i >= 0))
    if not idx:
        return {}
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='{sel}',scale={w}:{h}:flags=bicubic",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    n = w * h * 3
    out = {}
    for k, i in enumerate(idx):
        chunk = raw[k * n:(k + 1) * n]
        if len(chunk) == n:
            out[i] = np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
    return out


def per_frame_ydif(path: Path) -> np.ndarray:
    """Mean |Y(n) - Y(n-1)| per frame (0 for the first).  metadata=print writes %g, so the tiny values
    of repeated frames come out as e.g. 7.28202e-05: parse the whole number, exponent included."""
    log = QC / "ydif.txt"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                    f"signalstats,metadata=print:key=lavfi.signalstats.YDIF:file={esc(log)}",
                    "-an", "-f", "null", "-"], check=True)
    vals = [float(ln.split("=", 1)[1]) for ln in log.read_text().splitlines()
            if ln.startswith("lavfi.signalstats.YDIF=")]
    return np.array(vals)


def safe_overlay(img: Image.Image, scale: float) -> Image.Image:
    im = img.copy().convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    W, H = im.size
    d.rectangle([0, 0, W, SAFE["top"] * scale], fill=(255, 0, 0, 60))
    d.rectangle([0, SAFE["bottom"] * scale, W, H], fill=(255, 0, 0, 60))
    d.rectangle([SAFE["right"] * scale, 0, W, H], fill=(255, 120, 0, 50))
    d.rectangle([0, 0, SAFE["left"] * scale, H], fill=(255, 120, 0, 35))
    return Image.alpha_composite(im, ov).convert("RGB")


def title_boxes(edl: dict, ass: Path, total: float) -> list[dict]:
    """Render titles alone on black and measure their ink bounding boxes.  A title that cannot be
    rendered (missing/unreadable ASS, libass or font error, nothing drawn) comes back with an
    'error' and inside_safe False, so it fails instead of silently passing."""
    res = []
    if edl.get("titles") and not ass.exists():
        return [{"title": k, "text": t.get("text", ""), "phase": "all", "t": None, "bbox": None,
                 "inside_safe": False, "error": f"{ass} does not exist"} for k, t in enumerate(edl["titles"])]
    for k, t in enumerate(edl.get("titles", [])):
        for phase, tt in (("start+fade", t["start"] + t.get("fade_in", 0.8) + 0.05),
                          ("mid", 0.5 * (t["start"] + t["end"])), ("end-fade", t["end"] - t.get("fade_out", 0.7) - 0.05)):
            p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                f"color=black:s=1080x1920:r=30:d={total + 1:.2f}", "-vf",
                                f"ass={esc(ass)}:fontsdir={esc(FONTS)},trim=start={tt:.3f},setpts=PTS-STARTPTS,format=gray",
                                "-frames:v", "1", "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw = p.stdout
            if p.returncode != 0 or len(raw) < 1080 * 1920:
                err = p.stderr.decode("utf-8", "replace").strip().splitlines()
                res.append({"title": k, "text": t["text"], "phase": phase, "t": round(tt, 2), "bbox": None,
                            "inside_safe": False,
                            "error": (err[-1] if err else f"ffmpeg exit {p.returncode}, no frame")[-300:]})
                continue
            a = np.frombuffer(raw[:1080 * 1920], np.uint8).reshape(1920, 1080)
            ys, xs = np.where(a > 40)
            if xs.size == 0:
                res.append({"title": k, "text": t["text"], "phase": phase, "t": round(tt, 2), "bbox": None,
                            "inside_safe": False, "error": "nothing visible was drawn (font missing? title faded out?)"})
                continue
            bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            inside = (bb[0] >= SAFE["left"] and bb[2] <= SAFE["right"] and bb[1] >= SAFE["top"]
                      and bb[3] <= SAFE["bottom"])
            res.append({"title": k, "text": t["text"], "phase": phase, "t": round(tt, 2), "bbox": bb,
                        "inside_safe": bool(inside)})
    return res


def main() -> int:
    require("ffmpeg")
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="default: the first entry of work/render/timeline.json 'outputs', else output/<export.name>")
    ap.add_argument("--edl", default=str(REPO / "edit" / "edl.json"))
    args = ap.parse_args()
    edl = json.loads(Path(args.edl).read_text())
    tl = json.loads((WORK / "render" / "timeline.json").read_text())
    outputs = [Path(o) if Path(o).is_absolute() else REPO / o for o in (tl.get("outputs") or []) if o]
    if args.video:
        video = Path(args.video)
    elif outputs:
        video = outputs[0]
    else:
        video = REPO / "output" / (edl.get("export") or {}).get("name", "Madinah_Final_Reel.mp4")
    # the other outputs (the 4K master) get their own report and sheet names instead of overwriting the reel's
    tag = f"_{video.stem}" if any(video.resolve() == o.resolve() for o in outputs[1:]) else ""
    QC.mkdir(parents=True, exist_ok=True)
    fps = float(edl["fps"])
    W, H = edl.get("size", [1080, 1920])
    fails, warns, skipped = [], [], []

    # ---------------- spec
    info = ffprobe(video)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    vui = sps_vui(video) if v["codec_name"] == "h264" else {}
    spec = {
        "codec": v["codec_name"], "profile": v.get("profile"), "level": v.get("level"), "pix_fmt": v["pix_fmt"],
        "size": [v["width"], v["height"]], "r_frame_rate": v["r_frame_rate"], "avg_frame_rate": v["avg_frame_rate"],
        "frames": int(v.get("nb_frames", 0)), "color": [v.get("color_primaries"), v.get("color_transfer"),
                                                        v.get("color_space"), v.get("color_range")],
        "vui_color": ([vui.get("colour_primaries"), vui.get("transfer_characteristics"),
                       vui.get("matrix_coefficients"), vui.get("video_full_range_flag")]
                      if vui.get("colour_description_present_flag") else None),
        "colr": colr_nclx(video),
        "duration": float(info["format"]["duration"]), "bitrate_mbps": round(int(info["format"]["bit_rate"]) / 1e6, 2),
        "video_duration": float(v.get("duration") or 0), "video_start": float(v.get("start_time") or 0),
        "audio": a and {"codec": a["codec_name"], "profile": a.get("profile"), "rate": a["sample_rate"],
                        "channels": a["channels"], "kbps": round(int(a.get("bit_rate", 0)) / 1000),
                        "duration": float(a.get("duration") or 0), "start": float(a.get("start_time") or 0)},
        "faststart": moov_first(video),
    }
    exp_frames = tl["total_frames"]
    try:
        need = h264_level(v["width"], v["height"], fps)          # the level render.assemble writes
        need_idc = int(round(float(need) * 10))
    except SystemExit as e:                                     # beyond level 6.2
        need, need_idc = str(e), 999
    lvl = int(v.get("level") or 0)
    sizes = [[W, H]] + ([list(edl["master_size"])] if edl.get("master_size") else [])
    au = spec["audio"] or {}
    checks = [
        (spec["codec"] == "h264" and spec["profile"] == "High", "H.264 High profile", [spec["codec"], spec["profile"]]),
        (lvl >= need_idc, f"H.264 level >= {need} for {v['width']}x{v['height']} @ {fps:g} fps", lvl / 10),
        (spec["pix_fmt"] == "yuv420p", "yuv420p", spec["pix_fmt"]),
        (spec["size"] in sizes or video.name.find("Master") >= 0, f"size {W}x{H}", spec["size"]),
        (spec["r_frame_rate"] == spec["avg_frame_rate"], "constant frame rate",
         [spec["r_frame_rate"], spec["avg_frame_rate"]]),
        (abs(float(eval(spec["avg_frame_rate"])) - fps) < 0.01, f"{fps:g} fps", spec["avg_frame_rate"]),
        (spec["frames"] == exp_frames, f"frame count {exp_frames}", spec["frames"]),
        (spec["color"][:3] == ["bt709", "bt709", "bt709"] and spec["color"][3] == "tv", "BT.709 limited tags",
         spec["color"]),
        (spec["vui_color"] == [1, 1, 1, 0], "SPS VUI colour 1/1/1 limited range", spec["vui_color"]),
        (spec["colr"] == [1, 1, 1, 0], "MP4 colr nclx 1/1/1 limited range", spec["colr"]),
        (au.get("codec") == "aac" and au.get("rate") == "48000" and au.get("channels") == 2, "AAC 48 kHz stereo",
         au and [au["codec"], au["rate"], au["channels"]]),
        (au.get("kbps", 0) >= 128, "audio bitrate >= 128 kb/s", au.get("kbps")),
        (spec["faststart"], "moov before mdat (faststart)", spec["faststart"]),
    ]
    for ok, what, got in checks:
        (None if ok else fails.append(f"spec: {what} — got {got}"))
    if need_idc < 999 and lvl > need_idc:
        warns.append(f"H.264 level {lvl / 10:g} is higher than the {need} that {v['width']}x{v['height']} @ {fps:g} needs")
    if au:
        if abs(au["duration"] - spec["video_duration"]) > 1.0 / fps:
            fails.append(f"audio lasts {au['duration']:.3f}s, video {spec['video_duration']:.3f}s "
                         f"(A/V duration differs by more than a frame)")
        if abs(au["start"] - spec["video_start"]) > 0.5 / fps:
            fails.append(f"audio starts at {au['start']:.3f}s, video at {spec['video_start']:.3f}s (A/V start offset)")

    # ---------------- picture problems
    # d=0: a single black frame (1/30 s) is shorter than any useful minimum; dips and the end fade are excluded.
    # freezedetect from 6 frames, so the holds too long for the neighbour test below are seen.
    log = lavfi_log(video, vf=f"blackdetect=d=0:pix_th=0.08:pic_th=0.97,freezedetect=n=-55dB:d={6 / fps:.4f}")
    blacks = [(float(s), float(e)) for s, e in re.findall(rf"black_start:({NUM}) black_end:({NUM})", log)]
    freezes = re.findall(rf"freeze_start: ({NUM})|freeze_end: ({NUM})", log)
    fz, cur = [], None
    for s, e in freezes:
        if s:
            cur = float(s)
        elif e and cur is not None:
            fz.append((cur, float(e))); cur = None
    if cur is not None:
        fz.append((cur, spec["duration"]))
    end_fade = float((edl.get("end") or {}).get("fade_to_black", 0) or 0)
    dips = [((sh["T_frames"] - sh["d_in"] / 2) / fps - 0.05, (sh["T_frames"] + sh["d_in"] / 2) / fps + 0.05)
            for sh in tl["shots"] if (sh["shot"].get("transition") or {}).get("type") == "dip"]
    for s, e in blacks:
        if s >= spec["duration"] - end_fade - 0.1:
            continue
        if any(a <= s and e <= b for a, b in dips):
            continue          # intentional dip to black
        fails.append(f"black frames {s:.2f}–{e:.2f}s ({max(1, round((e - s) * fps))} frame(s))")

    # duplicate-frame stutter and stray/flash frames per shot (dissolve/dip halves and the end fade excluded)
    ydif = per_frame_ydif(video)
    if len(ydif) != exp_frames:
        fails.append(f"YDIF series has {len(ydif)} values for {exp_frames} frames (decode problem?)")
    fade0 = exp_frames - int(round(end_fade * fps))
    shot_rows, windows = [], []
    for sh in tl["shots"]:
        a0 = sh["T_frames"] + (sh["d_in"] // 2)
        a1 = min(sh["T_frames"] + sh["dur_frames"] - (sh["d_out"] // 2), fade0)
        seg = ydif[a0 + 1:a1] if a1 - a0 > 2 else np.array([])
        if seg.size == 0:
            continue
        windows.append((sh["id"], a0 + 1, a1))
        med = float(np.median(seg))
        flashes = flash_frames(seg)
        inside_flash = {i + j for i, n in flashes for j in range(n + 1)}
        dups = [i for i in dup_frames(seg) if i not in inside_flash]
        mode = (sh.get("retime") or {}).get("mode")
        step = float(sh.get("step") or 1.0)
        # A real-frame conform with step < 1 (24/25 fps or VFR on a 30 fps timeline, retime 'frames')
        # repeats about (1 - step) of its frames, evenly spaced, by design; more than that is stutter.
        expected = int(round(seg.size * (1 - step))) if mode == "frames" and step < 0.999 else 0
        limit = expected + (max(2, math.ceil(0.25 * expected)) if expected else 0)
        cv = float(np.std(seg) / (np.mean(seg) + 1e-9))
        shot_rows.append({"id": sh["id"], "speed": sh["speed"], "retime": mode, "step": step,
                          "motion_median": round(med, 2), "duplicate_frames": len(dups), "expected_dups": expected,
                          "flash_frames": [a0 + 1 + i for i, _ in flashes], "cadence_cv": round(cv, 2)})
        if len(dups) > limit:
            fails.append(f"{sh['id']}: {len(dups)} duplicated frames inside a moving shot (stutter), first at "
                         f"f{a0 + 1 + dups[0]}" + (f"; the step-{step:g} conform explains only ~{expected}"
                                                    if expected else ""))
        elif expected and fps * step < 20:
            warns.append(f"{sh['id']}: real-frame conform at step {step:g} shows only {fps * step:.0f} new frames/s "
                         f"(visible judder; retime 'blend' or a higher-fps source?)")
        for i, n in flashes:
            f0 = a0 + 1 + i
            fails.append(f"{sh['id']}: {'single' if n == 1 else 'two'} foreign/flash frame{'s' if n > 1 else ''} "
                         f"at f{f0} ({f0 / fps:.2f}s) inside the shot")
    for s, e in fz:
        # a moving picture that suddenly stops dead inside a shot is a stall, not a locked-off shot
        fs, fe = int(round(s * fps)), int(round(e * fps))
        w = next((w for w in windows if w[1] + 6 <= fs < w[2]), None)
        if w is not None:
            before, during = ydif[fs - 6:fs], ydif[fs + 1:min(fe, w[2])]
            if before.size == 6 and during.size and before.min() > 0.4 and during.max() < 0.1 * before.min():
                fails.append(f"{w[0]}: the picture freezes {s:.2f}–{e:.2f}s in the middle of a moving shot")
                continue
        if e - s >= 0.4:
            warns.append(f"freeze/static picture {s:.2f}–{e:.2f}s (fine for a locked-off shot; check it is intended)")

    # cuts vs timeline
    log = lavfi_log(video, vf="scdet=threshold=12")
    det = [round(float(t), 3) for t in re.findall(rf"lavfi\.scd\.time: ({NUM})", log)]
    planned = [sh["T_frames"] / fps for sh in tl["shots"][1:] if sh["d_in"] == 0]
    unexpected = [t for t in det if all(abs(t - p) > 1.5 / fps for p in planned)]
    missing = [p for p in planned if all(abs(t - p) > 1.5 / fps for t in det)]
    for p in list(missing):
        near = [t for t in unexpected if abs(t - p) <= 3.5 / fps]
        if near:          # the planned cut is absent but a change sits 2-3 frames from it: misplaced cut
            fails.append(f"cut planned at {p:.3f}s happens at {near[0]:.3f}s "
                         f"({round((near[0] - p) * fps):+d} frames): misplaced cut")
            missing.remove(p)
            unexpected.remove(near[0])
    if unexpected:
        warns.append(f"scene changes not in the timeline at {unexpected} (flash/jump inside a shot?)")
    if missing:
        warns.append(f"planned cuts not detected at {[round(m, 3) for m in missing]} (match cuts can do this)")

    # ---------------- audio
    alog = lavfi_log(video, af="ebur128=peak=true,silencedetect=n=-50dB:d=0.35")
    s = alog[alog.rfind("Summary:"):]
    def g(k):
        m = re.search(k + r":\s+(-?[\d.]+|-inf)", s)
        return float(m.group(1)) if m and m.group(1) != "-inf" else None
    loud = {"I": g("I"), "LRA": g("LRA"), "TP": g("Peak")}
    sil = re.findall(rf"silence_start: ({NUM})", alog)
    lpath = WORK / "render" / "loudness.json"
    lj = json.loads(lpath.read_text()) if lpath.exists() else {}
    target = float((edl.get("loudness") or {}).get("target_lufs", -14.0))
    if loud["TP"] is not None and loud["TP"] > -1.0:
        fails.append(f"true peak {loud['TP']} dBTP > -1.0 (risk of clipping after AAC re-encode)")
    if loud["I"] is not None and abs(loud["I"] - target) > 1.0:
        msg = f"integrated loudness {loud['I']} LUFS is {loud['I'] - target:+.1f} LU off the EDL target {target:g} LUFS"
        short = lj.get("tp_limited_shortfall_lu")
        if short is not None:
            msg += (f" (render: the true-peak ceiling held the gain back, tp_limited_shortfall_lu = {float(short):.2f})"
                    if float(short) > 0 else " (render reports tp_limited_shortfall_lu = 0: stale loudness.json?)")
        warns.append(msg)
    for w in lj.get("warnings") or []:
        warns.append(f"render audio: {w}")
    for t in sil:
        if float(t) < spec["duration"] - 2.5:
            warns.append(f"silence at {float(t):.2f}s (intended pause?)")
    av = {}
    if a:
        x = decode_audio(video)
        if au.get("duration"):           # drop the padding of the last AAC frame, past the stream's end
            x = x[:int(round(au["duration"] * SR))]
        e5 = SR // 200                                    # 5 ms
        mix = WORK / "render" / "mix.wav"
        ref = decode_audio(mix, 5.0) if mix.exists() else None
        if x.size:
            peak = float(np.abs(x).max())
            n_fs = int((np.abs(x) >= 0.999).sum())
            av.update(sample_peak_dbfs=round(20 * math.log10(peak), 2) if peak > 0 else None, full_scale_samples=n_fs)
            if n_fs:
                fails.append(f"{n_fs} audio samples at full scale (clipped)")
            # the decoded reel starts with the encoder priming, so the head is judged on mix.wav when it exists
            head = ref if ref is not None and ref.size else x
            for where, seg in (("starts", head[:e5]), ("ends", x[-e5:])):
                if seg.size and float(np.abs(seg).max()) > 0.02:
                    warns.append(f"audio {where} abruptly (|x| up to {float(np.abs(seg).max()):.3f} in the "
                                 f"{'first' if where == 'starts' else 'last'} 5 ms): click")
        if ref is None:
            skipped.append("A/V offset: work/render/mix.wav not found")
        else:
            r = av_lag(x[:len(ref)].mean(axis=1), ref.mean(axis=1))
            if r is None:
                skipped.append("A/V offset: the first 5 s of work/render/mix.wav are silent")
            else:
                lag, corr = r
                av.update(lag_samples=lag, lag_ms=round(lag * 1000 / SR, 2), correlation=round(corr, 3))
                if corr < 0.5:
                    warns.append(f"the reel's audio does not match work/render/mix.wav (correlation {corr:.2f}): "
                                 f"stale mix or wrong audio track — A/V offset not checked")
                elif abs(lag) > SR // 1000:
                    fails.append(f"audio is {abs(lag) * 1000 / SR:.1f} ms {'late' if lag > 0 else 'early'} against the "
                                 f"picture (cross-correlation of the first 5 s with work/render/mix.wav)")
        raw_mix = WORK / "render" / "mix_raw.wav"
        if raw_mix.exists():
            runs = flat_peak_runs(decode_audio(raw_mix))
            av["flat_peak_runs"] = runs
            if runs >= 10:
                warns.append(f"work/render/mix_raw.wav has {runs} flat-topped runs at its peak level: a music/ambience "
                             f"source looks hard-clipped (distortion that no gain change can undo)")

    # ---------------- music cut point vs. the phrases of the recitation / nasheed
    mus = tl.get("music")
    if not mus and edl.get("music"):
        m_in = float(edl["music"].get("in", 0.0))
        mus = {"file": edl["music"]["file"], "in": m_in, "end": m_in + tl["total"]}
    if mus:
        name = Path(mus["file"]).name
        aj = WORK / "audio" / (Path(mus["file"]).stem.replace(" ", "_") + ".json")
        an = json.loads(aj.read_text()) if aj.exists() else None
        rel = lambda p: (Path(p) if Path(p).is_absolute() else REPO / p).resolve()
        if an and an.get("file") and rel(an["file"]) != rel(mus["file"]):
            skipped.append(f"music_end_in_phrase: work/audio/{aj.name} describes {an['file']}, not {mus['file']}")
        elif an:
            end = float(mus["end"])
            inside = [p for p in an.get("phrases", []) if p[0] + 0.05 < end < p[1] - 0.05]
            if inside:
                p0, p1 = inside[0]
                prev = [q[1] for q in an["phrases"] if q[1] <= end and q[1] > float(mus.get("in", 0))]
                msg = (f"music_end_in_phrase: the music stops at {end:.2f}s of {name}, inside the phrase "
                       f"{p0:.2f}–{p1:.2f}s; make the reel {p1 - end:.2f}s longer (phrase end {p1:.2f}s)"
                       + (f" or {end - max(prev):.2f}s shorter ({max(prev):.2f}s)" if prev else "")
                       + ", or move music.in")
                (warns if (edl.get("music") or {}).get("allow_mid_phrase") else fails).append(msg)
            if end > float(an.get("duration", end)) + 0.05:
                fails.append(f"music_past_track_end: the reel needs {name} until {end:.2f}s but it ends at "
                             f"{float(an['duration']):.2f}s (the rest is silence)")
        else:
            skipped.append(f"music_end_in_phrase: no work/audio/{aj.name} (run tools/audio_analysis.py)")

    # ---------------- titles vs safe area (rebuilt from the EDL being checked)
    boxes = []
    try:
        ass = build_ass(edl, QC / "titles_check.ass")
    except SystemExit as e:
        fails.append(f"titles: the EDL's titles cannot be built: {e}")
    else:
        boxes = title_boxes(edl, ass, tl["total"])
    for b in boxes:
        if b.get("error"):
            fails.append(f"title {b['title']} ({b['phase']}) could not be rendered for the safe-area check: {b['error']}")
        elif b["inside_safe"] is False:
            fails.append(f"title {b['title']} ({b['phase']}) bbox {b['bbox']} leaves the Instagram-safe area")

    # ---------------- sheets
    sw, shh = 360, 640
    want = {0, exp_frames - 1}
    pairs = []
    for sh in tl["shots"][1:]:
        T = sh["T_frames"]
        if sh["d_in"]:
            trio = [T - sh["d_in"] // 2, T, T + sh["d_in"] // 2 - 1]
            pairs.append((f"{sh['id']} {(sh['shot'].get('transition') or {}).get('type', 'dissolve')}", trio))
        else:
            pairs.append((sh["id"] + " cut", [T - 1, T]))
        want.update(pairs[-1][1])
    tmid = [int(round(0.5 * (t["start"] + t["end"]) * fps)) for t in edl.get("titles", [])]
    want.update(tmid)
    slow = [sh for sh in tl["shots"] if sh["speed"] < 0.999]
    for sh in slow:
        mid = sh["T_frames"] + sh["dur_frames"] // 2
        want.update(range(mid - 6, mid + 6))
    fr = frames_at(video, sorted(want), sw, shh)
    blank = np.zeros((shh, sw, 3), np.uint8)
    get = lambda i: fr.get(i, blank)
    tile([get(0), get(exp_frames - 1)], ["first frame", "last frame"], 2, "first / last").save(QC / f"first_last{tag}.png")
    frames, labels = [], []
    for name, idxs in pairs:
        for i in idxs:
            frames.append(get(i)); labels.append(f"{name} f{i} {i / fps:.2f}s")
    if frames:
        tile(frames, labels, 6, "cuts & dissolves (frame before | frame at cut)").save(QC / f"cuts{tag}.png")
    if tmid:
        ims = [np.array(safe_overlay(Image.fromarray(get(i)), sw / 1080)) for i in tmid]
        tile(ims, [f"title {k} @ {i / fps:.2f}s" for k, i in enumerate(tmid)], min(4, len(ims)),
             "titles with Instagram UI zones (red/orange = keep clear)").save(QC / f"titles{tag}.png")
    for sh in slow:
        mid = sh["T_frames"] + sh["dur_frames"] // 2
        idxs = list(range(mid - 6, mid + 6))
        tile([get(i) for i in idxs], [f"f{i}" for i in idxs], 6,
             f"{sh['id']} x{sh['speed']} consecutive frames — check warping/ghosting").save(QC / f"slowmo_{sh['id']}{tag}.png")

    report = {"file": str(video), "spec": spec, "loudness": loud, "audio_checks": av, "blacks": blacks, "freezes": fz,
              "shots": shot_rows, "detected_cuts": det, "planned_cuts": planned, "music": mus, "title_boxes": boxes,
              "fails": fails, "warnings": warns, "skipped": skipped}
    (QC / f"report{tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    md = [f"# QC — {video.name}", "", "## Spec", "```json", json.dumps(spec, indent=1), "```",
          f"Loudness: {loud}  (EDL target {target:g} LUFS)", f"Audio: {av}", "", "## Shots",
          "| id | speed | retime | step | motion | dup frames (expected) | flash frames | cadence cv |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['speed']} | {r['retime']} | {r['step']:g} | {r['motion_median']} | "
           f"{r['duplicate_frames']} ({r['expected_dups']}) | {r['flash_frames'] or ''} | {r['cadence_cv']} |"
           for r in shot_rows]
    md += ["", "## Titles", ""] + [f"- {b.get('text', '')} [{b['phase']}] bbox={b['bbox']} safe={b['inside_safe']}"
                                   + (f" error={b['error']}" if b.get("error") else "") for b in boxes]
    md += ["", "## FAIL", ""] + [f"- {f}" for f in fails] + ["", "## Warnings", ""] + [f"- {w}" for w in warns]
    if skipped:
        md += ["", "## Skipped checks", ""] + [f"- {k}" for k in skipped]
    md += ["", f"Sheets: work/qc/first_last{tag}.png, cuts{tag}.png, titles{tag}.png, slowmo_*{tag}.png"]
    (QC / f"report{tag}.md").write_text("\n".join(md))
    print("\n".join(md))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
