#!/usr/bin/env python3
"""Step 2 — look at the footage.

    python3 tools/review.py [--only SUBSTRING] [--no-proxy] [--strip-fps 4]

For every clip in work/inventory.json writes, under work/review/<clip>/:
  contact.png     24 evenly spaced frames with timecodes (SDR-converted, display-oriented)
  strip_XX.png    dense filmstrip pages (default 4 fps) — read motion, focus pulls, people
  metrics.png     exposure / clipped highlights / sharpness / camera speed / jitter over time
  metrics.json    the same numbers + suggested "strongest moments"
and work/proxies/<clip>.mp4 (540 px wide SDR proxy for scrubbing).
<clip> is the file stem plus a short hash of its path (IMG_0001_5d07bf), so clips with the same
stem in different folders, or with different extensions, never overwrite each other.
Times (contact labels, metrics, suggested moments) are source seconds on the axis of the EDL's
src_in, taken from the real frame timestamps, so VFR phone clips are not stretched.
Originals are only read.  An FFmpeg error, an empty decode or an all-black contact sheet stops
that clip with FFmpeg's own message; the other clips are still reviewed and the exit code is 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color as colormod  # noqa: E402
from ffutil import esc  # noqa: E402
from mediainfo import REPO, from_rgb_filter, nominal_fps, require, to_rgb_filter  # noqa: E402

WORK = REPO / "work"
BLACK_LEVEL = 16          # contact sheet counts as black when no pixel of any frame exceeds this (0-255)


class ReviewError(RuntimeError):
    """A review step failed (FFmpeg error, nothing decoded, black preview).  main() reports it
    for that clip, carries on with the others and exits non-zero."""


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


_SEEN_STDERR: set[str] = set()


def _report_stderr(what: str, err: str) -> None:
    """FFmpeg exited 0 but logged errors (a damaged frame, say): show each new line once."""
    new = [ln for ln in err.strip().splitlines() if ln and ln not in _SEEN_STDERR]
    _SEEN_STDERR.update(new)
    if new:
        print(f"ffmpeg warning ({what}):\n  " + "\n  ".join(new[:10]), file=sys.stderr)


def run_ffmpeg(cmd: list[str], what: str) -> bytes:
    """Run FFmpeg and return stdout; raise ReviewError with FFmpeg's error text on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode != 0:
        raise ReviewError(f"{what}: ffmpeg exited with code {p.returncode}\n{_tail(err) or '(no error text)'}")
    _report_stderr(what, err)
    return p.stdout


def font(size: int):
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def preview_transform(col: dict) -> str:
    # dovi5 has no displayable base layer: it is shown as-is (sdr709 identity) and flagged.
    return col["transform"] if col["transform"] in ("hlg", "pq") else "sdr709"


def preview_warnings(clip: dict) -> list[str]:
    if clip["video"]["color"]["transform"] == "dovi5":
        return ["Dolby Vision profile 5: the base layer is IPTPQc2, not Y'CbCr, so colours and exposure in "
                "these previews and metrics are WRONG, and render.py refuses this clip. Export HLG or SDR "
                "from the phone or editor first; judge only framing and motion here."]
    return []


def neutral_lut(col: dict) -> Path:
    """Neutral (look-free) preview LUT, cached in work/luts.  The name carries the transform, the
    colour code's LUT_VERSION and, for PQ, the content peak, so a colour-science change or a clip
    with another peak never reuses a stale cube."""
    transform = preview_transform(col)
    ver = getattr(colormod, "LUT_VERSION", 1)
    peak = float(col.get("peak_nits") or 1000) if transform == "pq" else None
    path = WORK / "luts" / (f"neutral_{transform}_v{ver}" + (f"_{peak:g}" if peak else "") + ".cube")
    if not path.exists():
        lut = colormod.build_lut(transform, neutral=True, size=65, hdr_peak_nits=peak or 1000.0)
        tmp = path.with_suffix(".tmp")
        colormod.write_cube(tmp, lut, f"neutral {transform}")
        tmp.replace(path)                     # never leave a half-written cube in the cache
    return path


def sdr_chain(clip: dict, width: int, height: int | None = None, pix: str = "rgb24") -> str:
    """Filter chain: (autorotated) source -> SDR BT.709 preview at given width."""
    col = clip["video"]["color"]
    h = height or -2
    chain = [f"scale={width}:{h}:flags=bicubic", to_rgb_filter(col, "gbrpf32le"),
             f"lut3d=file={esc(neutral_lut(col))}:interp=tetrahedral"]
    if pix == "rgb24":
        chain.append("format=rgb24")
    else:
        chain.append(from_rgb_filter(pix))
    return ",".join(chain)


def src_path(clip: dict) -> Path:
    p = Path(clip["path"])
    return p if p.is_absolute() else REPO / p


def grab_frames(clip: dict, times: list[float], width: int) -> list[np.ndarray]:
    """The real frame nearest each time, picked the way render.py picks a shot's first frame:
    coarse input seek ~2 s early, exact trim in the graph (an accurate -ss on open-GOP HEVC can
    land 1-4 frames late).  A contact-sheet time used as src_in starts the shot on that frame."""
    dw, dh = clip["video"]["display_size"]
    height = int(round(width * dh / dw / 2)) * 2
    src_fps = clip["video"].get("fps") or 30.0
    need = width * height * 3
    frames = []
    for t in times:
        coarse = max(0.0, t - 2.0)
        vf = (f"trim=start={max(0.0, t - coarse - 0.5 / src_fps):.6f},setpts=PTS-STARTPTS,"
              + sdr_chain(clip, width, height))
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{coarse:.6f}", "-t", f"{t - coarse + 1.0:.6f}",
               "-i", str(src_path(clip)), "-an", "-frames:v", "1", "-vf", vf,
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = run_ffmpeg(cmd, f"contact frame at {t:.3f} s")
        if len(raw) < need:
            raise ReviewError(f"contact frame at {t:.3f} s: ffmpeg returned no picture "
                              f"({len(raw)} of {need} bytes)")
        frames.append(np.frombuffer(raw[:need], np.uint8).reshape(height, width, 3))
    return frames


def stream_frames(clip: dict, width: int, fps: float | None, gray: bool, pts: list[float] | None = None):
    """Yield every decoded frame (with fps: frames resampled to that rate by timestamp).

    Given a `pts` list, frames pass through unpadded (-fps_mode passthrough; the rawvideo muxer
    would otherwise duplicate frames up to r_frame_rate) and, once the stream has been read to
    the end, `pts` holds each yielded frame's real presentation time in seconds (FFmpeg >= 6.1
    per-frame encoder stats).  Raises ReviewError when FFmpeg fails or decodes nothing."""
    dw, dh = clip["video"]["display_size"]
    height = int(round(width * dh / dw / 2)) * 2
    vf = sdr_chain(clip, width, height)
    if fps:
        vf = f"fps={fps}," + vf
    if gray:
        vf += ",format=gray"
    pix = "gray" if gray else "rgb24"
    ch = 1 if gray else 3
    what = f"decode at {fps:g} fps" if fps else "full decode"
    with tempfile.TemporaryDirectory() as td:
        stats = Path(td) / "pts.txt"
        cmd = ["ffmpeg", "-v", "error", "-i", str(src_path(clip)), "-vf", vf, "-an"]
        if pts is not None:
            # A fine explicit time base keeps every real timestamp (to 11 us); the default (1/frame
            # rate) would snap them to frame numbers.  '-1'/'demux' differ between FFmpeg 6 and 7+.
            cmd += ["-fps_mode", "passthrough", "-enc_time_base", "1/90000",
                    "-stats_enc_pre", str(stats), "-stats_enc_pre_fmt", "{pts} {tb}"]
        cmd += ["-f", "rawvideo", "-pix_fmt", pix, "-"]
        with tempfile.TemporaryFile() as errf:       # a file, not a pipe: FFmpeg can never block on it
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
            n = width * height * ch
            count = 0
            try:
                while True:
                    buf = proc.stdout.read(n)
                    if len(buf) < n:
                        break
                    count += 1
                    a = np.frombuffer(buf, np.uint8)
                    yield a.reshape(height, width) if gray else a.reshape(height, width, 3)
            finally:
                proc.stdout.close()
                proc.wait()
            # Only reached when the stream was read to the end, not when the caller stopped early.
            errf.seek(0)
            err = errf.read().decode("utf-8", "replace")
        if proc.returncode != 0:
            raise ReviewError(f"{what}: ffmpeg exited with code {proc.returncode}\n{_tail(err) or '(no error text)'}")
        _report_stderr(what, err)
        if count == 0:
            raise ReviewError(f"{what}: ffmpeg decoded no frames\n{_tail(err)}")
        if pts is not None and stats.exists():
            for line in stats.read_text().splitlines():
                try:
                    p, tb = line.split()
                    pts.append(float(int(p) * Fraction(tb)))
                except (ValueError, ZeroDivisionError):
                    continue


def label(img: Image.Image, text: str, size: int = 18):
    d = ImageDraw.Draw(img)
    f = font(size)
    bbox = d.textbbox((0, 0), text, font=f)
    d.rectangle([4, 4, 12 + bbox[2], 10 + bbox[3]], fill=(0, 0, 0))
    d.text((8, 6), text, fill=(255, 230, 120), font=f)


def tile(frames: list[np.ndarray], labels: list[str], cols: int, title: str) -> Image.Image:
    h, w = frames[0].shape[:2]
    rows = math.ceil(len(frames) / cols)
    head = 34
    sheet = Image.new("RGB", (cols * w + (cols + 1) * 4, head + rows * h + (rows + 1) * 4), (18, 18, 18))
    ImageDraw.Draw(sheet).text((8, 6), title, fill=(230, 230, 230), font=font(20))
    for i, (fr, lb) in enumerate(zip(frames, labels)):
        im = Image.fromarray(fr)
        label(im, lb, max(12, w // 14))
        r, c = divmod(i, cols)
        sheet.paste(im, (4 + c * (w + 4), head + 4 + r * (h + 4)))
    return sheet


def phase_shift(a: np.ndarray, b: np.ndarray, win: np.ndarray) -> tuple[float, float, float]:
    fa = np.fft.rfft2(a * win)
    fb = np.fft.rfft2(b * win)
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-9)
    corr = np.fft.irfft2(r, s=a.shape)
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak = float(corr[idx])
    shifts = []
    for ax, i in enumerate(idx):
        n = corr.shape[ax]
        im1 = list(idx); ip1 = list(idx)
        im1[ax] = (i - 1) % n; ip1[ax] = (i + 1) % n
        c0, cm, cp = corr[idx], corr[tuple(im1)], corr[tuple(ip1)]
        den = cm - 2 * c0 + cp
        sub = 0.5 * (cm - cp) / den if abs(den) > 1e-12 else 0.0
        s = i + sub
        if s > n / 2:
            s -= n
        shifts.append(s)
    return shifts[1], shifts[0], peak   # dx, dy, confidence


def analyse(clip: dict, width: int = 320) -> dict:
    """Exposure / sharpness / motion per real frame, resampled onto a uniform time grid.

    Every frame is decoded once with its real timestamp (no CFR padding), so a VFR phone clip
    keeps its true time axis and padded duplicates never fake zero-motion samples.  The values
    are then interpolated onto a grid at the camera's cadence (the median real frame interval,
    snapped to a standard rate, at most 60 fps); the camera trajectory is interpolated in time,
    so speed stays in frame-widths per second."""
    v = clip["video"]
    lum, clip_hi, crush, sharp, dx, dy, conf = [], [], [], [], [], [], []
    pts: list[float] = []
    prev = None
    win = None
    for g in stream_frames(clip, width, None, gray=True, pts=pts):
        f = g.astype(np.float32) / 255.0
        lum.append(float(f.mean()))
        clip_hi.append(float((f > 0.975).mean()))
        crush.append(float((f < 0.02).mean()))
        lap = (-4 * f[1:-1, 1:-1] + f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:])
        sharp.append(float(lap.var() * 1e4))
        if win is None:
            win = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1])).astype(np.float32)
        if prev is not None:
            sx, sy, c = phase_shift(prev, f, win)
            dx.append(sx); dy.append(sy); conf.append(c)
        else:
            dx.append(0.0); dy.append(0.0); conf.append(1.0)
        prev = f
    frames = len(lum)
    if frames == 0:
        raise ReviewError("analysis: no frames decoded")
    warnings = []
    ft = np.array(pts, dtype=np.float64)
    if len(ft) != frames or np.any(np.diff(ft) <= 0):
        src_fps = float(v.get("fps") or v.get("fps_nominal") or 30.0)
        warnings.append(f"no usable frame timestamps ({len(ft)} for {frames} frames): "
                        f"times assume a constant {src_fps:g} fps")
        ft = np.arange(frames) / src_fps
    dt = float(np.median(np.diff(ft))) if frames > 1 else 0.0       # the camera's frame interval
    cadence = nominal_fps(1.0 / dt) if dt > 0 else float(v.get("fps_nominal") or v.get("fps") or 30.0)
    fps = min(cadence, 60.0)                                        # analysis grid rate
    end = float(ft[-1]) + (dt or 1.0 / fps)                         # end of the last real frame
    n = int(math.floor((ft[-1] - ft[0]) * fps + 1e-6)) + 1
    t = ft[0] + np.arange(n) / fps

    def rs(a):
        return np.interp(t, ft, np.asarray(a, dtype=np.float64))

    lum_g, clip_g, crush_g, sharp_g = rs(lum), rs(clip_hi), rs(crush), rs(sharp)
    traj_x, traj_y = rs(np.cumsum(dx)), rs(np.cumsum(dy))
    dxa, dya = np.diff(traj_x, prepend=traj_x[0]), np.diff(traj_y, prepend=traj_y[0])
    speed = np.hypot(dxa, dya) * fps / width          # frame-widths per second
    k = max(3, int(round(fps * 0.5)) | 1)
    kern = np.ones(k) / k
    smooth_x = np.convolve(np.pad(traj_x, k // 2, mode="edge"), kern, "valid")
    smooth_y = np.convolve(np.pad(traj_y, k // 2, mode="edge"), kern, "valid")
    jitter = np.hypot(traj_x - smooth_x, traj_y - smooth_y) / width * 1000   # milli-frame-widths
    sharp_rel = sharp_g / (np.percentile(sharp_g, 90) + 1e-9)

    step = 0.5
    ws = max(1, int(round(step * fps)))

    def win_stats(a):
        return [float(np.mean(a[i:i + ws])) for i in range(0, n, ws)]

    series = {
        "t": [round(float(t[i]), 2) for i in range(0, n, ws)],
        "luma": win_stats(lum_g), "clipped": win_stats(clip_g),
        "crushed": win_stats(crush_g), "sharpness": win_stats(sharp_rel),
        "speed": win_stats(speed), "jitter": win_stats(jitter),
    }
    # ---- suggested strongest moments (advisory; final picks are made by eye)
    best = []
    for length in (2.0, 3.0, 4.5):
        m = int(length * fps)
        if n < m:
            continue
        for s in range(0, n - m + 1, max(1, int(fps / 4))):
            sl = slice(s, s + m)
            score = (1.0 * np.clip(sharp_rel[sl].mean(), 0, 1.2)
                     - 0.04 * jitter[sl].mean()
                     - 3.0 * clip_g[sl].mean()
                     - 1.5 * max(0.0, 0.18 - lum_g[sl].mean())
                     - 0.3 * max(0.0, speed[sl].mean() - 0.35))
            best.append((float(score), round(float(t[s]), 2), length))
    best.sort(reverse=True)
    picks = []
    for sc, st, ln in best:
        if all(st + ln <= p["in"] or st >= p["in"] + p["len"] for p in picks):
            picks.append({"in": st, "len": ln, "score": round(sc, 3)})
        if len(picks) >= 4:
            break
    return {"fps": fps, "frames": frames, "duration": round(end, 3),
            "span": [round(float(ft[0]), 4), round(float(ft[-1]), 4)],
            "vfr": bool(v.get("vfr")), "series": series,
            "summary": {"luma_mean": round(float(np.mean(lum)), 3), "clipped_max": round(float(np.max(clip_hi)), 4),
                        "jitter_median": round(float(np.median(jitter)), 2),
                        "speed_median": round(float(np.median(speed)), 3)},
            "suggested_moments": picks, "warnings": warnings}


def plot_metrics(m: dict, title: str, out: Path):
    s = m["series"]
    W, H, pad = 1400, 760, 50
    img = Image.new("RGB", (W, H), (16, 16, 16))
    d = ImageDraw.Draw(img)
    d.text((pad, 10), title, fill=(235, 235, 235), font=font(20))
    tracks = [("luma", (240, 240, 240), 1.0), ("clipped", (255, 90, 90), 0.2), ("crushed", (90, 120, 255), 0.5),
              ("sharpness", (120, 230, 120), 1.3), ("speed", (255, 200, 60), 1.0), ("jitter", (220, 120, 255), 30.0)]
    th = (H - 60) // len(tracks)
    dur = max(m["duration"], 0.1)
    for i, (key, colr, vmax) in enumerate(tracks):
        y0 = 50 + i * th
        d.rectangle([pad, y0, W - 20, y0 + th - 12], outline=(60, 60, 60))
        d.text((6, y0 + 4), key, fill=colr, font=font(13))
        vals = s[key]
        pts = []
        for j, v in enumerate(vals):
            x = pad + (s["t"][j] / dur) * (W - 20 - pad)
            y = y0 + th - 12 - min(v / vmax, 1.0) * (th - 16)
            pts.append((x, y))
        if len(pts) > 1:
            d.line(pts, fill=colr, width=2)
    for sec in range(0, int(dur) + 1):
        x = pad + sec / dur * (W - 20 - pad)
        d.line([(x, 46), (x, H - 8)], fill=(45, 45, 45))
        if sec % (1 if dur < 20 else 5) == 0:
            d.text((x + 2, H - 22), f"{sec}s", fill=(150, 150, 150), font=font(11))
    for p in m.get("suggested_moments", []):
        x0 = pad + p["in"] / dur * (W - 20 - pad)
        x1 = pad + (p["in"] + p["len"]) / dur * (W - 20 - pad)
        d.rectangle([x0, 40, x1, 46], fill=(255, 200, 60))
    img.save(out)


def safe_name(path: str) -> str:
    """Folder / proxy name for a clip: its stem (characters that are illegal in file names on any
    OS replaced) plus 6 hex digits of a hash of its inventory path, so footage/IMG_0001.MOV,
    footage/day2/IMG_0001.MOV and footage/IMG_0001.mp4 each get their own review and proxy."""
    p = Path(path)
    h = hashlib.sha1(p.as_posix().encode("utf-8")).hexdigest()[:6]
    return re.sub(r"[^\w\-]+", "_", p.stem).strip("_") + "_" + h


def review_clip(clip: dict, args) -> None:
    name = safe_name(clip["path"])
    out = WORK / "review" / name
    out.mkdir(parents=True, exist_ok=True)
    dw, dh = clip["video"]["display_size"]
    vertical = dh > dw
    flag = "[DV5: COLOURS WRONG]  " if preview_warnings(clip) else ""
    # metrics first: they give the real frame span the contact sheet samples
    m = analyse(clip)
    m = {"file": clip["path"], **m, "warnings": preview_warnings(clip) + m["warnings"]}
    for w in m["warnings"]:
        print(f"WARNING {clip['path']}: {w}", file=sys.stderr)
    dur = m["duration"]
    t0, t1 = m["span"]
    # contact sheet: evenly spaced over the real frames, each grabbed at exactly its label time
    # (0.01 s, rounded down so it never passes the last frame), so a label used as src_in
    # renders from the frame shown
    n = 24 if dur >= 12 else max(6, int(dur * 2))
    times = [math.floor(min(t1, t0 + (i + 0.5) * (dur - t0) / n) * 100 + 1e-6) / 100 for i in range(n)]
    tw = 270 if vertical else 400
    frames = grab_frames(clip, times, tw)
    c = clip["video"]["color"]
    title = (f"{flag}{clip['path']}  |  {dw}x{dh} {clip['video']['fps']}fps{' VFR' if m['vfr'] else ''}  "
             f"{dur:.1f}s  {c['transform'].upper()} {c['bit_depth']}-bit")
    tile(frames, [f"{t:.2f}s" for t in times], 8 if vertical else 6, title).save(out / "contact.png")
    brightest = max(int(f.max()) for f in frames)
    if brightest <= BLACK_LEVEL:
        raise ReviewError(f"all {n} contact-sheet frames are black (brightest pixel {brightest}/255): the SDR "
                          f"preview chain produced no picture (check the LUT and the colour classification "
                          f"in work/inventory.md); if the clip really is black, ignore it. "
                          f"See {(out / 'contact.png').relative_to(REPO)}")
    # dense filmstrips (the fps filter starts on the first frame's timestamp, rounded to its grid)
    sw = 180 if vertical else 300
    strip, labels, page = [], [], 0
    per_page = 64 if vertical else 48
    s0 = round(t0 * args.strip_fps)
    for i, fr in enumerate(stream_frames(clip, sw, args.strip_fps, gray=False)):
        strip.append(fr)
        labels.append(f"{(s0 + i) / args.strip_fps:.2f}")
        if len(strip) == per_page:
            tile(strip, labels, 16 if vertical else 8, f"{flag}{name} @ {args.strip_fps} fps  page {page}").save(
                out / f"strip_{page:02d}.png")
            strip, labels, page = [], [], page + 1
    if strip:
        tile(strip, labels, 16 if vertical else 8, f"{flag}{name} @ {args.strip_fps} fps  page {page}").save(
            out / f"strip_{page:02d}.png")
    # metrics
    (out / "metrics.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    plot_metrics(m, f"{flag}{name}: luma / clipped / crushed / sharpness / speed / jitter", out / "metrics.png")
    # proxy
    if not args.no_proxy:
        pdir = WORK / "proxies"
        pdir.mkdir(parents=True, exist_ok=True)
        pw = 540 if vertical else 960
        run_ffmpeg(["ffmpeg", "-v", "error", "-y", "-i", str(src_path(clip)),
                    "-vf", sdr_chain(clip, pw, None, pix="yuv420p"), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                    str(pdir / f"{name}.mp4")], "proxy")
    print(f"reviewed {clip['path']} -> {out.relative_to(REPO)}  suggested: {m.get('suggested_moments')}")


def main() -> int:
    require("ffmpeg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--strip-fps", type=float, default=4.0)
    args = ap.parse_args()
    inv = json.loads((WORK / "inventory.json").read_text())
    clips = [v for v in inv["video"] if v.get("video") and (not args.only or args.only in v["path"])]
    if not clips:
        print("no clips — run tools/inventory.py first", file=sys.stderr)
        return 2
    failed = []
    for clip in clips:
        try:
            review_clip(clip, args)
        except ReviewError as e:
            failed.append(clip["path"])
            print(f"REVIEW FAILED {clip['path']}: {e}", file=sys.stderr)
    if failed:
        print(f"review failed for {len(failed)} of {len(clips)} clip(s): " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
