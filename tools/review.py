#!/usr/bin/env python3
"""Step 2 — look at the footage.

    python3 tools/review.py [--only SUBSTRING] [--no-proxy] [--strip-fps 4]

For every clip in work/inventory.json writes, under work/review/<clip>/:
  contact.png     24 evenly spaced frames with timecodes (SDR-converted, display-oriented)
  strip_XX.png    dense filmstrip pages (default 4 fps) — read motion, focus pulls, people
  metrics.png     exposure / clipped highlights / sharpness / camera speed / jitter over time
  metrics.json    the same numbers + suggested "strongest moments"
and work/proxies/<clip>.mp4 (540 px wide SDR proxy for scrubbing).
Originals are only read.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color as colormod  # noqa: E402
from mediainfo import REPO, from_rgb_filter, require, to_rgb_filter  # noqa: E402

WORK = REPO / "work"


def font(size: int):
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def neutral_lut(transform: str) -> Path:
    path = WORK / "luts" / f"neutral_{transform}.cube"
    if not path.exists():
        colormod.write_cube(path, colormod.build_lut(transform, neutral=True, size=65), f"neutral {transform}")
    return path


def sdr_chain(clip: dict, width: int, height: int | None = None, pix: str = "rgb24") -> str:
    """Filter chain: (autorotated) source -> SDR BT.709 preview at given width."""
    col = clip["video"]["color"]
    transform = col["transform"] if col["transform"] in ("hlg", "pq") else "sdr709"
    h = height or -2
    chain = [f"scale={width}:{h}:flags=bicubic", to_rgb_filter(col, "gbrpf32le"),
             f"lut3d=file='{neutral_lut(transform)}':interp=tetrahedral"]
    if pix == "rgb24":
        chain.append("format=rgb24")
    else:
        chain.append(from_rgb_filter(pix))
    return ",".join(chain)


def src_path(clip: dict) -> Path:
    p = Path(clip["path"])
    return p if p.is_absolute() else REPO / p


def grab_frames(clip: dict, times: list[float], width: int) -> list[np.ndarray]:
    dw, dh = clip["video"]["display_size"]
    height = int(round(width * dh / dw / 2)) * 2
    frames = []
    for t in times:
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src_path(clip)), "-frames:v", "1",
               "-vf", sdr_chain(clip, width, height), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if len(raw) >= width * height * 3:
            frames.append(np.frombuffer(raw[: width * height * 3], np.uint8).reshape(height, width, 3))
        else:
            frames.append(np.zeros((height, width, 3), np.uint8))
    return frames


def stream_frames(clip: dict, width: int, fps: float | None, gray: bool):
    dw, dh = clip["video"]["display_size"]
    height = int(round(width * dh / dw / 2)) * 2
    vf = sdr_chain(clip, width, height)
    if fps:
        vf = f"fps={fps}," + vf
    if gray:
        vf += ",format=gray"
    pix = "gray" if gray else "rgb24"
    ch = 1 if gray else 3
    cmd = ["ffmpeg", "-v", "error", "-i", str(src_path(clip)), "-vf", vf, "-an",
           "-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = width * height * ch
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(height, width) if gray else a.reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


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
    fps = clip["video"]["fps"] or 30.0
    lum, clip_hi, crush, sharp, dx, dy, conf = [], [], [], [], [], [], []
    prev = None
    win = None
    for g in stream_frames(clip, width, None, gray=True):
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
    n = len(lum)
    if n == 0:
        return {"error": "no frames decoded"}
    t = np.arange(n) / fps
    dxa, dya = np.array(dx), np.array(dy)
    speed = np.hypot(dxa, dya) * fps / width          # frame-widths per second
    k = max(3, int(round(fps * 0.5)) | 1)
    kern = np.ones(k) / k
    traj_x, traj_y = np.cumsum(dxa), np.cumsum(dya)
    smooth_x = np.convolve(np.pad(traj_x, k // 2, mode="edge"), kern, "valid")
    smooth_y = np.convolve(np.pad(traj_y, k // 2, mode="edge"), kern, "valid")
    jitter = np.hypot(traj_x - smooth_x, traj_y - smooth_y) / width * 1000   # milli-frame-widths
    sharp_a = np.array(sharp)
    sharp_rel = sharp_a / (np.percentile(sharp_a, 90) + 1e-9)

    def win_stats(a, step):
        m = int(round(step * fps))
        return [float(np.mean(a[i:i + m])) for i in range(0, n, m)]

    step = 0.5
    series = {
        "t": [round(i * step, 2) for i in range(math.ceil(n / round(step * fps)))],
        "luma": win_stats(np.array(lum), step), "clipped": win_stats(np.array(clip_hi), step),
        "crushed": win_stats(np.array(crush), step), "sharpness": win_stats(sharp_rel, step),
        "speed": win_stats(speed, step), "jitter": win_stats(jitter, step),
    }
    # ---- suggested strongest moments (advisory; final picks are made by eye)
    best = []
    dur = n / fps
    for length in (2.0, 3.0, 4.5):
        if dur < length:
            continue
        m = int(length * fps)
        for s in range(0, n - m + 1, max(1, int(fps / 4))):
            sl = slice(s, s + m)
            score = (1.0 * np.clip(sharp_rel[sl].mean(), 0, 1.2)
                     - 0.04 * jitter[sl].mean()
                     - 3.0 * np.array(clip_hi[sl.start:sl.stop]).mean()
                     - 1.5 * max(0.0, 0.18 - np.array(lum[sl.start:sl.stop]).mean())
                     - 0.3 * max(0.0, speed[sl].mean() - 0.35))
            best.append((float(score), round(s / fps, 2), length))
    best.sort(reverse=True)
    picks = []
    for sc, st, ln in best:
        if all(st + ln <= p["in"] or st >= p["in"] + p["len"] for p in picks):
            picks.append({"in": st, "len": ln, "score": round(sc, 3)})
        if len(picks) >= 4:
            break
    return {"fps": fps, "frames": n, "duration": round(dur, 3), "series": series,
            "summary": {"luma_mean": round(float(np.mean(lum)), 3), "clipped_max": round(float(np.max(clip_hi)), 4),
                        "jitter_median": round(float(np.median(jitter)), 2),
                        "speed_median": round(float(np.median(speed)), 3)},
            "suggested_moments": picks}


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
    return Path(path).stem.replace(" ", "_")


def review_clip(clip: dict, args) -> None:
    name = safe_name(clip["path"])
    out = WORK / "review" / name
    out.mkdir(parents=True, exist_ok=True)
    dur = clip["video"]["duration"] or clip["duration"]
    dw, dh = clip["video"]["display_size"]
    vertical = dh > dw
    # contact sheet
    n = 24 if dur >= 12 else max(6, int(dur * 2))
    times = [min(dur - 0.05, (i + 0.5) * dur / n) for i in range(n)]
    tw = 270 if vertical else 400
    frames = grab_frames(clip, times, tw)
    c = clip["video"]["color"]
    title = (f"{clip['path']}  |  {dw}x{dh} {clip['video']['fps']}fps  {dur:.1f}s  "
             f"{c['transform'].upper()} {c['bit_depth']}-bit")
    tile(frames, [f"{t:.1f}s" for t in times], 8 if vertical else 6, title).save(out / "contact.png")
    # dense filmstrips
    sw = 180 if vertical else 300
    strip, labels, page = [], [], 0
    per_page = 64 if vertical else 48
    for i, fr in enumerate(stream_frames(clip, sw, args.strip_fps, gray=False)):
        strip.append(fr)
        labels.append(f"{i / args.strip_fps:.2f}")
        if len(strip) == per_page:
            tile(strip, labels, 16 if vertical else 8, f"{name} @ {args.strip_fps} fps  page {page}").save(
                out / f"strip_{page:02d}.png")
            strip, labels, page = [], [], page + 1
    if strip:
        tile(strip, labels, 16 if vertical else 8, f"{name} @ {args.strip_fps} fps  page {page}").save(
            out / f"strip_{page:02d}.png")
    # metrics
    m = analyse(clip)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    if "series" in m:
        plot_metrics(m, f"{name}: luma / clipped / crushed / sharpness / speed / jitter", out / "metrics.png")
    # proxy
    if not args.no_proxy:
        pdir = WORK / "proxies"
        pdir.mkdir(parents=True, exist_ok=True)
        pw = 540 if vertical else 960
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src_path(clip)),
                        "-vf", sdr_chain(clip, pw, None, pix="yuv420p"), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                        str(pdir / f"{name}.mp4")], check=False)
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
    for clip in clips:
        review_clip(clip, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
