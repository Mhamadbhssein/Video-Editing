#!/usr/bin/env python3
"""Step 3 — study the soundtrack.

    python3 tools/audio_analysis.py [FILE ...]     (default: every audio file in work/inventory.json,
                                                   plus the audio of clips with --clips)

For each track writes work/audio/<name>.json and <name>_NN.png pages:
  waveform + RMS (dB), log-frequency spectrogram, onset strength, beat grid,
  pauses / phrase spans (critical for recitation: never cut inside a phrase),
  energy-change "section" candidates, EBU R128 loudness, and ranked excerpts of
  30–45 s that start after a pause and end on a phrase end.
numpy only (no librosa) so it runs anywhere FFmpeg runs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediainfo import REPO, require  # noqa: E402
from review import font  # noqa: E402

WORK = REPO / "work"
SR = 22050
HOP = 512
NFFT = 2048


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def loudness(path: Path) -> dict:
    p = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", str(path), "-vn",
                        "-af", "ebur128=peak=true", "-f", "null", "-"], stderr=subprocess.PIPE, text=True)
    txt = p.stderr[p.stderr.rfind("Summary:"):]
    def grab(key):
        m = re.search(key + r":\s+(-?[\d.]+|-inf)", txt)
        return float(m.group(1)) if m and m.group(1) != "-inf" else None
    return {"integrated_lufs": grab(r"I"), "lra_lu": grab(r"LRA"), "true_peak_dbfs": grab(r"Peak")}


def stft_mag(y: np.ndarray) -> np.ndarray:
    pad = np.pad(y, NFFT // 2)
    n = 1 + (len(pad) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    win = np.hanning(NFFT).astype(np.float32)
    frames = pad[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)   # (n, NFFT/2+1)


def onset_envelope(mag: np.ndarray) -> np.ndarray:
    logm = np.log1p(100.0 * mag)
    flux = np.maximum(np.diff(logm, axis=0), 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    flux -= np.convolve(flux, np.ones(16) / 16, "same")          # remove slow trend
    flux = np.maximum(flux, 0.0)
    return flux / (flux.max() + 1e-9)


def tempo_and_beats(env: np.ndarray, fps: float) -> tuple[float, list[float], float]:
    """Autocorrelation tempo + Ellis-style dynamic-programming beat tracking."""
    env_c = env - env.mean()
    ac = np.correlate(env_c, env_c, "full")[len(env_c) - 1:]
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60.0 * fps / np.maximum(lags, 1), 0)
    valid = (bpm >= 50) & (bpm <= 180)
    if not valid.any():
        return 0.0, [], 0.0
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 90.0) / 0.9) ** 2)
    score = np.where(valid, ac * prior, -np.inf)
    lag = int(np.argmax(score))
    strength = float(ac[lag] / (ac[0] + 1e-9))
    tempo = 60.0 * fps / lag
    period = lag
    n = len(env)
    cumscore = env.copy()
    backlink = -np.ones(n, dtype=int)
    tight = 100.0
    for i in range(n):
        lo, hi = i - 2 * period, i - period // 2
        if hi <= 0:
            continue
        cand = np.arange(max(0, lo), hi)
        pen = -tight * (np.log((i - cand) / period)) ** 2
        vals = cumscore[cand] + pen
        j = int(np.argmax(vals))
        cumscore[i] = env[i] + vals[j]
        backlink[i] = cand[j]
    tail = max(1, n - period)
    i = tail + int(np.argmax(cumscore[tail:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = backlink[i]
    beats = beats[::-1]
    return round(tempo, 2), [round(b / fps, 3) for b in beats], round(strength, 3)


def pauses_and_phrases(rms_db: np.ndarray, fps: float, min_pause: float = 0.22) -> tuple[list, list, float]:
    active = rms_db[rms_db > -70]
    if active.size == 0:
        return [], [], -60.0
    floor = float(np.percentile(active, 10))
    ref = float(np.percentile(active, 70))
    thr = floor + 0.35 * (ref - floor)
    thr = min(thr, ref - 10.0)
    quiet = rms_db < thr
    pauses, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) / fps >= min_pause:
                pauses.append([round(start / fps, 3), round(i / fps, 3)])
            start = None
    phrases, cursor = [], 0.0
    for a, b in pauses:
        if a - cursor > 0.25:
            phrases.append([round(cursor, 3), a])
        cursor = b
    end = len(rms_db) / fps
    if end - cursor > 0.25:
        phrases.append([round(cursor, 3), round(end, 3)])
    return pauses, phrases, round(thr, 1)


def sections(rms_db: np.ndarray, fps: float) -> list[dict]:
    w = int(fps * 2)
    if len(rms_db) < 3 * w:
        return []
    x = np.maximum(rms_db, -60.0)                       # digital silence must not dominate
    sm = np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, "valid")
    d = np.abs(sm[w:] - sm[:-w])
    out = []
    order = np.argsort(d)[::-1]
    for i in order:
        t = (i + w / 2) / fps
        if d[i] < 3.0:
            break
        if all(abs(t - o["t"]) > 4.0 for o in out):
            out.append({"t": round(t, 2), "delta_db": round(float(sm[i + w] - sm[i]), 1)})
        if len(out) >= 12:
            break
    return sorted(out, key=lambda o: o["t"])


def excerpts(duration: float, pauses: list, phrases: list, beats: list, rms_db: np.ndarray, fps: float,
             lo: float = 30.0, hi: float = 45.0) -> list[dict]:
    starts = sorted({0.0} | {p[1] for p in pauses})
    ends = sorted({p[0] for p in pauses} | {round(duration, 3)})
    res = []
    for s in starts:
        for e in ends:
            L = e - s
            if L < lo or L > hi:
                continue
            a, b = int(s * fps), int(e * fps)
            seg = rms_db[a:b]
            if seg.size < 10:
                continue
            q = len(seg) // 4
            arc = float(np.mean(seg[q:3 * q]) - 0.5 * (np.mean(seg[:q]) + np.mean(seg[3 * q:])))
            end_decay = float(np.mean(seg[-int(fps):]) < np.mean(seg) - 3)
            n_phr = sum(1 for p in phrases if p[0] >= s - 0.05 and p[1] <= e + 0.05)
            score = 0.08 * arc + 0.5 * end_decay + 0.02 * n_phr - 0.01 * abs(L - 38)
            res.append({"in": s, "out": e, "len": round(L, 2), "phrases": n_phr,
                        "arc_db": round(arc, 1), "ends_on_decay": bool(end_decay), "score": round(score, 3)})
    res.sort(key=lambda r: -r["score"])
    picked = []
    for r in res:
        if all(abs(r["in"] - p["in"]) > 3 for p in picked):
            picked.append(r)
        if len(picked) >= 8:
            break
    return picked


def colormap(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)[..., None]
    stops = np.array([[0, 0, 0], [40, 10, 90], [180, 40, 90], [250, 160, 40], [255, 250, 210]], float)
    pos = v * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = pos - i
    return (stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f).astype(np.uint8)


def render_pages(name: str, y: np.ndarray, mag: np.ndarray, env: np.ndarray, rms_db: np.ndarray,
                 beats: list, pauses: list, phrases: list, secs: list, out_dir: Path, page_len: float = 30.0):
    fps = SR / HOP
    dur = len(y) / SR
    W, H = 1800, 820
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    logf = np.geomspace(40, SR / 2 * 0.95, 220)
    bins = np.searchsorted(freqs, logf)
    spec_db = 20 * np.log10(mag[:, bins] + 1e-6)
    spec_db = (spec_db - (spec_db.max() - 80)) / 80
    pages = []
    p0 = 0.0
    k = 0
    while p0 < dur:
        p1 = min(dur, p0 + page_len)
        img = Image.new("RGB", (W, H), (14, 14, 14))
        d = ImageDraw.Draw(img)
        d.text((10, 6), f"{name}   {p0:.0f}–{p1:.0f}s   (blue=phrases, red=pauses, yellow ticks=beats)",
               fill=(230, 230, 230), font=font(18))
        L, R = 60, W - 20
        def X(t):
            return L + (t - p0) / page_len * (R - L)
        # waveform
        wy0, wh = 40, 220
        a, b = int(p0 * SR), int(p1 * SR)
        seg = y[a:b]
        cols = int(X(p1) - L)
        if cols > 0 and seg.size:
            chunks = np.array_split(seg, cols)
            for c, ch in enumerate(chunks):
                if ch.size:
                    mn, mx = float(ch.min()), float(ch.max())
                    d.line([(L + c, wy0 + wh / 2 - mx * wh / 2), (L + c, wy0 + wh / 2 - mn * wh / 2)],
                           fill=(120, 200, 255))
        # rms dB
        ry0, rh = 270, 150
        fa, fb = int(p0 * fps), int(p1 * fps)
        pts = [(X(i / fps), ry0 + rh - (max(rms_db[i], -60) + 60) / 60 * rh) for i in range(fa, min(fb, len(rms_db)))]
        if len(pts) > 1:
            d.line(pts, fill=(240, 240, 240), width=2)
        d.text((4, ry0), "RMS dB", fill=(200, 200, 200), font=font(12))
        # onset
        oy0, oh = 430, 90
        pts = [(X(i / fps), oy0 + oh - env[i] * oh) for i in range(fa, min(fb, len(env)))]
        if len(pts) > 1:
            d.line(pts, fill=(255, 170, 60), width=1)
        d.text((4, oy0), "onset", fill=(200, 200, 200), font=font(12))
        # spectrogram
        sy0, sh = 530, 250
        sub = spec_db[fa:fb]
        if sub.size:
            im = Image.fromarray(colormap(sub.T[::-1]))
            im = im.resize((max(1, int(X(p1) - L)), sh))
            img.paste(im, (L, sy0))
        d.text((4, sy0), "spec", fill=(200, 200, 200), font=font(12))
        # annotations
        for s0, s1 in phrases:
            if s1 > p0 and s0 < p1:
                d.rectangle([X(max(s0, p0)), 32, X(min(s1, p1)), 37], fill=(90, 150, 255))
        for s0, s1 in pauses:
            if s1 > p0 and s0 < p1:
                d.rectangle([X(max(s0, p0)), 32, X(min(s1, p1)), 37], fill=(255, 70, 70))
        for bt in beats:
            if p0 <= bt <= p1:
                d.line([(X(bt), 260), (X(bt), 268)], fill=(255, 230, 0), width=2)
        for s in secs:
            if p0 <= s["t"] <= p1:
                d.line([(X(s["t"]), 40), (X(s["t"]), 780)], fill=(0, 255, 160), width=1)
                d.text((X(s["t"]) + 3, 425), f"{s['delta_db']:+.0f}dB", fill=(0, 255, 160), font=font(11))
        for sec in range(int(p0), int(p1) + 1):
            x = X(sec)
            d.line([(x, 782), (x, 790)], fill=(150, 150, 150))
            d.text((x - 6, 794), f"{sec}", fill=(170, 170, 170), font=font(12))
        path = out_dir / f"{name}_{k:02d}.png"
        img.save(path)
        pages.append(str(path.relative_to(REPO)))
        p0 = p1
        k += 1
    return pages


def analyse(path: Path) -> dict:
    y = load_mono(path)
    dur = len(y) / SR
    mag = stft_mag(y)
    fps = SR / HOP
    rms = np.sqrt(np.mean((mag ** 2), axis=1) * 2) / (NFFT / 2)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-7))
    env = onset_envelope(mag)
    tempo, beats, strength = tempo_and_beats(env, fps)
    pauses, phrases, thr = pauses_and_phrases(rms_db, fps)
    secs = sections(rms_db, fps)
    centroid = (mag * np.fft.rfftfreq(NFFT, 1 / SR)[None, :]).sum(1) / (mag.sum(1) + 1e-9)
    name = path.stem.replace(" ", "_")
    out_dir = WORK / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = render_pages(name, y, mag, env, rms_db, beats, pauses, phrases, secs, out_dir)
    res = {
        "file": str(path.relative_to(REPO)) if REPO in path.resolve().parents else str(path),
        "duration": round(dur, 3), "loudness": loudness(path),
        "tempo_bpm": tempo, "beat_strength": strength,
        "beats": beats, "pause_threshold_db": thr, "pauses": pauses, "phrases": phrases,
        "energy_changes": secs,
        "spectral_centroid_hz_median": round(float(np.median(centroid)), 1),
        "suggested_excerpts": excerpts(dur, pauses, phrases, beats, rms_db, fps),
        "pages": pages,
        "note": ("beat grid is only meaningful when beat_strength is high (> ~0.3); for recitation "
                 "or free-flowing nasheed, cut on phrase ends / pauses instead."),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    require("ffmpeg")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--clips", action="store_true", help="also analyse the audio of every video clip")
    args = ap.parse_args()
    files = [Path(f).resolve() for f in args.files]
    if not files:
        inv = json.loads((WORK / "inventory.json").read_text())
        files = [Path(a["path"]) if Path(a["path"]).is_absolute() else REPO / a["path"]
                 for a in inv["audio"] if a.get("audio")]
        if args.clips:
            files += [Path(v["path"]) if Path(v["path"]).is_absolute() else REPO / v["path"]
                      for v in inv["video"] if v.get("audio")]
    for f in files:
        r = analyse(f)
        print(f"{r['file']}: {r['duration']}s  {r['loudness']}  tempo {r['tempo_bpm']} (strength {r['beat_strength']})"
              f"  pauses {len(r['pauses'])}  phrases {len(r['phrases'])}")
        for e in r["suggested_excerpts"][:4]:
            print("   excerpt", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
