#!/usr/bin/env python3
"""Step 3 — study the soundtrack.

    python3 tools/audio_analysis.py [FILE ...] [--len SECONDS]
        default: every audio file in work/inventory.json, plus the audio of clips with --clips.
        --len: the planned reel length L; excerpts are then sized to it instead of 30–45 s (at most
        L+1 s, shorter only by the pause after the excerpt, so the reel ends inside that pause).

For each track writes work/audio/<name>.json and <name>_NN.png pages:
  waveform + RMS (dBFS), log-frequency spectrogram, onset strength (SuperFlux, robust to vibrato),
  beat grid (only when the pulse is reliable: recitation, free nasheed and pads get none),
  pauses / phrase spans (critical for recitation: never cut inside a phrase; without a beat, audible
  breaths inside a pause are bridged and listed as "breaths"), energy-change "section" candidates, EBU R128 loudness,
  and ranked excerpts that start just before a phrase onset (pre-roll inside the pause) and end on a
  phrase end before a pause, or on bar lines when there is a reliable beat.
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
WIN = np.hanning(NFFT)
BEAT_MIN_STRENGTH = 0.35   # autocorrelation strength below which there is no real pulse (vibrato, legato voice, pads)
BEAT_LATENCY = 0.012       # s; SuperFlux peak (sub-frame) -> attack; measured on click, drum and frame-drum tracks
PREROLL = 0.12             # s; phrase / excerpt starts sit this far before the onset, inside the pause
BAR_PREROLL = 0.02         # s; bar-line in/out points sit just before the downbeat so its attack is kept
MAX_BREATH = 0.45          # s; longest unvoiced island inside a pause that is bridged as a breath


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
    frames = pad[idx] * WIN.astype(np.float32)
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)   # (n, NFFT/2+1)


def rms_dbfs(mag: np.ndarray) -> np.ndarray:
    """Per-frame RMS level in true dBFS (a full-scale sine reads -3.01) from the Hann-windowed STFT.

    Parseval: sum|X|^2 over the one-sided spectrum ~ NFFT/2 * sum((x*w)^2), and the window keeps
    sum(w^2) of the frame's energy."""
    pw = np.einsum("ij,ij->i", mag, mag, dtype=np.float64)
    rms = np.sqrt(2.0 * pw / (NFFT * float(np.sum(WIN ** 2))))
    return 20 * np.log10(np.maximum(rms, 1e-7))


def voicing(mag: np.ndarray) -> np.ndarray:
    """Per-frame periodicity 0..1: peak of the normalised autocorrelation for pitches of 70-500 Hz,
    corrected for the window's own autocorrelation. Voice and instruments read ~0.6-1, breath noise ~0.2."""
    lo, hi = int(SR / 500), int(SR / 70)
    wac = np.fft.irfft(np.abs(np.fft.rfft(WIN)) ** 2)[:hi + 1]
    r = np.fft.irfft(np.square(mag, dtype=np.float64), axis=1)[:, :hi + 1]
    r = r / (r[:, :1] + 1e-12) / (wac / wac[0])
    return r[:, lo:].max(axis=1)


def mel_filterbank(n_mels: int = 80, fmin: float = 30.0, fmax: float = 11000.0) -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)     # noqa: E731
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)   # noqa: E731
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(min(fmax, SR / 2)), n_mels + 2))
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    fb = np.zeros((n_mels, len(f)), np.float32)
    for i in range(n_mels):
        a, b, c = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((f - a) / (b - a), (c - f) / (c - b)), 0, None)
    return fb


MEL = mel_filterbank()


def onset_envelope(mag: np.ndarray, lag: int = 2) -> np.ndarray:
    """SuperFlux onset strength (Böck & Widmer 2013): log-mel flux against a reference two frames back
    that is max-filtered over +-1 mel band. A partial that drifts by a band (singer vibrato at 4.5-7 Hz,
    glides) is cancelled by the max filter, so only real attacks (consonants, drums, plucks) give flux;
    plain linear-bin flux turned vibrato into a periodic 'beat'."""
    S = np.log1p(100.0 * (mag @ MEL.T))
    if len(S) <= 16:                     # shorter than the trend filter (< 0.4 s): no onsets to speak of
        return np.zeros(len(S))
    padded = np.pad(S, ((0, 0), (1, 1)), mode="edge")
    ref = np.maximum(np.maximum(padded[:, :-2], padded[:, 1:-1]), padded[:, 2:])
    flux = np.maximum(S[lag:] - ref[:-lag], 0.0).sum(axis=1)
    flux = np.concatenate([np.zeros(lag), flux])
    flux -= np.convolve(flux, np.ones(16) / 16, "same")          # remove slow trend
    flux = np.maximum(flux, 0.0)
    return flux / (flux.max() + 1e-9)


def tempo_and_beats(env: np.ndarray, fps: float, pauses: list = ()) -> tuple[float | None, list[float], float]:
    """Autocorrelation tempo + Ellis-style dynamic-programming beat tracking.

    Returns (tempo_bpm, beat_times, strength), strength = normalised autocorrelation at the chosen lag.
    Below BEAT_MIN_STRENGTH there is no reliable pulse and (None, [], strength) is returned rather than a
    confident fake grid. The tempo comes from a parabolic fit of the autocorrelation peak (sub-frame).
    Beat times are refined to the sub-frame peak and shifted by BEAT_LATENCY so they land on the attacks.
    A chain may start at the first onset after silence instead of being dragged back into it, but it
    cannot stop, so beats are then removed where nothing plays: before the first and after the last onset
    activity, and inside pauses longer than one bar (a rest of a beat or two keeps its grid, a stop not)."""
    env_c = env - env.mean()
    ac = np.correlate(env_c, env_c, "full")[len(env_c) - 1:]
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60.0 * fps / np.maximum(lags, 1), 0)
    valid = (bpm >= 50) & (bpm <= 180)
    if not valid.any():
        return None, [], 0.0
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 90.0) / 0.9) ** 2)
    score = np.where(valid, ac * prior, -np.inf)
    lag = int(np.argmax(score))
    strength = round(float(ac[lag] / (ac[0] + 1e-9)), 3)
    if strength < BEAT_MIN_STRENGTH:
        return None, [], strength
    period = float(lag)
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        den = a - 2 * b + c
        if den < 0:
            period += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    tempo = 60.0 * fps / period
    n = len(env)
    cumscore = env.copy()
    backlink = -np.ones(n, dtype=int)
    tight = 100.0
    for i in range(n):
        lo, hi = i - 2 * lag, i - lag // 2
        if hi <= 0:
            continue
        cand = np.arange(max(0, lo), hi)
        pen = -tight * (np.log((i - cand) / period)) ** 2
        vals = cumscore[cand] + pen
        j = int(np.argmax(vals))
        if vals[j] > 0:                  # otherwise start a new chain here (first onset after silence)
            cumscore[i] = env[i] + vals[j]
            backlink[i] = cand[j]
    tail = max(1, n - lag)
    i = tail + int(np.argmax(cumscore[tail:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = backlink[i]
    bf = np.array(beats[::-1])
    keep = np.ones(len(bf), bool)
    w = max(1, int(round(0.1 * fps)))
    sm = np.convolve(env, np.ones(w) / w, "same")                # 100 ms onset activity
    act = np.flatnonzero(sm > 0.15 * np.median(env[bf]))
    if act.size:
        keep &= (bf >= act[0] - w) & (bf <= act[-1] + w)
    sub = np.zeros(len(bf))                                      # sub-frame peak position (parabola)
    inner = (bf > 0) & (bf < n - 1)
    ea, eb, ec = env[bf[inner] - 1], env[bf[inner]], env[bf[inner] + 1]
    den = ea - 2 * eb + ec
    sub[inner] = np.where(den < 0, np.clip(0.5 * (ea - ec) / np.where(den < 0, den, -1), -0.5, 0.5), 0.0)
    times = (bf + sub) / fps + BEAT_LATENCY
    for a, b in pauses:
        if b - a > 4 * period / fps:
            keep &= ~((times > a + 0.1) & (times < b - 0.1))
    return round(tempo, 2), [round(float(t), 3) for t in times[keep]], strength


def bar_lines(beats: list, mag: np.ndarray, fps: float) -> list[float]:
    """Estimated bar starts (4/4 assumed): in each unbroken run of beats, the one of the 4 phases whose
    beats carry the most low-frequency onset energy (< 250 Hz: kick, bass and chord changes land on the
    downbeat, while the backbeat snare / tak is mostly high-frequency)."""
    if len(beats) < 8:
        return []
    low = np.fft.rfftfreq(NFFT, 1 / SR)[np.argmax(MEL, axis=1)] < 250
    S = np.log1p(100.0 * (mag @ MEL[low].T))
    lflux = np.concatenate([np.zeros(2), np.maximum(S[2:] - S[:-2], 0.0).sum(axis=1)])
    bt = np.asarray(beats)
    per = float(np.median(np.diff(bt)))
    runs = np.split(np.arange(len(bt)), np.flatnonzero(np.diff(bt) > 1.5 * per) + 1)
    fr = np.clip(np.round((bt - BEAT_LATENCY) * fps).astype(int), 1, len(lflux) - 2)
    strength = np.maximum(np.maximum(lflux[fr - 1], lflux[fr]), lflux[fr + 1])
    bars = []
    for r in runs:
        if len(r) < 4:
            continue
        phase = int(np.argmax([strength[r[p::4]].mean() for p in range(4)]))
        bars += [float(t) for t in bt[r[phase::4]]]
    return bars


def preroll(a: float, b: float) -> float:
    """In-point for the phrase after pause [a, b]: up to PREROLL before the onset, never before the pause."""
    return round(max(a, b - min(PREROLL, 0.5 * (b - a))), 3)


def runs_of(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(mask)]])
    return [(bool(mask[a]), int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def pauses_and_phrases(rms_db: np.ndarray, fps: float, mag: np.ndarray | None = None,
                       min_pause: float = 0.22) -> tuple[list, list, float, list]:
    """Pauses = runs below a threshold set between the track's quiet floor and its voice level.

    Every gate is relative to the track (95th percentile), so the result does not depend on the
    recording level. A short island above the threshold with >= 0.1 s of quiet on both sides, quieter
    than the voice level and unvoiced (an audible in-breath, a click; voicing() tells it from a sung or
    recited syllable) is bridged into the pause and reported in `breaths`, so no in/out point lands
    inside an inhalation. Without `mag` (the STFT magnitudes) nothing is bridged. Phrase starts carry a
    pre-roll (see preroll()) so a cut there keeps the first consonant.
    Returns (pauses, phrases, threshold_dbfs, breaths)."""
    if rms_db.size == 0 or np.percentile(rms_db, 95) < -90:     # no signal (digital silence)
        return [], [], -90.0, []
    active = rms_db[rms_db > np.percentile(rms_db, 95) - 25]
    floor = float(np.percentile(active, 10))
    ref = float(np.percentile(active, 70))
    thr = floor + 0.35 * (ref - floor)
    thr = min(thr, ref - 10.0)
    quiet = rms_db < thr
    islands = []
    if mag is not None:
        loud = ~quiet                                            # island candidates, ragged edges closed
        for q, a, b in runs_of(quiet):
            if q and 0 < a and b < len(quiet) and (b - a) / fps < 0.06:
                loud[a:b] = True
        runs = runs_of(loud)
        flank = 0.1 * fps
        for k in range(1, len(runs) - 1):
            q, a, b = runs[k]
            if (q and not runs[k - 1][0] and not runs[k + 1][0]
                    and runs[k - 1][2] - runs[k - 1][1] >= flank and runs[k + 1][2] - runs[k + 1][1] >= flank
                    and (b - a) / fps <= MAX_BREATH and rms_db[a:b].max() < ref - 1
                    and float(np.median(voicing(mag[a:b]))) < 0.45):
                quiet[a:b] = True
                islands.append((a, b))
    pauses, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) / fps >= min_pause:
                pauses.append([round(start / fps, 3), round(i / fps, 3)])
            start = None
    breaths = [[round(a / fps, 3), round(b / fps, 3)] for a, b in islands
               if any(p0 <= a / fps and b / fps <= p1 for p0, p1 in pauses)]
    phrases, cursor = [], 0.0
    for a, b in pauses:
        if a - cursor > 0.25:
            phrases.append([round(cursor, 3), a])
        cursor = preroll(a, b)
    end = len(rms_db) / fps
    if end - cursor > 0.25:
        phrases.append([round(cursor, 3), round(end, 3)])
    return pauses, phrases, round(thr, 1), breaths


def sections(rms_db: np.ndarray, fps: float) -> list[dict]:
    w = int(fps * 2)
    if len(rms_db) < 3 * w:
        return []
    x = np.maximum(rms_db, np.percentile(rms_db, 95) - 40)   # silence must not dominate (relative to the track)
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


def level_minima(rms_db: np.ndarray, fps: float, min_depth: float = 1.0) -> list[tuple[float, float]]:
    """Dips of the 1 s smoothed level (a breath between legato phrases, a chord change in a pad):
    [(time, depth_db)], each the lowest point within +-1.5 s, at least 1.5 s apart."""
    w = max(1, int(fps))
    sm = np.convolve(np.pad(rms_db, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, "valid")
    k = int(1.5 * fps)
    if len(sm) < 2 * k + 1:
        return []
    view = np.lib.stride_tricks.sliding_window_view
    local_min = view(np.pad(sm, k, mode="edge"), 2 * k + 1).min(axis=1)
    wide_max = view(np.pad(sm, 3 * k, mode="edge"), 6 * k + 1).max(axis=1)
    out: list[tuple[float, float]] = []
    for i in np.flatnonzero(sm <= local_min):
        depth = float(wide_max[i] - sm[i])
        if depth >= min_depth and (not out or i / fps - out[-1][0] >= 1.5):
            out.append((i / fps, depth))
    return out


def excerpts(duration: float, pauses: list, phrases: list, beats: list, rms_db: np.ndarray, fps: float,
             thr: float | None = None, secs: list = (), bars: list = (), length: float | None = None,
             lo: float = 30.0, hi: float = 45.0) -> list[dict]:
    """Ranked music excerpts for the reel ('start'/'end' = 'in'/'out', in source seconds).

    Starts: just before a phrase onset (pause end minus the pre-roll, inside the pause) or the file start.
    Ends: a phrase end (pause start) or the end of the file. With a reliable beat, the cut points are bar
    lines (and energy changes snapped to a bar) plus the first / last sound; inside the music only a stop
    of a bar or more counts as a pause (shorter ones are rests or gaps between hits); if whole bars give
    too few windows, every beat is a candidate. Material without a beat and with too few windows gets the
    dips of the 1 s smoothed level and the energy changes (and, with `length`, an out wherever the reel
    ends).
    Length: lo..hi; with `length` (the reel, L) at most L+1 s (the end fade hides the cut) and shorter
    than L only by the pause after the end (min 0.1 s), so the reel ends inside that pause.
    Score: cadence (pause after the end) > clean start (pause before) > level arc on sounding frames >
    whole phrases, near the target length; a pause of more than 2 s inside (dead air) costs.
    Picks overlap each other by at most 50 % (IoU)."""
    period = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0
    edge = 1.5 / fps
    min_gap = 4 * period if len(bars) else period
    use = [p for p in pauses if p[1] - p[0] >= min_gap or p[0] <= edge or p[1] >= duration - edge]
    starts: dict = {}
    ends: dict = {}

    def add(d, t, gap=0.0, kind="phrase", depth=0.0):
        t = round(min(max(float(t), 0.0), duration), 3)
        c = d.setdefault(t, {"gap": 0.0, "kinds": set(), "depth": 0.0})
        c["gap"] = max(c["gap"], gap)
        c["kinds"].add(kind)
        c["depth"] = max(c["depth"], depth)

    if not (use and use[0][0] <= edge):
        add(starts, 0.0, 0.5, "file_start")
    if not (use and use[-1][1] >= duration - edge):
        add(ends, duration, 0.5, "file_end")
    for a, b in use:
        if b < duration - edge:
            add(starts, preroll(a, b), b - a, "phrase")
        if a > edge:
            add(ends, a, b - a, "phrase_end")
    barr = np.asarray(bars, float)
    for t in barr:
        add(starts, t - BAR_PREROLL, kind="bar")
        add(ends, t - BAR_PREROLL, kind="bar")
    if barr.size:
        for s in secs:
            t = float(barr[np.argmin(np.abs(barr - s["t"]))])
            if abs(t - s["t"]) <= 2 * period:
                add(starts, t - BAR_PREROLL, kind="section")
                add(ends, t - BAR_PREROLL, kind="section")

    target = float(length) if length else 38.0

    def fits(L, gap_after):
        if length:
            return length - max(0.1, min(gap_after, 1.5)) <= L <= length + 1.0
        return lo <= L <= hi

    few = sum(fits(e - s, ec["gap"]) for s in starts for e, ec in ends.items()) < 3
    if few and barr.size:
        for t in beats:                  # whole bars do not give the length: stay on the beat grid
            add(starts, t - BAR_PREROLL, kind="beat")
            add(ends, t - BAR_PREROLL, kind="beat")
    elif few:
        for t, depth in level_minima(rms_db, fps):
            add(starts, t, kind="level_min", depth=depth)
            add(ends, t, kind="level_min", depth=depth)
        for s in secs:
            add(starts, s["t"], kind="section")
            add(ends, s["t"], kind="section")
        if length:                       # continuous material: the reel's end (under its fade) is the out
            for s in list(starts):
                add(ends, s + length, kind="reel_end")

    act = rms_db > (thr if thr is not None else -np.inf)
    res = []
    for s, sc in starts.items():
        for e, ec in ends.items():
            L = e - s
            if not fits(L, ec["gap"]):
                continue
            a, b = int(round(s * fps)), int(round(e * fps))
            seg = rms_db[a:b][act[a:b]]
            if seg.size < 10:
                continue
            q = len(seg) // 4
            arc = float(np.mean(seg[q:3 * q]) - 0.5 * (np.mean(seg[:q]) + np.mean(seg[3 * q:])))
            n_phr = sum(1 for p in phrases if p[0] >= s - 0.05 and p[1] <= e + 0.05)
            on_bars = "bar" in sc["kinds"] and "bar" in ec["kinds"]
            n_sec = ("section" in sc["kinds"]) + ("section" in ec["kinds"])
            dead = max([min(p[1], e) - max(p[0], s) for p in pauses], default=0.0)   # longest silence inside
            score = (0.3 * min(ec["gap"], 1.2) + 0.15 * min(sc["gap"], 1.2) + 0.08 * arc + 0.02 * n_phr
                     + 0.2 * on_bars + 0.1 * n_sec + 0.03 * (min(sc["depth"], 6) + min(ec["depth"], 6))
                     - 0.15 * max(0.0, dead - 2.0) - (0.1 if length else 0.01) * abs(L - target))
            res.append({"start": s, "end": e, "in": s, "out": e, "len": round(L, 2), "phrases": n_phr,
                        "arc_db": round(arc, 1), "pause_before_s": round(sc["gap"], 2),
                        "pause_after_s": round(ec["gap"], 2), "ends_on_decay": ec["gap"] >= 0.3,
                        "on_bars": on_bars, "start_kind": "+".join(sorted(sc["kinds"])),
                        "end_kind": "+".join(sorted(ec["kinds"])), "score": round(score, 3)})
    res.sort(key=lambda r: -r["score"])

    def iou(r, p):
        inter = min(r["end"], p["end"]) - max(r["start"], p["start"])
        return max(0.0, inter) / (max(r["end"], p["end"]) - min(r["start"], p["start"]))

    picked: list[dict] = []
    for r in res:
        if all(iou(r, p) <= 0.5 for p in picked):
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
                 beats: list, pauses: list, phrases: list, secs: list, out_dir: Path, page_len: float = 30.0,
                 breaths: list = ()):
    fps = SR / HOP
    dur = len(y) / SR
    rlo = float(np.percentile(rms_db, 95)) - 40 if rms_db.size else -60.0   # RMS lane: rlo .. rlo+45 dBFS
    legend = "yellow ticks=beats" if beats else "no reliable beat"
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
        d.text((10, 6), f"{name}   {p0:.0f}–{p1:.0f}s   (blue=phrases, red=pauses, white=breaths, {legend})",
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
        pts = [(X(i / fps), ry0 + rh - (min(max(rms_db[i], rlo), rlo + 45) - rlo) / 45 * rh)
               for i in range(fa, min(fb, len(rms_db)))]
        if len(pts) > 1:
            d.line(pts, fill=(240, 240, 240), width=2)
        d.text((4, ry0), "RMS dB", fill=(200, 200, 200), font=font(12))
        d.text((4, ry0 + 14), f"{rlo + 45:.0f}", fill=(150, 150, 150), font=font(11))
        d.text((4, ry0 + rh - 12), f"{rlo:.0f}", fill=(150, 150, 150), font=font(11))
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
        for s0, s1 in breaths:
            if s1 > p0 and s0 < p1:
                d.rectangle([X(max(s0, p0)), 32, X(min(s1, p1)), 37], fill=(235, 235, 235))
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


def analyse(path: Path, length: float | None = None) -> dict:
    y = load_mono(path)
    dur = len(y) / SR
    mag = stft_mag(y)
    fps = SR / HOP
    rms_db = rms_dbfs(mag)
    env = onset_envelope(mag)
    pauses, phrases, thr, breaths = pauses_and_phrases(rms_db, fps, mag)
    tempo, beats, strength = tempo_and_beats(env, fps, pauses)
    if beats and breaths:        # with a pulse, short unvoiced islands are hi-hats / shakers, not breaths
        pauses, phrases, thr, breaths = pauses_and_phrases(rms_db, fps)
    bars = bar_lines(beats, mag, fps)
    secs = sections(rms_db, fps)
    centroid = (mag * np.fft.rfftfreq(NFFT, 1 / SR)[None, :]).sum(1) / (mag.sum(1) + 1e-9)
    name = path.stem.replace(" ", "_")
    out_dir = WORK / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = render_pages(name, y, mag, env, rms_db, beats, pauses, phrases, secs, out_dir, breaths=breaths)
    res = {
        "file": str(path.relative_to(REPO)) if REPO in path.resolve().parents else str(path),
        "duration": round(dur, 3), "loudness": loudness(path),
        "tempo_bpm": tempo, "beat_strength": strength, "beat_reliable": bool(beats),
        "beats": beats, "bars": [round(b, 3) for b in bars],
        "pause_threshold_db": thr, "pauses": pauses, "phrases": phrases, "breaths": breaths,
        "energy_changes": secs,
        "spectral_centroid_hz_median": round(float(np.median(centroid)), 1),
        "reel_length_s": length,
        "excerpt_length_s": ([round(length - 1.5, 2), round(length + 1.0, 2)] if length else [30.0, 45.0]),
        "suggested_excerpts": excerpts(dur, pauses, phrases, beats, rms_db, fps, thr, secs, bars, length),
        "pages": pages,
        "note": (f"tempo_bpm / beats / bars are given only when beat_strength >= {BEAT_MIN_STRENGTH} (a real "
                 "pulse); otherwise tempo_bpm is null and beats is empty: there is no reliable beat (recitation, "
                 "free-flowing nasheed, vibrato, pads), so cut on phrase ends / pauses instead. bars assume 4/4. "
                 f"Phrase and excerpt starts sit up to {PREROLL} s before the onset, inside the pause; without a "
                 "reliable beat, audible breaths inside pauses are bridged (listed in breaths). Levels are dBFS."),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    require("ffmpeg")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--clips", action="store_true", help="also analyse the audio of every video clip")
    ap.add_argument("--len", type=float, default=None, metavar="SECONDS",
                    help="planned reel length: size the suggested excerpts to it (default 30-45 s)")
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
        r = analyse(f, args.len)
        beat = (f"tempo {r['tempo_bpm']}" if r["tempo_bpm"] else "no reliable beat") + f" (strength {r['beat_strength']})"
        print(f"{r['file']}: {r['duration']}s  {r['loudness']}  {beat}"
              f"  pauses {len(r['pauses'])}  phrases {len(r['phrases'])}  breaths {len(r['breaths'])}")
        for e in r["suggested_excerpts"][:4]:
            print("   excerpt", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
