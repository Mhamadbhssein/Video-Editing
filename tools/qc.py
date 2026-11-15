#!/usr/bin/env python3
"""Step 5 — quality control of the exported reel.

    python3 tools/qc.py [output/Madinah_Final_Reel.mp4] [--edl edit/edl.json]

Checks (report in work/qc/report.md + report.json; non-zero exit on hard failures):
  * delivery spec: H.264 High, yuv420p, size, CFR fps, BT.709 tags, AAC 48 kHz stereo, faststart, duration
  * black frames, freezes, duplicate-frame stutter per shot, unexpected cuts vs. the timeline
  * EBU R128 loudness, true peak (no clipping), unintended silences
  * title boxes inside the Instagram-safe area
and writes inspection sheets (cuts, titles + safe-zone overlay, slow-motion frame strips,
first/last frame) to work/qc/ for visual review.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediainfo import REPO, ffprobe, require  # noqa: E402
from render import FONTS, SAFE, esc  # noqa: E402
from review import font, tile  # noqa: E402

WORK = REPO / "work"
QC = WORK / "qc"


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
    log = QC / "ydif.txt"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                    f"signalstats,metadata=print:key=lavfi.signalstats.YDIF:file='{esc(log)}'",
                    "-an", "-f", "null", "-"], check=True)
    vals = [float(m) for m in re.findall(r"YDIF=([\d.]+)", log.read_text())]
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
    """Render titles alone on black and measure their ink bounding boxes."""
    res = []
    for k, t in enumerate(edl.get("titles", [])):
        for phase, tt in (("start+fade", t["start"] + t.get("fade_in", 0.8) + 0.05),
                          ("mid", 0.5 * (t["start"] + t["end"])), ("end-fade", t["end"] - t.get("fade_out", 0.7) - 0.05)):
            raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                  f"color=black:s=1080x1920:r=30:d={total + 1:.2f}", "-vf",
                                  f"ass='{esc(ass)}':fontsdir='{esc(FONTS)}',trim=start={tt:.3f},setpts=PTS-STARTPTS,format=gray",
                                  "-frames:v", "1", "-f", "rawvideo", "-"], stdout=subprocess.PIPE).stdout
            if len(raw) < 1080 * 1920:
                continue
            a = np.frombuffer(raw[:1080 * 1920], np.uint8).reshape(1920, 1080)
            ys, xs = np.where(a > 40)
            if xs.size == 0:
                res.append({"title": k, "phase": phase, "t": round(tt, 2), "bbox": None, "inside_safe": None})
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
    ap.add_argument("video", nargs="?", default=str(REPO / "output" / "Madinah_Final_Reel.mp4"))
    ap.add_argument("--edl", default=str(REPO / "edit" / "edl.json"))
    args = ap.parse_args()
    video = Path(args.video)
    edl = json.loads(Path(args.edl).read_text())
    tl = json.loads((WORK / "render" / "timeline.json").read_text())
    QC.mkdir(parents=True, exist_ok=True)
    fps = float(edl["fps"])
    W, H = edl.get("size", [1080, 1920])
    fails, warns = [], []

    # ---------------- spec
    info = ffprobe(video)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    spec = {
        "codec": v["codec_name"], "profile": v.get("profile"), "level": v.get("level"), "pix_fmt": v["pix_fmt"],
        "size": [v["width"], v["height"]], "r_frame_rate": v["r_frame_rate"], "avg_frame_rate": v["avg_frame_rate"],
        "frames": int(v.get("nb_frames", 0)), "color": [v.get("color_primaries"), v.get("color_transfer"),
                                                        v.get("color_space"), v.get("color_range")],
        "duration": float(info["format"]["duration"]), "bitrate_mbps": round(int(info["format"]["bit_rate"]) / 1e6, 2),
        "audio": a and {"codec": a["codec_name"], "profile": a.get("profile"), "rate": a["sample_rate"],
                        "channels": a["channels"], "kbps": round(int(a.get("bit_rate", 0)) / 1000)},
        "faststart": moov_first(video),
    }
    exp_frames = tl["total_frames"]
    checks = [
        (spec["codec"] == "h264" and spec["profile"] == "High", "H.264 High profile"),
        (spec["pix_fmt"] == "yuv420p", "yuv420p"),
        (spec["size"] == [W, H] or video.name.find("Master") >= 0, f"size {W}x{H}"),
        (spec["r_frame_rate"] == spec["avg_frame_rate"], "constant frame rate"),
        (abs(float(eval(spec["avg_frame_rate"])) - fps) < 0.01, f"{fps:g} fps"),
        (spec["frames"] == exp_frames, f"frame count {exp_frames}"),
        (spec["color"][:3] == ["bt709", "bt709", "bt709"] and spec["color"][3] == "tv", "BT.709 limited tags"),
        (bool(spec["audio"]) and spec["audio"]["codec"] == "aac" and spec["audio"]["rate"] == "48000"
         and spec["audio"]["channels"] == 2, "AAC 48 kHz stereo"),
        (spec["faststart"], "moov before mdat (faststart)"),
    ]
    for ok, what in checks:
        (None if ok else fails.append(f"spec: {what} — got {spec}"))

    # ---------------- picture problems
    log = lavfi_log(video, vf="blackdetect=d=0.04:pix_th=0.08:pic_th=0.97,freezedetect=n=-55dB:d=0.4")
    blacks = [(float(s), float(e)) for s, e in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]
    freezes = re.findall(r"freeze_start: ([\d.]+)|freeze_end: ([\d.]+)", log)
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
        fails.append(f"black frames {s:.2f}–{e:.2f}s")
    for s, e in fz:
        warns.append(f"freeze/static picture {s:.2f}–{e:.2f}s (fine for a locked-off shot; check it is intended)")

    # duplicate-frame stutter per shot
    ydif = per_frame_ydif(video)
    shot_rows = []
    for sh in tl["shots"]:
        a0 = sh["T_frames"] + (sh["d_in"] // 2)
        a1 = sh["T_frames"] + sh["dur_frames"] - (sh["d_out"] // 2)
        seg = ydif[a0 + 1:a1] if a1 - a0 > 2 else np.array([])
        if seg.size == 0:
            continue
        med = float(np.median(seg))
        dups = int(((seg < 0.02 * max(med, 0.05)) & (med > 0.4)).sum())
        cv = float(np.std(seg) / (np.mean(seg) + 1e-9))
        shot_rows.append({"id": sh["id"], "speed": sh["speed"], "retime": (sh.get("retime") or {}).get("mode"),
                          "motion_median": round(med, 2), "duplicate_frames": dups, "cadence_cv": round(cv, 2)})
        if dups:
            fails.append(f"{sh['id']}: {dups} duplicated frames inside a moving shot (stutter)")

    # cuts vs timeline
    log = lavfi_log(video, vf="scdet=threshold=12")
    det = [round(float(t), 3) for t in re.findall(r"lavfi\.scd\.time: ([\d.]+)", log)]
    planned = [sh["T_frames"] / fps for sh in tl["shots"][1:] if sh["d_in"] == 0]
    unexpected = [t for t in det if all(abs(t - p) > 1.5 / fps for p in planned)]
    missing = [p for p in planned if all(abs(t - p) > 1.5 / fps for t in det)]
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
    sil = re.findall(r"silence_start: ([\d.]+)", alog)
    if loud["TP"] is not None and loud["TP"] > -1.0:
        fails.append(f"true peak {loud['TP']} dBTP > -1.0 (risk of clipping after AAC re-encode)")
    if loud["I"] is not None and not (-17.5 <= loud["I"] <= -12.5):
        warns.append(f"integrated loudness {loud['I']} LUFS outside -17.5..-12.5")
    for t in sil:
        if float(t) < spec["duration"] - 2.5:
            warns.append(f"silence at {float(t):.2f}s (intended pause?)")

    # ---------------- titles vs safe area
    boxes = title_boxes(edl, WORK / "render" / "titles.ass", tl["total"])
    for b in boxes:
        if b["inside_safe"] is False:
            fails.append(f"title {b['title']} ({b['phase']}) bbox {b['bbox']} leaves the Instagram-safe area")

    # ---------------- sheets
    sw, shh = 360, 640
    want = {0, exp_frames - 1}
    pairs = []
    for sh in tl["shots"][1:]:
        T = sh["T_frames"]
        if sh["d_in"]:
            trio = [T - sh["d_in"] // 2, T, T + sh["d_in"] // 2 - 1]
            pairs.append((sh["id"] + " dissolve", trio))
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
    tile([get(0), get(exp_frames - 1)], ["first frame", "last frame"], 2, "first / last").save(QC / "first_last.png")
    frames, labels = [], []
    for name, idxs in pairs:
        for i in idxs:
            frames.append(get(i)); labels.append(f"{name} f{i} {i / fps:.2f}s")
    if frames:
        tile(frames, labels, 6, "cuts & dissolves (frame before | frame at cut)").save(QC / "cuts.png")
    if tmid:
        ims = [np.array(safe_overlay(Image.fromarray(get(i)), sw / 1080)) for i in tmid]
        tile(ims, [f"title {k} @ {i / fps:.2f}s" for k, i in enumerate(tmid)], min(4, len(ims)),
             "titles with Instagram UI zones (red/orange = keep clear)").save(QC / "titles.png")
    for sh in slow:
        mid = sh["T_frames"] + sh["dur_frames"] // 2
        idxs = list(range(mid - 6, mid + 6))
        tile([get(i) for i in idxs], [f"f{i}" for i in idxs], 6,
             f"{sh['id']} x{sh['speed']} consecutive frames — check warping/ghosting").save(QC / f"slowmo_{sh['id']}.png")

    report = {"file": str(video), "spec": spec, "loudness": loud, "blacks": blacks, "freezes": fz,
              "shots": shot_rows, "detected_cuts": det, "planned_cuts": planned, "title_boxes": boxes,
              "fails": fails, "warnings": warns}
    (QC / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    md = [f"# QC — {video.name}", "", "## Spec", "```json", json.dumps(spec, indent=1), "```",
          f"Loudness: {loud}", "", "## Shots", "| id | speed | retime | motion | dup frames | cadence cv |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['speed']} | {r['retime']} | {r['motion_median']} | {r['duplicate_frames']} | "
           f"{r['cadence_cv']} |" for r in shot_rows]
    md += ["", "## Titles", ""] + [f"- {b.get('text', '')} [{b['phase']}] bbox={b['bbox']} safe={b['inside_safe']}"
                                   for b in boxes]
    md += ["", "## FAIL", ""] + [f"- {f}" for f in fails] + ["", "## Warnings", ""] + [f"- {w}" for w in warns]
    md += ["", "Sheets: work/qc/first_last.png, cuts.png, titles.png, slowmo_*.png"]
    (QC / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
