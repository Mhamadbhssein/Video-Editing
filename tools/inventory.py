#!/usr/bin/env python3
"""Step 1 — find and describe every piece of footage and audio (read-only).

    python3 tools/inventory.py [ROOT ...]

Writes work/inventory.json and work/inventory.md and prints a summary with
recommendations (timeline frame rate, HDR handling, whether a 4K master is justified).
Original files are only ever read.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediainfo import REPO, describe, find_media  # noqa: E402

WORK = REPO / "work"


def crop_resolution_9x16(display_size: list[int]) -> tuple[int, int]:
    w, h = display_size
    if w / h > 9 / 16:            # wider than 9:16 -> limited by height
        return int(h * 9 / 16) // 2 * 2, h
    return w, int(w * 16 / 9) // 2 * 2


def main(argv: list[str]) -> int:
    roots = [Path(a).resolve() for a in argv] or [REPO]
    media = find_media(roots)
    WORK.mkdir(exist_ok=True)
    items = {"video": [], "audio": [], "image": [str(p) for p in media["image"]]}
    for p in media["video"]:
        try:
            items["video"].append(describe(p))
        except Exception as e:  # keep going; report broken files
            items["video"].append({"path": str(p), "error": str(e)})
    for p in media["audio"]:
        try:
            items["audio"].append(describe(p))
        except Exception as e:
            items["audio"].append({"path": str(p), "error": str(e)})

    vids = [v for v in items["video"] if v.get("video")]
    rec: dict = {}
    if vids:
        fps_c = Counter(v["video"]["fps_nominal"] for v in vids)
        base = [f for f in fps_c if f <= 31]
        rec["fps_histogram"] = {str(k): n for k, n in fps_c.most_common()}
        if base:
            rec["timeline_fps"] = max(base, key=lambda f: fps_c[f])
        else:
            rec["timeline_fps"] = 30
        rec["high_fps_clips"] = [v["path"] for v in vids if v["video"]["fps_nominal"] >= 48]
        rec["hdr_clips"] = {v["path"]: v["video"]["color"]["transform"] for v in vids if v["video"]["color"]["hdr"]}
        rec["vfr_clips"] = [v["path"] for v in vids if v["video"]["vfr"]]
        crops = {v["path"]: crop_resolution_9x16(v["video"]["display_size"]) for v in vids}
        rec["native_9x16_crop"] = {k: f"{w}x{h}" for k, (w, h) in crops.items()}
        rec["supports_2160x3840_master"] = [k for k, (w, h) in crops.items() if w >= 2160]
        rec["needs_upscale_for_1080x1920"] = [k for k, (w, h) in crops.items() if w < 1080]
        notes = [n for v in vids for n in v["video"]["color"]["notes"]]
        rec["color_notes"] = sorted(set(notes))
    items["recommendations"] = rec

    (WORK / "inventory.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))

    lines = ["# Media inventory", "", "## Video", "",
             "| file | dur s | display | orient | fps | VFR | codec | bit | colour | HDR→SDR | audio |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in items["video"]:
        if "error" in v:
            lines.append(f"| {v['path']} | ERROR: {v['error'][:60]} |||||||||")
            continue
        vv = v["video"]
        c = vv["color"]
        a = v.get("audio") or {}
        lines.append(f"| {v['path']} | {v['duration']:.2f} | {vv['display_size'][0]}x{vv['display_size'][1]}"
                     f" (rot {vv['rotation']}) | {vv['orientation']} | {vv['fps']} | {'yes' if vv['vfr'] else ''}"
                     f" | {vv['codec']} {vv['profile'] or ''} | {c['bit_depth']} | {c['primaries']}/{c['transfer']}/"
                     f"{c['matrix']} {c['range']} | {c['transform']} | "
                     f"{(str(a.get('channels')) + 'ch ' + str(a.get('sample_rate'))) if a else '—'} |")
    lines += ["", "## Audio", "", "| file | dur s | codec | ch | rate | kbps |", "|---|---|---|---|---|---|"]
    for v in items["audio"]:
        if "error" in v:
            lines.append(f"| {v['path']} | ERROR |||||")
            continue
        a = v.get("audio") or {}
        lines.append(f"| {v['path']} | {v['duration']:.2f} | {a.get('codec')} | {a.get('channels')} | "
                     f"{a.get('sample_rate')} | {a.get('bitrate_kbps')} |")
    if items["image"]:
        lines += ["", "## Stills (not used in motion edit unless chosen)", ""] + [f"- {p}" for p in items["image"]]
    lines += ["", "## Recommendations", "", "```json", json.dumps(rec, indent=2, ensure_ascii=False), "```", ""]
    (WORK / "inventory.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if not items["video"] and not items["audio"]:
        print("\nNo footage or audio found. Put originals in footage/ and audio/ (see README).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
