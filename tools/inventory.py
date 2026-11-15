#!/usr/bin/env python3
"""Step 1 — find and describe every piece of footage and audio (read-only).

    python3 tools/inventory.py [ROOT ...]

Writes work/inventory.json and work/inventory.md and prints a summary with
recommendations (timeline frame rate, HDR handling, whether a 4K master is justified).
Files are classified by their streams, not only their extension: an audio-only .mp4/.mov
(e.g. a nasheed delivered that way) is listed under audio.  Unreadable files (including Git
LFS pointers left by a clone without `git lfs pull`) are reported with the reason; the exit
code is 2 when nothing readable was found.  Original files are only ever read.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediainfo import REPO, describe, find_media, is_lfs_pointer  # noqa: E402

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

    def broken(p: Path, err: str) -> dict:
        return {"path": str(p), "error": err, "lfs_pointer": is_lfs_pointer(p)}

    for p in media["video"]:
        try:
            d = describe(p)
        except Exception as e:  # keep going; report broken files
            items["video"].append(broken(p, str(e)))
            continue
        if d.get("video"):
            items["video"].append(d)
        elif d.get("audio"):    # e.g. a nasheed/recitation delivered as .mp4/.mov: it is a music track
            d["note"] = "audio-only file in a video container"
            items["audio"].append(d)
        else:
            items["video"].append(broken(p, "no video or audio stream"))
    for p in media["audio"]:
        try:
            d = describe(p)
        except Exception as e:
            items["audio"].append(broken(p, str(e)))
            continue
        items["audio"].append(d if d.get("audio") else broken(p, "no audio stream"))

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

    def cell(t: str) -> str:
        return " / ".join(str(t).split("\n")).replace("|", "\\|")

    for v in items["video"]:
        if "error" in v:
            lines.append(f"| {v['path']} | ERROR: {cell(v['error'])} |||||||||")
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
            lines.append(f"| {v['path']} | ERROR: {cell(v['error'])} |||||")
            continue
        a = v.get("audio") or {}
        note = f" ({v['note']})" if v.get("note") else ""
        lines.append(f"| {v['path']}{note} | {v['duration']:.2f} | {a.get('codec')} | {a.get('channels')} | "
                     f"{a.get('sample_rate')} | {a.get('bitrate_kbps')} |")
    if items["image"]:
        lines += ["", "## Stills (not used in motion edit unless chosen)", ""] + [f"- {p}" for p in items["image"]]
    lines += ["", "## Recommendations", "", "```json", json.dumps(rec, indent=2, ensure_ascii=False), "```", ""]
    (WORK / "inventory.md").write_text("\n".join(lines))
    print("\n".join(lines))
    errors = [v for v in items["video"] + items["audio"] if "error" in v]
    if errors:
        print(f"\nWARNING: {len(errors)} file(s) could not be read:", file=sys.stderr)
        for v in errors:
            print(f"  {v['path']}: {v['error']}", file=sys.stderr)
        n_lfs = sum(1 for v in errors if v.get("lfs_pointer"))
        if n_lfs:
            print(f"  -> {n_lfs} of them are Git LFS pointers (the clone has no media objects): "
                  "run `git lfs install && git lfs pull`, then re-run this step.", file=sys.stderr)
    if not any("error" not in v for v in items["video"] + items["audio"]):
        print("\nNo readable footage or audio found. Put originals in footage/ and audio/ (see README).",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
