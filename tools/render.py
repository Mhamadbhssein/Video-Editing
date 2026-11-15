#!/usr/bin/env python3
"""Step 4 — render the reel from the edit decision list (edit/edl.json).

    python3 tools/render.py [--edl edit/edl.json] [--only-shot ID] [--draft] [--skip-segments]

Pipeline (all parameters live in the EDL, so the edit is fully reproducible/editable):
  per shot:  coarse seek + exact trim -> [vid.stab] -> retime (frames | blend) -> 9:16 reframe (static crop
             or sub-pixel push/pan) -> spline36 resize -> [hqdn3d] -> [retime mci] -> [CAS sharpen]
             -> camera R'G'B' -> per-shot 3D LUT (HDR->SDR + balance + look; on >8-bit sources it runs before
             vid.stab/push/mci, which are 8-bit only) -> BT.709 Y'CbCr 4:2:2 10-bit -> ProRes 422 HQ segment
  assembly:  frame-exact cuts / centred dissolves (xfade) -> end fade -> Arabic titles (libass,
             HarfBuzz + FriBidi, fonts from assets/fonts) -> 4:2:0 -> H.264 High, CFR, BT.709 tags, +faststart
  audio:     music excerpt + optional ambience from the clips -> fades -> two-pass loudness
             (gain only, no limiter/compression: dynamics and pitch untouched) -> AAC-LC 48 kHz stereo

Timeline model: shot i starts (its cut point) at T_i = sum of previous `dur`.  A dissolve of length d
into shot i is centred on T_i; the renderer adds d/2 handles to both neighbouring segments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color as colormod  # noqa: E402
from ffutil import esc, fps_fraction, fps_rate  # noqa: E402,F401  (esc/fps_rate re-exported for qc.py)
from mediainfo import REPO, describe, from_rgb_filter, require, to_rgb_filter  # noqa: E402

WORK = REPO / "work"
RENDER = WORK / "render"
FONTS = REPO / "assets" / "fonts"
OUT = REPO / "output"

# Instagram Reels UI-safe region on a 1080x1920 canvas (conservative):
SAFE = {"left": 90, "right": 1080 - 170, "top": 250, "bottom": 1920 - 440}


# ------------------------------------------------------------------ helpers
def sh(cmd: list[str], log: Path | None = None) -> None:
    print("  $", " ".join(c if len(c) < 160 else c[:157] + "..." for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(p.stdout)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        raise SystemExit(f"command failed ({p.returncode})")


def abspath(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


# ------------------------------------------------------------------ timeline
def _frames(seconds, rate) -> int:
    """Seconds (as written in the EDL) -> nearest whole frame at the exact rate (round half up)."""
    from fractions import Fraction
    return math.floor(Fraction(str(seconds)) * rate + Fraction(1, 2))


def resolve(edl: dict) -> dict:
    """Frame-exact timeline.  Frame <-> time maths uses the exact rational rate (29.97 -> 30000/1001)."""
    from fractions import Fraction
    fps = float(edl["fps"])
    rate = fps_fraction(edl["fps"])
    shots = edl["shots"]
    clips: dict[str, dict] = {}
    T = 0
    out = []
    for i, s in enumerate(shots):
        f = s["file"]
        if f not in clips:
            clips[f] = describe(abspath(f))
        clip = clips[f]
        dur_frames = _frames(s["dur"], rate)
        tr = s.get("transition") or {"type": "cut"}
        d_in = 0
        if i > 0 and tr.get("type", "cut") in ("dissolve", "dip"):
            d_in = max(2, _frames(Fraction(str(tr.get("dur", 0.5))) / 2, rate) * 2)      # even frame count
        out.append({"index": i, "id": s.get("id", f"s{i + 1:02d}"), "shot": s, "clip": clip,
                    "T_frames": T, "dur_frames": dur_frames, "d_in": d_in})
        T += dur_frames
    for i, r in enumerate(out):
        d_out = out[i + 1]["d_in"] if i + 1 < len(out) else 0
        r["d_out"] = d_out
        r["seg_frames"] = r["dur_frames"] + r["d_in"] // 2 + d_out // 2
        speed = float(r["shot"].get("speed", 1.0))
        r["speed"] = speed
        sp = Fraction(str(speed))
        r["src_start"] = float(Fraction(str(r["shot"]["src_in"])) - Fraction(r["d_in"] // 2) / rate * sp)
        r["src_len"] = float(r["seg_frames"] / rate * sp)
        # source frames consumed per output frame (average rate for VFR clips)
        r["step"] = round((r["clip"]["video"]["fps"] or fps) * speed / fps, 6)
        cdur = r["clip"]["video"]["duration"] or r["clip"]["duration"]
        problems = []
        if r["src_start"] < 0:
            problems.append(f"needs {-r['src_start']:.2f}s more head handle (src_in too early for the dissolve)")
        if r["src_start"] + r["src_len"] > cdur - 1.0 / max(r["clip"]["video"]["fps"], 1):
            problems.append(f"runs {r['src_start'] + r['src_len'] - cdur:.2f}s past the end of the clip")
        r["problems"] = problems
    total_frames = out[-1]["T_frames"] + out[-1]["dur_frames"] if out else 0
    return {"fps": fps, "rate": fps_rate(edl["fps"]), "shots": out, "total_frames": total_frames,
            "total": float(total_frames / rate)}


def retime_mode(r: dict, fps: float) -> tuple[str, str]:
    """auto: an integer cadence, or any real-time (speed 1.0) shot, uses real frames only ('frames':
    the nearest source frame per output frame, so 24/25 fps or VFR clips on a 30 fps timeline repeat a
    frame now and then and 30 fps clips on 24/25 skip one, as an NLE conforms them; never synthetic
    in-betweens).  Other non-integer slow/fast-motion cadences blend the two nearest real frames
    (even motion; at 120 fps x0.8 this replaces a 3,3,3,3,4 judder by blends of frames 1/120 s apart).
    Motion interpolation only when the source has fewer frames than the slow motion needs."""
    src_fps = r["clip"]["video"]["fps"] or fps
    speed = r["speed"]
    mode = r["shot"].get("retime", "auto")
    step = src_fps * speed / fps          # source frames consumed per output frame
    note = f"src {src_fps:g}fps x{speed:g} -> step {step:.3f}"
    if mode != "auto":
        return mode, note
    if abs(step - round(step)) < 0.04 and round(step) >= 1:
        return "frames", note + " (exact frame cadence)"
    if abs(speed - 1.0) < 1e-6:
        return "frames", note + " (real time: conform with real frames, nearest frame per output frame)"
    if step >= 2.5:
        return "blend", note + " (non-integer high-fps cadence: blend of real frames 1/src_fps apart, no judder)"
    if step >= 0.98:
        return "blend", note + " (non-integer cadence: gentle blend of neighbouring real frames)"
    return "mci", note + " (WARNING: source fps too low for this speed — motion interpolation, inspect!)"


def reframe_filters(r: dict, out_w: int, out_h: int, pre_warp: list[str] | None = None,
                    out_fmt: str | None = None) -> list[str]:
    """9:16 window inside the display-oriented source.  frame: {cx, cy, zoom, end:{cx,cy,zoom}, ease}.

    A push/pan ('end') runs over the whole shot in OUTPUT frames: r['warp_step'] is the number of
    frames the warp sees per output frame (1 when it runs after the retime, the source step when it
    runs before it).  The source is first cropped to the union of the start and end windows, then
    `pre_warp` runs (the early grade: perspective is 8-bit only) and perspective warps the window.
    `out_fmt` sets the pixel format of the final resize (error-diffused when it is 8-bit)."""
    W, H = r["clip"]["video"]["display_size"]
    fr = r["shot"].get("frame") or {}
    aspect = out_w / out_h
    if W / H > aspect:
        bw, bh = H * aspect, H
    else:
        bw, bh = W, W / aspect
    resize = f"zscale=w={out_w}:h={out_h}:filter=spline36"
    if out_fmt:
        resize += (":dither=error_diffusion" if out_fmt in ("yuv444p", "yuv422p", "yuv420p") else "")
        resize += f",format={out_fmt}"

    def window(p):
        z = max(1.0, float(p.get("zoom", 1.0)))
        w, h = bw / z, bh / z
        cx = float(p.get("cx", 0.5)) * W
        cy = float(p.get("cy", 0.5)) * H
        x = min(max(cx - w / 2, 0), W - w)
        y = min(max(cy - h / 2, 0), H - h)
        return x, y, w, h

    x0, y0, w0, h0 = window(fr)
    end = fr.get("end")
    if not end:
        w, h = even(w0), even(h0)
        x, y = min(even(x0), W - w), min(even(y0), H - h)
        x, y = max(0, x), max(0, y)
        return [f"crop={w}:{h}:{x}:{y}", resize, "setsar=1"]
    x1, y1, w1, h1 = window({**fr, **end})
    # even-aligned union of both windows plus a margin for the cubic kernel
    ux = max(0, int(math.floor((min(x0, x1) - 4) / 2)) * 2)
    uy = max(0, int(math.floor((min(y0, y1) - 4) / 2)) * 2)
    uw = min(W - W % 2, int(math.ceil((max(x0 + w0, x1 + w1) + 4) / 2)) * 2) - ux
    uh = min(H - H % 2, int(math.ceil((max(y0 + h0, y1 + h1) + 4) / 2)) * 2) - uy
    # progress 0..1 from the first to the last output frame; perspective's `in` counts from 1
    n = max(1.0, (r["seg_frames"] - 1) * float(r.get("warp_step", 1.0)))
    t = f"min(1,(in-1)/{n:.6f})"
    ease = fr.get("ease", "inout")
    p = t
    if ease == "inout":
        p = f"(0.5-0.5*cos(PI*{t}))"
    elif ease == "out":
        p = f"(sin(PI/2*{t}))"

    def lerp(a, b):
        return f"({a:.4f}+({b - a:.4f})*{p})"
    # perspective (sense=source): the four source points (in the union crop) that become the output corners.
    X0, Y0 = lerp(x0 - ux, x1 - ux), lerp(y0 - uy, y1 - uy)
    X1, Y1 = lerp(x0 + w0 - ux, x1 + w1 - ux), lerp(y0 + h0 - uy, y1 + h1 - uy)
    persp = (f"perspective=x0='{X0}':y0='{Y0}':x1='{X1}':y1='{Y0}':x2='{X0}':y2='{Y1}':x3='{X1}':y3='{Y1}'"
             f":interpolation=cubic:sense=source:eval=frame")
    # perspective keeps the input size; resize the (stretched) result to the 9:16 output.
    return [f"crop={uw}:{uh}:{ux}:{uy}", *(pre_warp or []), persp, resize, "setsar=1"]


def shot_lut(r: dict, look: colormod.Look) -> Path:
    col = r["clip"]["video"]["color"]
    kind = col["transform"]
    if kind == "dovi5":
        raise SystemExit(f"{r['shot']['file']}: Dolby Vision profile 5 needs a DV-aware decoder; export HLG/SDR first.")
    kind = kind if kind in ("hlg", "pq") else "sdr709"
    if r["shot"].get("input_transform"):
        kind = r["shot"]["input_transform"]
    grade = colormod.shot_grade_from_dict(r["shot"].get("grade"))
    pq_peak = float(r["shot"].get("hdr_peak_nits") or col.get("peak_nits") or 1000)
    hlg_lw = float(r["shot"].get("hlg_display_nits") or 1000)
    key = json.dumps({"v": getattr(colormod, "LUT_VERSION", 1), "k": kind, "g": grade.__dict__, "l": look.__dict__,
                      "pq_peak": pq_peak, "hlg_lw": hlg_lw}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    path = RENDER / "luts" / f"{r['id']}_{h}.cube"
    if not path.exists():
        lut = colormod.build_lut(kind, grade, look, size=65, hdr_peak_nits=pq_peak, hlg_display_nits=hlg_lw)
        colormod.write_cube(path, lut, f"{r['id']} {kind}")
    return path


def render_segment(r: dict, edl: dict, out_w: int, out_h: int, look: colormod.Look, draft: bool) -> Path:
    fps = float(edl["fps"])
    rate = fps_rate(edl["fps"])           # exact rational (29.97 -> 30000/1001) for fps=/framerate=/-r
    s = r["shot"]
    v = r["clip"]["video"]
    src = abspath(s["file"])
    seg_dir = RENDER / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / f"{r['index']:02d}_{r['id']}.mov"
    log = RENDER / "logs" / f"{r['index']:02d}_{r['id']}.log"
    # In-point: coarse input seek ~2 s early, exact trim in the graph.  An accurate `-ss` on open-GOP
    # HEVC can land on a CRA whose leading (RASL) pictures are dropped, starting the shot up to 4
    # frames late; after a coarse seek those dropped pictures fall before the trim.  Time zero is the
    # real frame nearest the requested start (half a frame of tolerance, never a two-frame tie).
    src_fps = v["fps"] or fps
    start = max(0.0, r["src_start"])
    coarse = max(0.0, start - 2.0)
    pre = ["ffmpeg", "-v", "error", "-y", "-ss", f"{coarse:.6f}",
           "-t", f"{start - coarse + r['src_len'] + 0.6:.6f}", "-i", str(src)]
    head = [f"trim=start={max(0.0, start - coarse - 0.5 / src_fps):.6f}", "setpts=PTS-STARTPTS"]
    chain = list(head)

    mode, note = retime_mode(r, fps)
    r["retime"] = {"mode": mode, "note": note}
    if mode not in ("frames", "blend", "mci"):
        raise SystemExit(f"unknown retime mode {mode}")
    speed, step = r["speed"], r["step"]
    stab = s.get("stabilize")
    push = bool((s.get("frame") or {}).get("end"))
    # vidstabtransform, perspective and minterpolate are 8-bit only (FFmpeg silently converts to
    # yuv420p in front of them).  For >8-bit (HDR) sources the HDR->SDR LUT therefore runs before
    # the first of them, and the 8-bit step happens in the SDR output domain with error diffusion
    # (4:2:0 like the source, so the 8-bit filters cost what they did before).
    early = v["color"].get("bit_depth", 8) > 8 and (bool(stab) or push or mode == "mci")
    lut = shot_lut(r, look)
    grade = [to_rgb_filter(v["color"], "gbrpf32le"), f"lut3d=file={esc(lut)}:interp=tetrahedral"]
    grade8 = grade + [from_rgb_filter("yuv420p")]
    graded = False

    if stab:
        stab = stab if isinstance(stab, dict) else {}
        trf = RENDER / "stab" / f"{r['id']}.trf"
        trf.parent.mkdir(parents=True, exist_ok=True)
        sh(pre + ["-vf", ",".join(head + [f"vidstabdetect=shakiness={stab.get('shakiness', 6)}:accuracy=15:"
                                          f"stepsize=6:mincontrast=0.25:result={esc(trf)}"]),
                  "-an", "-f", "null", "-"])
        if early:
            chain += grade8
            graded = True
        chain.append(f"vidstabtransform=input={esc(trf)}:smoothing={stab.get('smoothing', 18)}:"
                     f"optzoom=0:zoom={stab.get('zoom', 4)}:interpol=bicubic:crop=black:relative=1")

    # frames/blend retime BEFORE the reframe, so a push counts output frames (and later filters only
    # see output frames); minterpolate is far too slow at source size, so it stays after the reframe.
    if mode == "frames":
        # fps=round=near keeps the LAST frame before slot+0.5; this lead makes it the NEAREST one
        lead = (0.5 - 0.5 / step) / fps if step > 0 else 0.0
        chain += [f"setpts=PTS/{speed:.6f}+({lead:.9f})/TB", f"fps={rate}:start_time=0:round=near"]
    elif mode == "blend":
        chain += [f"setpts=PTS/{speed:.6f}", f"framerate=fps={rate}:interp_start=0:interp_end=255:scene=100"]
    r["warp_step"] = step if mode == "mci" else 1.0
    pre_warp = grade8 if (early and push and not graded) else []
    graded = graded or bool(pre_warp)
    # once graded to 8 bits, resize back into 10 bits (or stay 8-bit, dithered, for minterpolate)
    chain += reframe_filters(r, out_w, out_h, pre_warp,
                             ("yuv420p" if mode == "mci" else "yuv422p10le") if graded else None)
    dn = float(s.get("denoise", 0) or 0)
    if dn > 0:
        chain.append(f"hqdn3d={1.2 * dn:.2f}:{1.0 * dn:.2f}:{3.0 * dn:.2f}:{2.5 * dn:.2f}")
    if mode == "mci":
        if early and not graded:
            chain += grade8
            graded = True
        chain += [f"setpts=PTS/{speed:.6f}",
                  f"minterpolate=fps={rate}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:vsbmc=1:scd=none"]

    shp = float(s.get("sharpen", 0) or 0)
    if shp > 0:
        chain.append(f"cas=strength={min(1.0, shp):.2f}")
    chain += ["format=yuv422p10le"] if graded else grade + [from_rgb_filter("yuv422p10le")]
    chain += [f"trim=end_frame={r['seg_frames']}", "setpts=N/FRAME_RATE/TB"]
    vf = ",".join(chain)
    codec = (["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-qscale:v", "4" if draft else "2"])
    sh(pre + ["-vf", vf, "-an", "-r", rate, "-fps_mode", "cfr", "-frames:v", str(r["seg_frames"])] + codec +
       ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        str(out)], log)
    # verify frame count
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                           stdout=subprocess.PIPE, text=True).stdout.strip() or 0)
    if n != r["seg_frames"]:
        raise SystemExit(f"segment {out.name}: {n} frames, expected {r['seg_frames']} (source too short?)")
    return out


# ------------------------------------------------------------------ titles
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""


def ass_color(rgb: str, alpha: int = 0) -> str:
    rgb = rgb.lstrip("#")
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(edl: dict, path: Path) -> Path:
    ty = edl.get("typography", {})
    styles = {
        # libass sizes by ascender+descender; Amiri reserves tall space for stacked marks, so the
        # nominal size is large: 340 -> «المدينة المنوّرة» ~490 px wide on the 1080 canvas.
        "title": {"font": "Amiri", "size": 340, "bold": 1, "color": "F7F2E8"},
        "line": {"font": "IBM Plex Sans Arabic Light", "size": 82, "bold": 0, "color": "F4EFE6"},
        "small": {"font": "IBM Plex Sans Arabic Light", "size": 60, "bold": 0, "color": "EDE6DA"},
    }
    for k, v in (ty.get("styles") or {}).items():
        styles.setdefault(k, {}).update(v)
    lines = [ASS_HEADER]
    for name, st in styles.items():
        # Encoding -1 = automatic base direction (RTL for Arabic): punctuation lands correctly.
        lines.append(f"Style: {name},{st['font']},{st['size']},{ass_color(st['color'])},&H000000FF,&H00000000,"
                     f"&H00000000,{st.get('bold', 0)},0,0,0,100,100,0,0,1,0,0,5,0,0,0,-1\n")
        lines.append(f"Style: {name}_glow,{st['font']},{st['size']},&H00000000,&H000000FF,&H00000000,"
                     f"&H00000000,{st.get('bold', 0)},0,0,0,100,100,0,0,1,0,0,5,0,0,0,-1\n")
    lines.append("\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    for t in edl.get("titles", []):
        text = t["text"]
        if re.search(r"\\fsp", text):
            raise SystemExit("letter-spacing (\\fsp) breaks Arabic joining — remove it")
        st = t.get("style", "line")
        x, y = t.get("x", 540), t.get("y", 760)
        fi, fo = int(t.get("fade_in", 0.8) * 1000), int(t.get("fade_out", 0.7) * 1000)
        drift = t.get("drift", 10)
        dur_ms = int((t["end"] - t["start"]) * 1000)
        move = f"\\move({x},{y + drift},{x},{y - drift * 0.3:.1f},0,{dur_ms})" if drift else f"\\pos({x},{y})"
        focus = f"\\blur{t.get('blur_from', 3)}\\t(0,{int(fi * 1.2)},\\blur{t.get('blur', 0.6)})"
        glow_alpha = t.get("shadow_alpha", 0x70)
        glow = (f"{{{move}\\fad({fi},{fo})\\blur{t.get('shadow_blur', 14)}\\bord{t.get('shadow_bord', 6)}"
                f"\\3c&H000000&\\alpha&H{glow_alpha:02X}&\\1a&HFF&}}{text}")
        main = f"{{{move}\\fad({fi},{fo}){focus}}}{text}"
        a, b = ass_time(t["start"]), ass_time(t["end"])
        lines.append(f"Dialogue: 0,{a},{b},{st}_glow,,0,0,0,,{glow}\n")
        lines.append(f"Dialogue: 1,{a},{b},{st},,0,0,0,,{main}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------ audio
def parse_ebur128(stderr: str) -> dict:
    txt = stderr[stderr.rfind("Summary:"):]
    def g(key):
        m = re.search(key + r":\s+(-?[\d.]+|-inf)", txt)
        return float(m.group(1)) if m and m.group(1) != "-inf" else None
    return {"I": g(r"I"), "LRA": g(r"LRA"), "TP": g(r"Peak")}


def measure(path: Path) -> dict:
    p = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], stderr=subprocess.PIPE, text=True)
    return parse_ebur128(p.stderr)


def render_audio(edl: dict, total: float) -> Path:
    a = edl.get("music")
    amb = edl.get("ambience", [])
    raw = RENDER / "mix_raw.wav"
    final = RENDER / "mix.wav"
    inputs, labels, fc = [], [], []
    k = 0
    if a:
        m_in = float(a.get("in", 0.0))
        fo = float(a.get("fade_out", 2.0))
        fi = float(a.get("fade_in", 0.0))
        inputs += ["-ss", f"{m_in:.4f}", "-t", f"{total + 0.1:.4f}", "-i", str(abspath(a["file"]))]
        chain = [f"aresample=48000", "aformat=sample_fmts=fltp:channel_layouts=stereo",
                 f"volume={a.get('gain_db', 0)}dB"]
        if fi > 0:
            chain.append(f"afade=t=in:st=0:d={fi}:curve=qsin")
        chain.append(f"afade=t=out:st={max(0.0, total - fo):.4f}:d={fo}:curve={a.get('fade_curve', 'qsin')}")
        fc.append(f"[{k}:a]{','.join(chain)}[m{k}]")
        labels.append(f"[m{k}]")
        k += 1
    for amb_i in amb:
        d = float(amb_i["dur"])
        inputs += ["-ss", f"{float(amb_i.get('src_in', 0)):.4f}", "-t", f"{d:.4f}", "-i", str(abspath(amb_i["file"]))]
        chain = ["aresample=48000", "aformat=sample_fmts=fltp:channel_layouts=stereo",
                 f"highpass=f={amb_i.get('highpass', 90)}", f"lowpass=f={amb_i.get('lowpass', 14000)}",
                 f"volume={amb_i.get('gain_db', -18)}dB",
                 f"afade=t=in:st=0:d={amb_i.get('fade_in', 0.8)}:curve=qsin",
                 f"afade=t=out:st={max(0.0, d - amb_i.get('fade_out', 1.2)):.4f}:d={amb_i.get('fade_out', 1.2)}:curve=qsin",
                 f"adelay={int(float(amb_i.get('at', 0)) * 1000)}:all=1"]
        fc.append(f"[{k}:a]{','.join(chain)}[m{k}]")
        labels.append(f"[m{k}]")
        k += 1
    if not labels:
        sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo", "-t", f"{total:.4f}",
            "-c:a", "pcm_s24le", str(final)])
        return final
    fc.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,"
              f"apad=whole_dur={total:.4f},atrim=0:{total:.4f}[mix]")
    sh(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex", ";".join(fc), "-map", "[mix]",
                                                   "-c:a", "pcm_f32le", str(raw)])
    m = measure(raw)
    target = float(edl.get("loudness", {}).get("target_lufs", -14.0))
    ceiling = float(edl.get("loudness", {}).get("true_peak_ceiling", -1.5))
    gain = target - (m["I"] if m["I"] is not None else target)
    if m["TP"] is not None and m["TP"] + gain > ceiling:
        gain = ceiling - m["TP"]
    sh(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", f"volume={gain:.3f}dB", "-c:a", "pcm_s24le",
        str(final)])
    after = measure(final)
    (RENDER / "loudness.json").write_text(json.dumps({"before": m, "gain_db": round(gain, 2), "after": after,
                                                       "target": target, "ceiling": ceiling}, indent=1))
    print(f"  audio: {m} -> gain {gain:+.2f} dB -> {after}")
    return final


# ------------------------------------------------------------------ assembly + export
# 'dip': through limited-range black (segments meet xfade as 10-bit 4:4:4: Y'=64, Cb=Cr=512), centred
# on the cut.  xfade's progress P runs 1 -> 0 and is exactly 0.5 on the cut frame T_i: the outgoing
# shot eases down to black over the first half (smoothstep of 2P-1), the incoming shot eases up from
# black over the second half (smoothstep of 1-2P).  Only frame T_i is pure black; never below black.
_K = "if(eq(PLANE,0),64,512)"
DIP_EXPR = (f"if(gt(P,0.5),{_K}+(A-{_K})*(2*P-1)*(2*P-1)*(5-4*P),"
            f"{_K}+(B-{_K})*(1-2*P)*(1-2*P)*(1+4*P))")
AAC_DELAY = 1024          # FFmpeg's native AAC encoder priming, in samples


def h264_level(w: int, h: int, fps: float) -> str:
    """Lowest H.264 level (from 4.1 up) whose frame size and macroblock rate fit."""
    mbs = math.ceil(w / 16) * math.ceil(h / 16)
    for level, max_mbps, max_fs in (("4.1", 245760, 8192), ("4.2", 522240, 8704), ("5.1", 983040, 36864),
                                    ("5.2", 2073600, 36864), ("6.1", 8355840, 139264), ("6.2", 16711680, 139264)):
        if mbs <= max_fs and mbs * fps <= max_mbps:
            return level
    raise SystemExit(f"{w}x{h} @ {fps:g} fps exceeds H.264 level 6.2")


def assemble(edl: dict, tl: dict, segs: list[Path], ass: Path, audio: Path, out_w: int, out_h: int,
             deliver_w: int, deliver_h: int, out_path: Path, draft: bool) -> Path:
    fps = tl["fps"]
    rate = fps_rate(edl["fps"])
    fr = fps_fraction(edl["fps"])
    # Exact integer frame clock (time base = one frame, pts = frame index) on every stream that meets
    # concat/xfade: a microsecond time base rounds k/fps, so xfade's `first_pts + offset <= pts` test
    # could start a dissolve one frame late after earlier cuts.
    clock = f"settb={fr.denominator}/{fr.numerator},setpts=N"
    inputs = []
    for p in segs:
        inputs += ["-i", str(p)]
    fc = [f"[{i}:v]setsar=1,{clock}[s{i}]" for i in range(len(segs))]
    acc, acc_frames = "[s0]", tl["shots"][0]["seg_frames"]
    for i in range(1, len(segs)):
        r = tl["shots"][i]
        tr = r["shot"].get("transition") or {"type": "cut"}
        lab = f"[v{i}]"
        if r["d_in"] == 0:
            fc.append(f"{acc}[s{i}]concat=n=2:v=1:a=0,{clock}{lab}")
            acc_frames += r["seg_frames"]
        else:
            xf = f"xfade=transition=custom:expr='{DIP_EXPR}'" if tr.get("type") == "dip" else "xfade=transition=fade"
            off = float((acc_frames - r["d_in"]) / fr)
            dur = float(r["d_in"] / fr)
            fc.append(f"{acc}format=yuv444p10le[a{i}];[s{i}]format=yuv444p10le[b{i}];"
                      f"[a{i}][b{i}]{xf}:duration={dur:.6f}:offset={off:.6f},{clock}{lab}")
            acc_frames += r["seg_frames"] - r["d_in"]
        acc = lab
    if acc_frames != tl["total_frames"]:
        raise SystemExit(f"assembly length {acc_frames} != timeline {tl['total_frames']} frames")
    post = []
    end = edl.get("end") or {}
    if end.get("fade_to_black"):
        d = float(end["fade_to_black"])
        post.append(f"fade=t=out:st={tl['total'] - d:.5f}:d={d:.5f}")
    if (deliver_w, deliver_h) != (out_w, out_h):
        post.append(f"zscale=w={deliver_w}:h={deliver_h}:filter=spline36")
    post.append(f"ass={esc(ass)}:fontsdir={esc(FONTS)}")
    post.append("zscale=dither=error_diffusion,format=yuv420p")
    post.append("setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")
    fc.append(f"{acc}{','.join(post)}[vout]")
    a_idx = len(segs)
    # No edit lists in the MP4 (Meta's Reels spec): -use_editlist 0 plus negative CTS offsets keeps the
    # first video frame at t=0 despite B-frames; without an edit list every player also plays the AAC
    # priming, so the mix is advanced by exactly the encoder delay to stay sample-aligned with the picture.
    fc.append(f"[{a_idx}:a]aresample=48000,atrim=start_sample={AAC_DELAY},asetpts=PTS-STARTPTS[aout]")
    level = h264_level(deliver_w, deliver_h, fps)
    crf = "22" if draft else str(edl.get("export", {}).get("crf", 16))
    maxrate = "60M" if deliver_w > 1080 else str(edl.get("export", {}).get("maxrate", "25M"))
    gop = int(round(fps * 2))
    cmd = (["ffmpeg", "-v", "error", "-y"] + inputs + ["-i", str(audio), "-filter_complex", ";".join(fc),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium" if draft else "slow", "-crf", crf,
            "-profile:v", "high", "-level:v", level, "-pix_fmt", "yuv420p",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off:aq-mode=3",
            "-g", str(gop), "-keyint_min", str(int(fps)), "-bf", "3", "-maxrate", maxrate, "-bufsize",
            str(int(maxrate[:-1]) * 2) + "M",
            "-r", rate, "-fps_mode", "cfr",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
            "-t", f"{tl['total']:.6f}", "-use_editlist", "0", "-movflags", "+faststart+negative_cts_offsets",
            "-metadata", f"title={edl.get('title', 'المدينة المنوّرة')}", str(out_path)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sh(cmd, RENDER / "logs" / f"assemble_{deliver_w}x{deliver_h}.log")
    return out_path


def main() -> int:
    require("ffmpeg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--edl", default=str(REPO / "edit" / "edl.json"))
    ap.add_argument("--only-shot", default=None, help="render just one segment (for look/crop checks)")
    ap.add_argument("--skip-segments", action="store_true", help="reuse existing segments")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    edl = json.loads(Path(args.edl).read_text())
    tl = resolve(edl)
    bad = [(r["id"], p) for r in tl["shots"] for p in r["problems"]]
    for sid, p in bad:
        print(f"  ! {sid}: {p}")
    if bad:
        raise SystemExit("fix the EDL handles above")
    out_w, out_h = edl.get("size", [1080, 1920])
    master = edl.get("master_size")
    seg_w, seg_h = (master if master else (out_w, out_h))
    look = colormod.look_from_dict(edl.get("look"))
    RENDER.mkdir(parents=True, exist_ok=True)
    segs = []
    for r in tl["shots"]:
        if args.only_shot and r["id"] != args.only_shot:
            continue
        seg = RENDER / "segments" / f"{r['index']:02d}_{r['id']}.mov"
        if args.skip_segments and seg.exists():
            mode, note = retime_mode(r, tl["fps"])
            r["retime"] = {"mode": mode, "note": note}
            segs.append(seg)
            continue
        print(f"[{r['index'] + 1}/{len(tl['shots'])}] {r['id']}  {r['shot']['file']} @ {r['shot']['src_in']}s "
              f"dur {r['dur_frames']}f x{r['speed']}")
        segs.append(render_segment(r, edl, seg_w, seg_h, look, args.draft))
        print(f"      retime: {r['retime']['note']} -> {r['retime']['mode']}")
    if args.only_shot:
        return 0
    timeline = {"fps": tl["fps"], "total": tl["total"], "total_frames": tl["total_frames"], "shots": [
        {k: v for k, v in r.items() if k not in ("clip",)} | {"T": r["T_frames"] / tl["fps"],
                                                              "file": r["shot"]["file"]}
        for r in tl["shots"]]}
    (RENDER / "timeline.json").write_text(json.dumps(timeline, indent=1, ensure_ascii=False, default=str))
    ass = build_ass(edl, RENDER / "titles.ass")
    audio = render_audio(edl, tl["total"])
    name = args.out or edl.get("export", {}).get("name", "Madinah_Final_Reel.mp4")
    final = assemble(edl, tl, segs, ass, audio, seg_w, seg_h, out_w, out_h, OUT / name, args.draft)
    print(f"wrote {final.relative_to(REPO)}")
    if master and not args.draft:
        mname = edl.get("export", {}).get("master_name", "Madinah_Master_2160x3840.mp4")
        m = assemble(edl, tl, segs, ass, audio, seg_w, seg_h, master[0], master[1], OUT / mname, False)
        print(f"wrote {m.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
