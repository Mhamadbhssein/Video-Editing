#!/usr/bin/env python3
"""Step 4 — render the reel from the edit decision list (edit/edl.json).

    python3 tools/render.py [--edl edit/edl.json] [--only-shot ID] [--draft] [--skip-segments]

Pipeline (all parameters live in the EDL, so the edit is fully reproducible/editable):
  per shot:  accurate seek -> [vid.stab] -> 9:16 reframe (static crop or sub-pixel push/pan)
             -> spline36 resize -> [hqdn3d] -> retime (frames | blend | mci) -> [CAS sharpen]
             -> camera R'G'B' -> per-shot 3D LUT (HDR->SDR + balance + look) -> BT.709 Y'CbCr 4:2:2 10-bit
             -> ProRes 422 HQ segment (work/render/segments/)
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
def resolve(edl: dict) -> dict:
    fps = float(edl["fps"])
    shots = edl["shots"]
    clips: dict[str, dict] = {}
    t = 0.0
    out = []
    for i, s in enumerate(shots):
        f = s["file"]
        if f not in clips:
            clips[f] = describe(abspath(f))
        clip = clips[f]
        dur_frames = int(round(s["dur"] * fps))
        tr = s.get("transition") or {"type": "cut"}
        d_in = 0
        if i > 0 and tr.get("type", "cut") in ("dissolve", "dip"):
            d_in = max(2, int(round(tr.get("dur", 0.5) * fps / 2.0)) * 2)      # even frame count
        out.append({"index": i, "id": s.get("id", f"s{i + 1:02d}"), "shot": s, "clip": clip,
                    "T_frames": int(round(t * fps)), "dur_frames": dur_frames, "d_in": d_in})
        t = (out[-1]["T_frames"] + dur_frames) / fps
    for i, r in enumerate(out):
        d_out = out[i + 1]["d_in"] if i + 1 < len(out) else 0
        r["d_out"] = d_out
        r["seg_frames"] = r["dur_frames"] + r["d_in"] // 2 + d_out // 2
        speed = float(r["shot"].get("speed", 1.0))
        r["speed"] = speed
        r["src_start"] = float(r["shot"]["src_in"]) - (r["d_in"] // 2) / fps * speed
        r["src_len"] = r["seg_frames"] / fps * speed
        cdur = r["clip"]["video"]["duration"] or r["clip"]["duration"]
        problems = []
        if r["src_start"] < 0:
            problems.append(f"needs {-r['src_start']:.2f}s more head handle (src_in too early for the dissolve)")
        if r["src_start"] + r["src_len"] > cdur - 1.0 / max(r["clip"]["video"]["fps"], 1):
            problems.append(f"runs {r['src_start'] + r['src_len'] - cdur:.2f}s past the end of the clip")
        r["problems"] = problems
    total_frames = out[-1]["T_frames"] + out[-1]["dur_frames"] if out else 0
    return {"fps": fps, "shots": out, "total_frames": total_frames, "total": total_frames / fps}


def retime_mode(r: dict, fps: float) -> tuple[str, str]:
    src_fps = r["clip"]["video"]["fps"] or fps
    speed = r["speed"]
    mode = r["shot"].get("retime", "auto")
    step = src_fps * speed / fps          # source frames consumed per output frame
    note = f"src {src_fps:g}fps x{speed:g} -> step {step:.3f}"
    if mode != "auto":
        return mode, note
    if abs(step - round(step)) < 0.04 and round(step) >= 1:
        return "frames", note + " (exact frame cadence)"
    if step >= 2.5:
        return "frames", note + " (high-fps source, sub-frame jitter negligible)"
    if step >= 0.98:
        return "blend", note + " (non-integer cadence: gentle blend of neighbouring real frames)"
    return "mci", note + " (WARNING: source fps too low for this speed — motion interpolation, inspect!)"


def reframe_filters(r: dict, out_w: int, out_h: int) -> list[str]:
    """9:16 window inside the display-oriented source.  frame: {cx, cy, zoom, end:{cx,cy,zoom}, ease}."""
    W, H = r["clip"]["video"]["display_size"]
    fr = r["shot"].get("frame") or {}
    aspect = out_w / out_h
    if W / H > aspect:
        bw, bh = H * aspect, H
    else:
        bw, bh = W, W / aspect

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
        return [f"crop={w}:{h}:{x}:{y}",
                f"zscale=w={out_w}:h={out_h}:filter=spline36", "setsar=1"]
    x1, y1, w1, h1 = window({**fr, **end})
    n = max(1, r["seg_frames"] - 1)
    ease = fr.get("ease", "inout")
    # perspective exposes the input frame counter as `in`
    p = f"min(1,in/{n})"
    if ease == "inout":
        p = f"(0.5-0.5*cos(PI*min(1,in/{n})))"
    elif ease == "out":
        p = f"(sin(PI/2*min(1,in/{n})))"

    def lerp(a, b):
        return f"({a:.4f}+({b - a:.4f})*{p})"
    # perspective (sense=source): the four source points that become the output corners.
    X0, Y0 = lerp(x0, x1), lerp(y0, y1)
    X1, Y1 = lerp(x0 + w0, x1 + w1), lerp(y0 + h0, y1 + h1)
    persp = (f"perspective=x0='{X0}':y0='{Y0}':x1='{X1}':y1='{Y0}':x2='{X0}':y2='{Y1}':x3='{X1}':y3='{Y1}'"
             f":interpolation=cubic:sense=source:eval=frame")
    # perspective keeps the input size; resize the (stretched) result to the 9:16 output.
    return [persp, f"zscale=w={out_w}:h={out_h}:filter=spline36", "setsar=1"]


def shot_lut(r: dict, look: colormod.Look) -> Path:
    col = r["clip"]["video"]["color"]
    kind = col["transform"]
    if kind == "dovi5":
        raise SystemExit(f"{r['shot']['file']}: Dolby Vision profile 5 needs a DV-aware decoder; export HLG/SDR first.")
    kind = kind if kind in ("hlg", "pq") else "sdr709"
    if r["shot"].get("input_transform"):
        kind = r["shot"]["input_transform"]
    grade = colormod.shot_grade_from_dict(r["shot"].get("grade"))
    key = json.dumps({"k": kind, "g": grade.__dict__, "l": look.__dict__,
                      "peak": r["shot"].get("hdr_peak_nits", 1000)}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    path = RENDER / "luts" / f"{r['id']}_{h}.cube"
    if not path.exists():
        lut = colormod.build_lut(kind, grade, look, size=65, hdr_peak_nits=r["shot"].get("hdr_peak_nits", 1000))
        colormod.write_cube(path, lut, f"{r['id']} {kind}")
    return path


def render_segment(r: dict, edl: dict, out_w: int, out_h: int, look: colormod.Look, draft: bool) -> Path:
    fps = float(edl["fps"])
    s = r["shot"]
    src = abspath(s["file"])
    seg_dir = RENDER / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / f"{r['index']:02d}_{r['id']}.mov"
    log = RENDER / "logs" / f"{r['index']:02d}_{r['id']}.log"
    pre = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, r['src_start']):.5f}",
           "-t", f"{r['src_len'] + 0.6:.5f}", "-i", str(src)]
    chain = ["setpts=PTS-STARTPTS"]

    stab = s.get("stabilize")
    if stab:
        stab = stab if isinstance(stab, dict) else {}
        trf = RENDER / "stab" / f"{r['id']}.trf"
        trf.parent.mkdir(parents=True, exist_ok=True)
        sh(pre + ["-vf", f"setpts=PTS-STARTPTS,vidstabdetect=shakiness={stab.get('shakiness', 6)}:accuracy=15:"
                         f"stepsize=6:mincontrast=0.25:result={esc(trf)}", "-an", "-f", "null", "-"])
        chain.append(f"vidstabtransform=input={esc(trf)}:smoothing={stab.get('smoothing', 18)}:"
                     f"optzoom=0:zoom={stab.get('zoom', 4)}:interpol=bicubic:crop=black:relative=1")

    chain += reframe_filters(r, out_w, out_h)
    dn = float(s.get("denoise", 0) or 0)
    if dn > 0:
        chain.append(f"hqdn3d={1.2 * dn:.2f}:{1.0 * dn:.2f}:{3.0 * dn:.2f}:{2.5 * dn:.2f}")

    mode, note = retime_mode(r, fps)
    r["retime"] = {"mode": mode, "note": note}
    speed = r["speed"]
    if mode == "frames":
        chain += [f"setpts=(PTS-STARTPTS)/{speed:.6f}", f"fps={fps:g}:start_time=0:round=near"]
    elif mode == "blend":
        chain += [f"setpts=(PTS-STARTPTS)/{speed:.6f}", f"framerate=fps={fps:g}:interp_start=0:interp_end=255:scene=100"]
    elif mode == "mci":
        chain += [f"setpts=(PTS-STARTPTS)/{speed:.6f}",
                  f"minterpolate=fps={fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:vsbmc=1:scd=none"]
    else:
        raise SystemExit(f"unknown retime mode {mode}")

    shp = float(s.get("sharpen", 0) or 0)
    if shp > 0:
        chain.append(f"cas=strength={min(1.0, shp):.2f}")
    lut = shot_lut(r, look)
    chain += [to_rgb_filter(r["clip"]["video"]["color"], "gbrpf32le"),
              f"lut3d=file={esc(lut)}:interp=tetrahedral",
              from_rgb_filter("yuv422p10le"),
              f"trim=end_frame={r['seg_frames']}", "setpts=N/FRAME_RATE/TB"]
    vf = ",".join(chain)
    codec = (["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-qscale:v", "4" if draft else "2"])
    sh(pre + ["-vf", vf, "-an", "-r", f"{fps:g}", "-fps_mode", "cfr", "-frames:v", str(r["seg_frames"])] + codec +
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
def assemble(edl: dict, tl: dict, segs: list[Path], ass: Path, audio: Path, out_w: int, out_h: int,
             deliver_w: int, deliver_h: int, out_path: Path, draft: bool) -> Path:
    fps = tl["fps"]
    inputs = []
    for p in segs:
        inputs += ["-i", str(p)]
    fc = [f"[{i}:v]setsar=1,settb=AVTB[s{i}]" for i in range(len(segs))]
    acc, acc_frames = "[s0]", tl["shots"][0]["seg_frames"]
    for i in range(1, len(segs)):
        r = tl["shots"][i]
        tr = r["shot"].get("transition") or {"type": "cut"}
        lab = f"[v{i}]"
        if r["d_in"] == 0:
            fc.append(f"{acc}[s{i}]concat=n=2:v=1:a=0,settb=AVTB{lab}")
            acc_frames += r["seg_frames"]
        else:
            kind = "fadeblack" if tr.get("type") == "dip" else "fade"
            off = (acc_frames - r["d_in"]) / fps
            fc.append(f"{acc}[s{i}]xfade=transition={kind}:duration={r['d_in'] / fps:.5f}:offset={off:.5f},settb=AVTB{lab}")
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
    level = "5.1" if deliver_w * deliver_h > 1920 * 1088 else ("4.2" if fps > 30.5 else "4.1")
    crf = "22" if draft else str(edl.get("export", {}).get("crf", 16))
    maxrate = "60M" if deliver_w > 1080 else str(edl.get("export", {}).get("maxrate", "25M"))
    gop = int(round(fps * 2))
    cmd = (["ffmpeg", "-v", "error", "-y"] + inputs + ["-i", str(audio), "-filter_complex", ";".join(fc),
            "-map", "[vout]", "-map", f"{a_idx}:a",
            "-c:v", "libx264", "-preset", "medium" if draft else "slow", "-crf", crf,
            "-profile:v", "high", "-level:v", level, "-pix_fmt", "yuv420p",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off:aq-mode=3",
            "-g", str(gop), "-keyint_min", str(int(fps)), "-bf", "3", "-maxrate", maxrate, "-bufsize",
            str(int(maxrate[:-1]) * 2) + "M",
            "-r", f"{fps:g}", "-fps_mode", "cfr",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
            "-t", f"{tl['total']:.5f}", "-movflags", "+faststart",
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
