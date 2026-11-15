# Video-Editing — «المدينة المنوّرة» Instagram Reel

A reproducible, script-driven edit: every creative decision (shots, in-points, speed, framing,
grade, transitions, Arabic titles, music excerpt, mix) lives in `edit/edl.json`, and the tools
render it to a finished reel with FFmpeg. Original files are only ever read.

## Status

**Waiting for the original footage and audio.** The pipeline is built and tested end-to-end on
synthetic test signals; the creative edit starts as soon as the material is in `footage/` and `audio/`.

## Hand-off: how to get the material here

Pick one:

1. **Push it to this branch with Git LFS** (`.gitattributes` already routes `footage/` and `audio/` media through LFS):
   ```bash
   git lfs install
   git fetch origin claude/madinah-instagram-reel-vzfdnh && git checkout claude/madinah-instagram-reel-vzfdnh
   cp /path/to/your/clips/*  footage/      # originals straight from the phone (.MOV/.MP4)
   cp /path/to/your/audio/*  audio/        # the track(s) to use
   git add footage audio && git commit -m "Add original footage and audio" && git push
   ```
   Export the originals with AirDrop / Image Capture / "Export Unmodified Original" so HDR,
   frame rate and metadata are kept (a WhatsApp/Telegram copy is recompressed and loses them).
2. **Run the session on your own computer** (Claude desktop app or `claude remote-control` in the
   folder that holds the clips) — no upload needed; requires FFmpeg with libass + libzimg.

## Layout

| path | what |
|---|---|
| `footage/`, `audio/` | your originals (read-only for the tools) |
| `assets/fonts/` | Amiri, IBM Plex Sans Arabic, Noto Naskh Arabic (variable font: libass 0.17 uses only its Regular instance) (+ OFL licences) |
| `tools/` | the pipeline (Python 3 + numpy + Pillow + FFmpeg; `ffutil.py` = shared filtergraph escaping + exact NTSC rates) |
| `edit/edl.json` | **the edit** (template: `edit/edl.template.json`), Arabic line candidates in `edit/lines_bank.md` |
| `output/` | `Madinah_Final_Reel.mp4` (+ `Madinah_Master_2160x3840.mp4` when the footage supports it) |
| `work/` | regenerated scratch: contact sheets, filmstrips, proxies, audio pages, segments, QC sheets |

## Pipeline

```bash
tools/run_all.sh study     # 1-3: inventory, footage review, audio analysis
# write edit/edl.json from what the review shows
tools/run_all.sh render    # 4-5: render + QC
```

1. `inventory.py` — resolution, rotation, orientation, fps / VFR (edit-list aware), codec, bit depth,
   primaries/transfer/matrix, HLG / PQ (+ MaxCLL / mastering peak) / Dolby Vision detection, audio streams
   → `work/inventory.md`. Git LFS pointer files are reported as such (`git lfs pull`), audio-only files in a
   video container are treated as audio, and the step stops if no readable media is found.
2. `review.py` — per clip (`work/review/<stem>_<hash>/`): contact sheet, dense 4 fps filmstrips, 540 px SDR
   proxy, metrics (exposure, clipped highlights, sharpness, camera speed, jitter) and suggested strongest
   moments. Times are real source timestamps on the `src_in` axis (VFR-safe) and frames are picked exactly
   as the renderer picks them. It exits 1 if a clip fails to decode or comes out all black, and flags
   Dolby Vision profile 5 clips, whose preview colours are wrong. Needs FFmpeg ≥ 6.1.
3. `audio_analysis.py` — waveform / spectrogram pages, EBU R128 loudness (levels in true dBFS), tempo + beat
   grid only when the beat is reliable (beatless recitation, nasheed or pads report "no reliable beat"),
   pauses (breaths don't split them) and phrase spans (never cut inside a recited or sung phrase), energy
   changes, ranked excerpts (30–45 s, or `--len SECONDS`) that start just before a phrase onset and end at a
   real pause, on bar lines when there is a beat.
4. `render.py` — preflight first (fonts actually used by libass, music length and phrase end, every ambience
   source), then per shot: frame-accurate in-point (coarse seek + trim, safe on open-GOP HEVC) → optional
   vid.stab → retime (real frames at normal speed; real frames or an even-cadence blend for 0.7–0.85× slow
   motion; motion interpolation only on request) → 9:16 reframe (static crop or a sub-pixel push timed in
   output frames) → spline36 resize → light denoise → optional sharpening → **one 3D LUT per shot** (HDR→SDR BT.2100/BT.2408
   conversion, exposure/white balance and the film's look graded on the full HDR range, then one
   hue-preserving highlight roll-off; on 10-bit sources it runs before any 8-bit-only filter, so no banding)
   → ProRes 422 HQ segment. Then frame-exact cuts, centred dissolves and dips through black on an exact frame
   clock, Arabic titles drawn in RGB (libass + HarfBuzz + FriBidi, automatic RTL, soft halo for legibility),
   H.264 High (level chosen from size × fps) / yuv420p / BT.709 / CFR / +faststart / no edit lists,
   AAC-LC 48 kHz, 320 kb/s target.
   Audio: sample-accurate music excerpt + optional clip ambience, fades, two-pass loudness to −14 LUFS with a
   −1.5 dBTP ceiling using gain only (no limiter, no compression, no pitch change). If the ceiling stops the
   gain short, the shortfall is reported; an oversampled limiter is available only as an opt-in
   (`loudness.max_limiter_db`).
5. `qc.py` — delivery spec, black/frozen/duplicated frames, unexpected cuts, loudness/true peak,
   silences, title boxes vs. the Instagram-safe area, plus inspection sheets in `work/qc/`.

`tools/color.py` holds the colour science (run it to print its self-checks).
