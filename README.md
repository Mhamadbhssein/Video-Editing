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
| `assets/fonts/` | Amiri, IBM Plex Sans Arabic, Noto Naskh Arabic (+ OFL licences) |
| `tools/` | the pipeline (Python 3 + numpy + Pillow + FFmpeg) |
| `edit/edl.json` | **the edit** (template: `edit/edl.template.json`), Arabic line candidates in `edit/lines_bank.md` |
| `output/` | `Madinah_Final_Reel.mp4` (+ `Madinah_Master_2160x3840.mp4` when the footage supports it) |
| `work/` | regenerated scratch: contact sheets, filmstrips, proxies, audio pages, segments, QC sheets |

## Pipeline

```bash
tools/run_all.sh study     # 1-3: inventory, footage review, audio analysis
# write edit/edl.json from what the review shows
tools/run_all.sh render    # 4-5: render + QC
```

1. `inventory.py` — resolution, rotation, orientation, fps / VFR, codec, bit depth, primaries/transfer/matrix,
   HLG / PQ / Dolby Vision detection, audio streams → `work/inventory.md`.
2. `review.py` — per clip: contact sheet, dense 4 fps filmstrips, 540 px SDR proxy, metrics
   (exposure, clipped highlights, sharpness, camera speed, jitter) and suggested strongest moments.
3. `audio_analysis.py` — waveform / spectrogram pages, EBU R128 loudness, tempo + beat grid,
   pauses and phrase spans (never cut inside a recited or sung phrase), energy changes, ranked 30–45 s excerpts.
4. `render.py` — per shot: accurate seek → optional vid.stab → 9:16 reframe (static crop or sub-pixel push)
   → spline36 resize → light denoise → retime (real frames / even-cadence blend / motion interpolation)
   → **one 3D LUT per shot** (HDR→SDR BT.2100/BT.2408 conversion + exposure/white balance + the film's look)
   → ProRes 422 HQ segment; then frame-exact cuts and centred dissolves, Arabic titles (libass +
   HarfBuzz + FriBidi), H.264 High / yuv420p / BT.709 / CFR / +faststart, AAC-LC 48 kHz 320 kb/s.
   Audio: music excerpt + optional clip ambience, fades, two-pass loudness to −14 LUFS with a
   −1.5 dBTP ceiling using gain only (no limiter, no compression, no pitch change).
5. `qc.py` — delivery spec, black/frozen/duplicated frames, unexpected cuts, loudness/true peak,
   silences, title boxes vs. the Instagram-safe area, plus inspection sheets in `work/qc/`.

`tools/color.py` holds the colour science (run it to print its self-checks).
