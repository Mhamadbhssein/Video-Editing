Put the soundtrack candidates here (.m4a / .mp3 / .wav …). The original files are never modified.
`tools/audio_analysis.py` analyses every file and ranks candidate excerpts that start after a pause and end on a
phrase end (`work/audio/<track>.json` → `suggested_excerpts`, plus waveform / spectrogram pages). The excerpt
actually used is the music `in` point chosen in `edit/edl.json`. From that point the track must run for the
whole reel; render.py checks this before rendering.
