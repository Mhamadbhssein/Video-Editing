#!/usr/bin/env bash
# One-command pipeline.  Steps 1-3 study the material; step 4 renders edit/edl.json; step 5 checks the export.
set -euo pipefail
cd "$(dirname "$0")/.."
step="${1:-all}"
if [[ "$step" == all || "$step" == study ]]; then
  python3 tools/inventory.py footage audio
  python3 tools/review.py
  python3 tools/audio_analysis.py --clips
fi
if [[ "$step" == all || "$step" == render ]]; then
  [[ -f edit/edl.json ]] || { echo "edit/edl.json missing — write the edit first (see edit/edl.template.json)"; exit 2; }
  python3 tools/render.py --edl edit/edl.json
  python3 tools/qc.py output/Madinah_Final_Reel.mp4 --edl edit/edl.json
fi
