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
  # QC every file render.py just wrote (timeline.json "outputs": the reel, then the 4K master if any).
  # Without that list, qc.py picks its own default (the reel named in the EDL).
  mapfile -t outs < <(python3 -c '
import json, sys
tl = json.load(open("work/render/timeline.json"))
print("\n".join(o for o in (tl.get("outputs") or []) if o))' || true)
  rc=0
  if [[ ${#outs[@]} -eq 0 || -z "${outs[0]}" ]]; then
    python3 tools/qc.py --edl edit/edl.json || rc=$?
  else
    for f in "${outs[@]}"; do
      [[ -n "$f" ]] || continue
      python3 tools/qc.py "$f" --edl edit/edl.json || rc=$?
    done
  fi
  exit $rc
fi
