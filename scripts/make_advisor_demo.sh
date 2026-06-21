#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv. Create it with: python3.10 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt" >&2
  exit 2
fi

.venv/bin/python scripts/run_paper_pose_demo.py \
  --seed 3 \
  --stones 4 \
  --levels 4 \
  --samples-per-stone 5 \
  --output-json reports/paper_pose_demo.json \
  --output-html outputs/paper_pose_demo.html \
  --save-final-xml outputs/paper_pose_demo_final.xml

.venv/bin/python scripts/render_paper_demo_video.py \
  --input-json reports/paper_pose_demo.json \
  --output-mp4 outputs/paper_pose_demo.mp4 \
  --fps 24

echo "Demo video: outputs/paper_pose_demo.mp4"

