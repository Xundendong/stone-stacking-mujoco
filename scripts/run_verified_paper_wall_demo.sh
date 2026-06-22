#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .venv/bin/activate ]]; then
  # Keep this script runnable from a fresh terminal as well as an active venv.
  source .venv/bin/activate
fi

exec python scripts/run_official_ur5e_robotiq_wall_stack.py \
  --report reports/stability_sequence_planner_4_3_2_1_paper_light_24_s16_m28.json \
  --max-placements 10 \
  --close 0.30 \
  --grasp-retries 1 \
  --grasp-retry-close-step 0.06 \
  --upper-place-clearance 0.000 \
  --upper-place-descent-time 1.50 \
  --contact-aware-place \
  --settle-time 1.20 \
  --robot-visual clean \
  --stone-visual-roughness 0.0045 \
  --stone-visual-subdivisions 2 \
  --save-xml outputs/official_ur5e_robotiq_paper_light_4_3_2_1_10.xml \
  --output-json reports/official_ur5e_robotiq_paper_light_4_3_2_1_10.json \
  "$@"
