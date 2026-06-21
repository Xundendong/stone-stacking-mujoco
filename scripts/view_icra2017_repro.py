#!/usr/bin/env python3
"""Open the generated ICRA 2017 reproduction scene in MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_repro_final.xml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
