#!/usr/bin/env python3
"""Render a still image from a wall-stack execution report."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "official_ur5e_robotiq_default_safe_queue_6.xml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "official_ur5e_robotiq_default_safe_queue_6.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "assets" / "official_ur5e_wall_snapshot.png",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=142.0)
    parser.add_argument("--elevation", type=float, default=-22.0)
    parser.add_argument("--distance", type=float, default=0.88)
    return parser.parse_args()


def set_freejoint_pose(model, data, joint_name: str, pos: list[float], quat: list[float]) -> None:
    joint_id = model.joint(joint_name).id
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    for step in report["steps"]:
        set_freejoint_pose(
            model,
            data,
            f"{step['name']}_free",
            step["final_pos"],
            step["final_quat"],
        )

    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.11]
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
