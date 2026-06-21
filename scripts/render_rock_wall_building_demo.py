#!/usr/bin/env python3
"""Render a robotic dry-stone wall building demo in MuJoCo.

This is a visual execution demo: stones are procedurally generated with a
natural irregular convex-hull mesh, and a UR10-scale visual arm performs a
scripted pick-carry-place sequence to assemble a small wall. It is intended as
the short-term advisor demo for the wall-building task; it is not yet a full
contact-grasp or articulated-UR10 dynamics reproduction.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_icra2017_grasp_sequence import (  # noqa: E402
    make_transition,
    quat_lerp,
    quat_normalize,
    set_robot_pose,
    set_stone_pose,
)
from stone_stack.rock_wall_stones import make_natural_wall_rocks  # noqa: E402
from stone_stack.rocks import FlatStone, flatten_faces, flatten_vertices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--stones", type=int, default=9)
    parser.add_argument("--rock-irregularity", type=float, default=1.0)
    parser.add_argument("--rock-subdivisions", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "rock_wall_building_demo.mp4")
    parser.add_argument("--frames-dir", type=Path, default=PROJECT_ROOT / "outputs" / "rock_wall_building_frames")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "rock_wall_building_demo.json")
    parser.add_argument("--save-final-xml", type=Path, default=PROJECT_ROOT / "outputs" / "rock_wall_building_final.xml")
    return parser.parse_args()


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


def supply_pose(index: int, stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
    x = -0.38 + 0.19 * (index % 5)
    y = 0.43 + 0.135 * (index // 5)
    z = 0.5 * stone.thickness + 0.012
    yaw = -0.35 + 0.23 * index
    return np.array([x, y, z], dtype=float), yaw_quat(yaw)


def build_wall_plan(stones: list[FlatStone]) -> tuple[list[list[str]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Create a staggered 4-3-2 stone wall layout."""

    courses = [stones[:4], stones[4:7], stones[7:9]]
    gap = 0.010
    y = -0.025
    course_names: list[list[str]] = []
    final_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    previous_centers: list[float] = []
    course_base_z = 0.0
    for course_index, course in enumerate(courses):
        course_names.append([stone.name for stone in course])
        if course_index == 0:
            total = sum(stone.length for stone in course) + gap * (len(course) - 1)
            cursor = -0.5 * total
            centers = []
            for stone in course:
                x = cursor + 0.5 * stone.length
                centers.append(x)
                cursor += stone.length + gap
        else:
            centers = [
                0.5 * (previous_centers[i] + previous_centers[i + 1])
                for i in range(len(course))
            ]

        course_height = max(stone.thickness for stone in course)
        for stone, x in zip(course, centers):
            yaw = (-0.08 if course_index % 2 == 0 else 0.08) + 0.04 * math.sin(13.0 * x)
            z = course_base_z + 0.5 * stone.thickness + 0.004
            final_poses[stone.name] = (np.array([x, y, z], dtype=float), yaw_quat(yaw))
        previous_centers = centers
        course_base_z += course_height * 0.90

    return course_names, final_poses


def build_wall_scene(
    stones: list[FlatStone],
    body_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> str:
    body_poses = body_poses or {}
    mesh_assets = []
    bodies = []
    for index, stone in enumerate(stones):
        pos, quat = body_poses.get(stone.name, supply_pose(index, stone))
        mesh_assets.append(
            f'<mesh name="{escape(stone.name)}_mesh" '
            f'vertex="{flatten_vertices(stone.vertices)}" '
            f'face="{flatten_faces(stone.faces)}"/>'
        )
        bodies.append(
            f'''
    <body name="{escape(stone.name)}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">
      <freejoint name="{escape(stone.name)}_free"/>
      <geom name="{escape(stone.name)}_geom" type="mesh" mesh="{escape(stone.name)}_mesh"
            mass="{stone.mass:.6g}" rgba="{_fmt(stone.rgba)}"
            friction="0.65 0.012 0.0001" condim="4"
            solref="0.006 1" solimp="0.90 0.99 0.001"/>
    </body>'''
        )

    return f'''<mujoco model="rock_wall_building_demo">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0025" integrator="implicitfast" cone="elliptic"
          gravity="0 0 -9.81" iterations="120"/>
  <size nconmax="1600" njmax="3200"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.28 0.28 0.28" diffuse="0.70 0.70 0.68" specular="0.12 0.12 0.12"/>
  </visual>
  <asset>
    <texture name="table_grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.54 0.54 0.50" rgb2="0.42 0.42 0.39"/>
    <material name="table_mat" texture="table_grid" texrepeat="5 4" reflectance="0.025"/>
    <material name="ur10_white" rgba="0.86 0.88 0.88 1"/>
    <material name="ur10_blue" rgba="0.05 0.25 0.58 1"/>
    <material name="ur10_dark" rgba="0.08 0.09 0.10 1"/>
    <material name="ur10_metal" rgba="0.64 0.64 0.60 1"/>
    {' '.join(mesh_assets)}
  </asset>
  <worldbody>
    <light name="key" pos="-0.65 -1.0 1.8" dir="0 0 -1" diffuse="1.00 0.98 0.92"/>
    <light name="fill" pos="0.7 0.8 1.3" dir="0 0 -1" diffuse="0.55 0.55 0.52"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="0.95 0.75 0.025"
          material="table_mat" friction="0.75 0.01 0.0001" condim="4"/>

    <geom name="ur10_base" type="cylinder" pos="-0.54 -0.36 0.070"
          size="0.115 0.070" material="ur10_dark" contype="0" conaffinity="0"/>
    <geom name="ur10_base_ring" type="cylinder" pos="-0.54 -0.36 0.145"
          size="0.090 0.025" material="ur10_blue" contype="0" conaffinity="0"/>
    <geom name="ur10_shoulder_joint" type="sphere" pos="-0.54 -0.36 0.245"
          size="0.087" material="ur10_blue" contype="0" conaffinity="0"/>

    <body name="upper_arm_mocap" mocap="true" pos="-0.45 -0.25 0.42">
      <geom name="upper_arm" type="capsule" fromto="0 0 -0.31 0 0 0.31"
            size="0.052" material="ur10_white" contype="0" conaffinity="0"/>
      <geom name="upper_arm_side" type="capsule" fromto="0.035 0 -0.28 0.035 0 0.28"
            size="0.026" material="ur10_blue" contype="0" conaffinity="0"/>
    </body>
    <body name="forearm_mocap" mocap="true" pos="-0.20 -0.12 0.45">
      <geom name="forearm" type="capsule" fromto="0 0 -0.31 0 0 0.31"
            size="0.044" material="ur10_white" contype="0" conaffinity="0"/>
      <geom name="forearm_side" type="capsule" fromto="0.030 0 -0.27 0.030 0 0.27"
            size="0.022" material="ur10_blue" contype="0" conaffinity="0"/>
    </body>
    <body name="elbow_mocap" mocap="true" pos="-0.3 -0.2 0.55">
      <geom name="elbow_joint" type="sphere" size="0.075" material="ur10_blue"
            contype="0" conaffinity="0"/>
    </body>
    <body name="wrist_mocap" mocap="true" pos="0 0 0.4">
      <geom name="wrist_1" type="cylinder" euler="1.5708 0 0" size="0.052 0.050"
            material="ur10_dark" contype="0" conaffinity="0"/>
      <geom name="wrist_2" type="cylinder" pos="0 0 -0.055" size="0.043 0.035"
            material="ur10_blue" contype="0" conaffinity="0"/>
    </body>
    <body name="palm_mocap" mocap="true" pos="0 0 0.35">
      <geom name="ft150_sensor" type="cylinder" pos="0 0 0.025" size="0.048 0.018"
            material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="robotiq_palm" type="cylinder" pos="0 0 -0.020" size="0.060 0.030"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_a_mocap" mocap="true" pos="0 0 0.3">
      <geom name="finger_a" type="capsule" fromto="0 0 -0.060 0 0 0.060"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="finger_a_tip" type="box" pos="0 0 -0.078" size="0.018 0.011 0.020"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_b_mocap" mocap="true" pos="0 0 0.3">
      <geom name="finger_b" type="capsule" fromto="0 0 -0.060 0 0 0.060"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="finger_b_tip" type="box" pos="0 0 -0.078" size="0.018 0.011 0.020"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_c_mocap" mocap="true" pos="0 0 0.3">
      <geom name="finger_c" type="capsule" fromto="0 0 -0.060 0 0 0.060"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="finger_c_tip" type="box" pos="0 0 -0.078" size="0.018 0.011 0.020"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>

    {''.join(bodies)}
  </worldbody>
</mujoco>
'''


def append_frame(frames: list[dict], stone_poses: dict[str, tuple[np.ndarray, np.ndarray]], palm: np.ndarray, opening: float):
    frames.append(
        {
            "stone_poses": {name: (pose[0].copy(), pose[1].copy()) for name, pose in stone_poses.items()},
            "palm": palm.copy(),
            "opening": float(opening),
        }
    )


def render_frames(
    model,
    data,
    frames: list[dict],
    frames_dir: Path,
    width: int,
    height: int,
):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.02, 0.02, 0.18]
    camera.distance = 1.06
    camera.elevation = -17.0

    for frame_index, frame in enumerate(frames):
        for stone_name, (pos, quat) in frame["stone_poses"].items():
            set_stone_pose(model, data, stone_name, pos, quat)
        set_robot_pose(model, data, frame["palm"], frame["opening"])
        mujoco.mj_forward(model, data)
        u = frame_index / max(1, len(frames) - 1)
        camera.azimuth = 128.0 + 10.0 * math.sin(2.0 * math.pi * (u - 0.08))
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        Image.fromarray(image).save(frames_dir / f"frame_{frame_index:05d}.png")

    renderer.close()


def encode_video(frames_dir: Path, fps: int, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write MP4 video")
    if args.stones < 9:
        raise SystemExit("--stones must be at least 9 for the 4-3-2 wall demo")

    stones = make_natural_wall_rocks(
        seed=args.seed,
        count=args.stones,
        irregularity=args.rock_irregularity,
        subdivisions=args.rock_subdivisions,
    )
    wall_stones = stones[:9]
    course_names, final_poses = build_wall_plan(wall_stones)

    initial_poses = {stone.name: supply_pose(index, stone) for index, stone in enumerate(stones)}
    current_poses = {name: (pos.copy(), quat.copy()) for name, (pos, quat) in initial_poses.items()}

    xml = build_wall_scene(stones)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    frames: list[dict] = []
    home = np.array([-0.05, -0.18, 0.54], dtype=float)
    palm = home.copy()
    open_width = 0.106
    closed_width = 0.046
    palm_to_stone = np.array([0.0, 0.0, 0.105], dtype=float)

    for _ in range(args.fps):
        append_frame(frames, current_poses, palm, open_width)

    place_order = [name for course in course_names for name in course]
    for name in place_order:
        pick_pos, pick_quat = current_poses[name]
        final_pos, final_quat = final_poses[name]
        above_pick = pick_pos + np.array([0.0, 0.0, 0.180])
        at_pick = pick_pos + palm_to_stone
        above_place = final_pos + np.array([0.0, 0.0, 0.205])
        at_place = final_pos + palm_to_stone

        for p in make_transition(int(0.48 * args.fps), palm, above_pick):
            palm = p
            append_frame(frames, current_poses, palm, open_width)
        for p in make_transition(int(0.32 * args.fps), above_pick, at_pick):
            palm = p
            append_frame(frames, current_poses, palm, open_width)
        for i in range(max(1, int(0.22 * args.fps))):
            t = i / max(1, int(0.22 * args.fps) - 1)
            opening = (1.0 - t) * open_width + t * closed_width
            append_frame(frames, current_poses, palm, opening)

        carry_start = (pick_pos.copy(), pick_quat.copy())
        carry_end = (final_pos.copy(), final_quat.copy())
        for p in make_transition(int(0.36 * args.fps), at_pick, above_pick):
            palm = p
            current_poses[name] = (p - palm_to_stone, pick_quat)
            append_frame(frames, current_poses, palm, closed_width)
        for i, p in enumerate(make_transition(int(0.70 * args.fps), above_pick, above_place)):
            t = i / max(1, int(0.70 * args.fps) - 1)
            palm = p
            carried_pos = (1.0 - t) * carry_start[0] + t * (carry_end[0] + np.array([0.0, 0.0, 0.105]))
            carried_quat = quat_lerp(carry_start[1], carry_end[1], t)
            current_poses[name] = (carried_pos, carried_quat)
            append_frame(frames, current_poses, palm, closed_width)
        for i, p in enumerate(make_transition(int(0.34 * args.fps), above_place, at_place)):
            t = i / max(1, int(0.34 * args.fps) - 1)
            palm = p
            current_poses[name] = ((1.0 - t) * (final_pos + np.array([0.0, 0.0, 0.105])) + t * final_pos, final_quat)
            append_frame(frames, current_poses, palm, closed_width)

        current_poses[name] = (final_pos.copy(), quat_normalize(final_quat.copy()))
        for i in range(max(1, int(0.22 * args.fps))):
            t = i / max(1, int(0.22 * args.fps) - 1)
            opening = (1.0 - t) * closed_width + t * open_width
            append_frame(frames, current_poses, palm, opening)
        retreat = final_pos + np.array([0.05, -0.12, 0.235])
        for p in make_transition(int(0.34 * args.fps), palm, retreat):
            palm = p
            append_frame(frames, current_poses, palm, open_width)

    for p in make_transition(args.fps, palm, home):
        palm = p
        append_frame(frames, current_poses, palm, open_width)
    for _ in range(args.fps):
        append_frame(frames, current_poses, palm, open_width)

    final_scene_xml = build_wall_scene(stones, {**initial_poses, **final_poses})
    args.save_final_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_final_xml.write_text(final_scene_xml, encoding="utf-8")

    render_frames(model, data, frames, args.frames_dir, args.width, args.height)
    encode_video(args.frames_dir, args.fps, args.output)

    report = {
        "task": "robotic dry-stone wall building visual demo",
        "parameters": {
            "seed": args.seed,
            "stone_count": args.stones,
            "wall_stone_count": len(wall_stones),
            "rock_irregularity": args.rock_irregularity,
            "rock_subdivisions": args.rock_subdivisions,
            "fps": args.fps,
        },
        "method_status": (
            "Natural convex-hull rock generation and scripted UR10-scale pick-carry-place wall assembly. "
            "This is a visual demo, not yet a full contact-grasp or articulated UR10 dynamics reproduction."
        ),
        "courses": course_names,
        "place_order": place_order,
        "stones": [
            {
                "name": stone.name,
                "vertices": len(stone.vertices),
                "faces": len(stone.faces),
                "length_m": stone.length,
                "width_m": stone.width,
                "thickness_m": stone.thickness,
                "mass_kg": stone.mass,
            }
            for stone in stones
        ],
        "final_poses": {
            name: {"pos": pose[0].tolist(), "quat": pose[1].tolist()}
            for name, pose in final_poses.items()
        },
        "outputs": {
            "video": str(args.output),
            "final_xml": str(args.save_final_xml),
            "frames_dir": str(args.frames_dir),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"video": str(args.output), "report": str(args.report), "final_xml": str(args.save_final_xml)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
