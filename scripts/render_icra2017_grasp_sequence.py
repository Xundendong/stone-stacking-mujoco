#!/usr/bin/env python3
"""Render a MuJoCo 3D pick-carry-place sequence for the ICRA 2017 demo.

This video shows the planned stack being built stone by stone. The gripper
motion is scripted and the carried stone is attached to the gripper after the
close phase. That keeps this as a visual execution demo, while the stack poses
and selected order still come from the MuJoCo next-best-pose planner.
"""

from __future__ import annotations

import argparse
from html import escape
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

from stone_stack.paper_stones import make_paper_limestones
from stone_stack.rock_wall_stones import make_rock_wall_stones
from stone_stack.rocks import FlatStone, flatten_faces, flatten_vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "icra2017_repro.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_repro_grasp_sequence.mp4")
    parser.add_argument("--frames-dir", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_grasp_frames")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stone-seed", type=int, default=None)
    parser.add_argument("--stone-generator", choices=("icra-limestone", "from-rocks-to-walls"), default=None)
    parser.add_argument("--rock-irregularity", type=float, default=None)
    parser.add_argument("--rock-subdivisions", type=int, default=None)
    return parser.parse_args()


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1.0e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    if float(a @ b) < 0.0:
        b = -b
    return quat_normalize((1.0 - t) * a + t * b)


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def supply_pose(index: int, stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
    x = -0.23 + 0.23 * (index % 3)
    y = 0.42 + 0.18 * (index // 3)
    z = 0.5 * stone.thickness + 0.016
    yaw = 0.25 * index
    return np.array([x, y, z], dtype=float), np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_from_z_axis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = direction / n
    z = np.array([0.0, 0.0, 1.0])
    dot = float(z @ v)
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, v)
    q = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float)
    return quat_normalize(q)


def build_animation_scene(stones: list[FlatStone]) -> str:
    mesh_assets = []
    bodies = []
    for index, stone in enumerate(stones):
        pos, quat = supply_pose(index, stone)
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
            friction="0.1 0.004 0.0001" condim="4"
            solref="0.006 1" solimp="0.90 0.99 0.001"/>
    </body>'''
        )

    return f'''<mujoco model="icra2017_grasp_sequence">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0025" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>
  <size nconmax="1200" njmax="2400"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture name="table_grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.55 0.55 0.52" rgb2="0.45 0.45 0.42"/>
    <material name="table_mat" texture="table_grid" texrepeat="4 4" reflectance="0.04"/>
    <material name="ur10_white" rgba="0.86 0.88 0.88 1"/>
    <material name="ur10_blue" rgba="0.05 0.25 0.58 1"/>
    <material name="ur10_dark" rgba="0.08 0.09 0.10 1"/>
    <material name="ur10_metal" rgba="0.64 0.64 0.60 1"/>
    {' '.join(mesh_assets)}
  </asset>
  <worldbody>
    <light name="key" pos="-0.5 -1.0 1.8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="0.7 0.8 1.4" dir="0 0 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="0.85 0.70 0.025"
          material="table_mat" friction="0.6 0.01 0.0001" condim="4"/>

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
      <geom name="finger_a" type="capsule" fromto="0 0 -0.055 0 0 0.055"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="finger_a_tip" type="box" pos="0 0 -0.070" size="0.017 0.010 0.018"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_b_mocap" mocap="true" pos="0 0 0.3">
      <geom name="finger_b" type="capsule" fromto="0 0 -0.055 0 0 0.055"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="finger_b_tip" type="box" pos="0 0 -0.070" size="0.017 0.010 0.018"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_c_mocap" mocap="true" pos="0 0 0.3">
      <geom name="finger_c" type="capsule" fromto="0 0 -0.055 0 0 0.055"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="finger_c_tip" type="box" pos="0 0 -0.070" size="0.017 0.010 0.018"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>

    {''.join(bodies)}
  </worldbody>
</mujoco>
'''


def freejoint_address(model, joint_name: str) -> int:
    jid = model.joint(joint_name).id
    return int(model.jnt_qposadr[jid])


def set_stone_pose(model, data, name: str, pos: np.ndarray, quat: np.ndarray):
    qadr = freejoint_address(model, f"{name}_free")
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = quat_normalize(quat)


def mocap_id(model, body_name: str) -> int:
    bid = model.body(body_name).id
    return int(model.body_mocapid[bid])


def set_mocap(model, data, name: str, pos: np.ndarray, quat: np.ndarray):
    mid = mocap_id(model, name)
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat_normalize(quat)


def ik_points(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shoulder = np.array([-0.54, -0.36, 0.245], dtype=float)
    l1 = 0.62
    l2 = 0.62
    wrist = np.asarray(target, dtype=float)
    v = wrist - shoulder
    d = float(np.linalg.norm(v))
    if d < 1.0e-6:
        return shoulder, shoulder + np.array([0.0, 0.0, l1]), wrist
    r = v / d
    d_eff = min(max(d, 0.10), l1 + l2 - 0.02)
    a = (l1 * l1 - l2 * l2 + d_eff * d_eff) / (2.0 * d_eff)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    bend = np.array([0.0, 0.0, 1.0]) - r * float(np.array([0.0, 0.0, 1.0]) @ r)
    if np.linalg.norm(bend) < 1.0e-6:
        bend = np.array([0.0, 1.0, 0.0])
    bend = bend / np.linalg.norm(bend)
    elbow = shoulder + a * r + h * bend
    return shoulder, elbow, wrist


def set_robot_pose(model, data, palm_pos: np.ndarray, opening: float):
    shoulder, elbow, wrist = ik_points(palm_pos + np.array([-0.06, -0.04, 0.08]))
    upper_mid = 0.5 * (shoulder + elbow)
    forearm_mid = 0.5 * (elbow + wrist)
    set_mocap(model, data, "upper_arm_mocap", upper_mid, quat_from_z_axis(elbow - shoulder))
    set_mocap(model, data, "forearm_mocap", forearm_mid, quat_from_z_axis(wrist - elbow))
    set_mocap(model, data, "elbow_mocap", elbow, np.array([1.0, 0.0, 0.0, 0.0]))
    set_mocap(model, data, "wrist_mocap", wrist, np.array([1.0, 0.0, 0.0, 0.0]))
    set_mocap(model, data, "palm_mocap", palm_pos, np.array([1.0, 0.0, 0.0, 0.0]))

    dirs = [
        np.array([0.0, -1.0, -0.25]),
        np.array([-0.866, 0.50, -0.25]),
        np.array([0.866, 0.50, -0.25]),
    ]
    names = ["finger_a_mocap", "finger_b_mocap", "finger_c_mocap"]
    for name, direction in zip(names, dirs):
        radial = np.array([direction[0], direction[1], 0.0])
        radial = radial / max(np.linalg.norm(radial), 1.0e-9)
        pos = palm_pos + radial * opening + np.array([0.0, 0.0, -0.075])
        set_mocap(model, data, name, pos, quat_from_z_axis(direction))


def make_hold_frames(count: int, value):
    return [value for _ in range(count)]


def make_transition(count: int, start, end):
    frames = []
    for i in range(count):
        t = smooth(i / max(1, count - 1))
        if isinstance(start, tuple):
            pos = (1.0 - t) * start[0] + t * end[0]
            quat = quat_lerp(start[1], end[1], t)
            opening = (1.0 - t) * start[2] + t * end[2]
            carry = start[3] if i < count - 1 else end[3]
            frames.append((pos, quat, opening, carry))
        else:
            frames.append((1.0 - t) * start + t * end)
    return frames


def load_report(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def make_stones(
    stone_generator: str,
    stone_seed: int,
    rock_irregularity: float,
    rock_subdivisions: int,
) -> list[FlatStone]:
    if stone_generator == "icra-limestone":
        return make_paper_limestones(stone_seed)
    if stone_generator == "from-rocks-to-walls":
        return make_rock_wall_stones(
            seed=stone_seed,
            count=6,
            irregularity=rock_irregularity,
            subdivisions=rock_subdivisions,
        )
    raise ValueError(f"unknown stone generator: {stone_generator}")


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write MP4 video")
    report = load_report(args.report)
    params = report.get("parameters", {})
    stone_seed = args.stone_seed if args.stone_seed is not None else int(params.get("stone_seed", 17))
    stone_generator = args.stone_generator or str(params.get("stone_generator", "icra-limestone"))
    rock_irregularity = (
        args.rock_irregularity if args.rock_irregularity is not None else float(params.get("rock_irregularity", 0.75))
    )
    rock_subdivisions = (
        args.rock_subdivisions if args.rock_subdivisions is not None else int(params.get("rock_subdivisions", 5))
    )
    stones = make_stones(stone_generator, stone_seed, rock_irregularity, rock_subdivisions)
    by_name = {stone.name: stone for stone in stones}

    stack_entries = report["stack"]
    final_poses = {
        entry["name"]: (np.asarray(entry["pos"], dtype=float), np.asarray(entry["quat"], dtype=float))
        for entry in stack_entries
    }
    available = report["available_subset"]
    base_name = stack_entries[0]["name"]
    place_order = [entry["name"] for entry in stack_entries[1:]]

    initial_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, name in enumerate(available):
        if name == base_name:
            initial_poses[name] = final_poses[name]
        else:
            initial_poses[name] = supply_pose(index, by_name[name])
    for index, stone in enumerate(stones):
        if stone.name not in initial_poses:
            initial_poses[stone.name] = supply_pose(index + 3, stone)

    xml = build_animation_scene(stones)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    current_poses = {name: (pos.copy(), quat.copy()) for name, (pos, quat) in initial_poses.items()}

    home = np.array([-0.04, -0.16, 0.52], dtype=float)
    open_width = 0.095
    closed_width = 0.050
    palm_clearance = np.array([0.0, 0.0, 0.145])
    frames: list[dict] = []
    palm = home.copy()
    opening = open_width

    def append_frame(stone_poses, palm_pos, grip_opening):
        frames.append(
            {
                "stone_poses": {k: (v[0].copy(), v[1].copy()) for k, v in stone_poses.items()},
                "palm": palm_pos.copy(),
                "opening": float(grip_opening),
            }
        )

    for _ in range(args.fps):
        append_frame(current_poses, palm, opening)

    for name in place_order:
        pick_pos, pick_quat = current_poses[name]
        final_pos, final_quat = final_poses[name]
        above_pick = pick_pos + palm_clearance
        at_pick = pick_pos + np.array([0.0, 0.0, 0.095])
        above_place = final_pos + palm_clearance
        at_place = final_pos + np.array([0.0, 0.0, 0.100])

        for p in make_transition(args.fps, palm, above_pick):
            palm = p
            append_frame(current_poses, palm, open_width)
        for p in make_transition(int(0.55 * args.fps), above_pick, at_pick):
            palm = p
            append_frame(current_poses, palm, open_width)
        for i in range(int(0.45 * args.fps)):
            t = smooth(i / max(1, int(0.45 * args.fps) - 1))
            opening = (1.0 - t) * open_width + t * closed_width
            append_frame(current_poses, palm, opening)

        carry_start = (pick_pos.copy(), pick_quat.copy())
        carry_end = (final_pos.copy(), final_quat.copy())
        for p in make_transition(int(0.55 * args.fps), at_pick, above_pick):
            palm = p
            current_poses[name] = (p - palm_clearance + np.array([0.0, 0.0, 0.050]), pick_quat)
            append_frame(current_poses, palm, closed_width)
        for i, p in enumerate(make_transition(int(1.15 * args.fps), above_pick, above_place)):
            t = smooth(i / max(1, int(1.15 * args.fps) - 1))
            palm = p
            stone_pos = (1.0 - t) * carry_start[0] + t * carry_end[0] + np.array([0.0, 0.0, 0.055])
            stone_quat = quat_lerp(carry_start[1], carry_end[1], t)
            current_poses[name] = (stone_pos, stone_quat)
            append_frame(current_poses, palm, closed_width)
        for i, p in enumerate(make_transition(int(0.55 * args.fps), above_place, at_place)):
            t = smooth(i / max(1, int(0.55 * args.fps) - 1))
            palm = p
            stone_pos = (1.0 - t) * (final_pos + np.array([0.0, 0.0, 0.055])) + t * final_pos
            current_poses[name] = (stone_pos, final_quat)
            append_frame(current_poses, palm, closed_width)

        current_poses[name] = (final_pos.copy(), final_quat.copy())
        for i in range(int(0.45 * args.fps)):
            t = smooth(i / max(1, int(0.45 * args.fps) - 1))
            opening = (1.0 - t) * closed_width + t * open_width
            append_frame(current_poses, palm, opening)
        retreat = final_pos + np.array([0.10, -0.08, 0.22])
        for p in make_transition(int(0.65 * args.fps), palm, retreat):
            palm = p
            append_frame(current_poses, palm, open_width)

    for _ in range(args.fps):
        append_frame(current_poses, palm, open_width)

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    for old in args.frames_dir.glob("frame_*.png"):
        old.unlink()

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.00, 0.06, 0.24]
    camera.distance = 1.18
    camera.elevation = -23.0

    for frame_index, frame in enumerate(frames):
        for stone_name, (pos, quat) in frame["stone_poses"].items():
            set_stone_pose(model, data, stone_name, pos, quat)
        set_robot_pose(model, data, frame["palm"], frame["opening"])
        mujoco.mj_forward(model, data)
        u = frame_index / max(1, len(frames) - 1)
        camera.azimuth = 140.0 + 30.0 * math.sin(math.tau * (u - 0.10))
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        Image.fromarray(image).save(args.frames_dir / f"frame_{frame_index:05d}.png")

    renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(args.frames_dir / "frame_%05d.png"),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(args.output),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
