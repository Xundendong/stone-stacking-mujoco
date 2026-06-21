#!/usr/bin/env python3
"""Replay planner results as a UR10-scale pick-place execution in MuJoCo.

This script is the execution-layer viewer for the hybrid path. It reads the
planner report, reconstructs the generated From-Rocks-to-Walls stones, and
shows a UR10-scale visual arm moving to each selected stone, closing the
gripper, carrying it to the selected target pose, releasing it, and then
syncing to the MuJoCo-settled planner state.

The grasp is kinematic at this stage: the carried stone follows the gripper
during transport. Real contact-grasp execution is a separate next step.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.rock_wall_stones import make_rock_wall_stones
from stone_stack.rocks import FlatStone, flatten_faces, flatten_vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "hybrid_icra_wall_planner.json")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Build the model and motion plan, then exit.")
    return parser.parse_args()


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1.0e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


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
    x = -0.36 + 0.18 * (index % 5)
    y = 0.44 + 0.14 * (index // 5)
    z = 0.5 * stone.thickness + 0.014
    return np.array([x, y, z], dtype=float), yaw_quat(0.22 * index)


def quat_from_z_axis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    v = direction / n
    z = np.array([0.0, 0.0, 1.0])
    dot = float(z @ v)
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    axis = np.cross(z, v)
    return quat_normalize(np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float))


def build_execution_scene(stones: list[FlatStone]) -> str:
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
            friction="0.65 0.012 0.0001" condim="4"
            solref="0.006 1" solimp="0.90 0.99 0.001"/>
    </body>'''
        )

    return f'''<mujoco model="hybrid_icra_arm_execution">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0025" integrator="implicitfast" cone="elliptic"
          gravity="0 0 -9.81" iterations="120"/>
  <size nconmax="1800" njmax="3600"/>
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


def freejoint_address(model, stone_name: str) -> int:
    jid = model.joint(f"{stone_name}_free").id
    return int(model.jnt_qposadr[jid])


def set_stone_pose(model, data, name: str, pos: np.ndarray, quat: np.ndarray):
    qadr = freejoint_address(model, name)
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
    set_mocap(model, data, "upper_arm_mocap", 0.5 * (shoulder + elbow), quat_from_z_axis(elbow - shoulder))
    set_mocap(model, data, "forearm_mocap", 0.5 * (elbow + wrist), quat_from_z_axis(wrist - elbow))
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
        set_mocap(model, data, name, palm_pos + radial * opening + np.array([0.0, 0.0, -0.075]), quat_from_z_axis(direction))


def report_stones(report: dict) -> list[FlatStone]:
    params = report["parameters"]
    return make_rock_wall_stones(
        seed=int(params["stone_seed"]),
        count=int(params["stones"]),
        irregularity=float(params["rock_irregularity"]),
        subdivisions=int(params["rock_subdivisions"]),
    )


def committed_poses(state: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (np.asarray(pose["pos"], dtype=float), np.asarray(pose["quat"], dtype=float))
        for name, pose in state["poses"].items()
    }


def apply_poses(model, data, poses: dict[str, tuple[np.ndarray, np.ndarray]]):
    for name, (pos, quat) in poses.items():
        set_stone_pose(model, data, name, pos, quat)


def render_step(model, data, viewer, poses, palm, opening):
    import mujoco

    apply_poses(model, data, poses)
    set_robot_pose(model, data, palm, opening)
    mujoco.mj_forward(model, data)
    viewer.sync()


def interpolate(start: np.ndarray, end: np.ndarray, frames: int):
    for frame in range(max(1, frames)):
        t = smooth(frame / max(1, frames - 1))
        yield (1.0 - t) * start + t * end


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    trajectory = report.get("trajectory")
    if not trajectory:
        raise SystemExit("Missing trajectory. Re-run: python scripts/run_hybrid_icra_wall_planner.py")

    import mujoco
    import mujoco.viewer

    stones = report_stones(report)
    stone_names = [stone.name for stone in stones]
    model = mujoco.MjModel.from_xml_string(build_execution_scene(stones))
    data = mujoco.MjData(model)

    wall_order = [entry["name"] for entry in report["wall"]]
    initial_poses = {stone.name: supply_pose(index, stone) for index, stone in enumerate(stones)}
    committed_states = [committed_poses(state) for state in trajectory]
    if args.check_only:
        print(
            json.dumps(
                {
                    "stones": len(stones),
                    "placements": len(wall_order),
                    "trajectory_states": len(trajectory),
                    "model_nbody": model.nbody,
                    "model_ngeom": model.ngeom,
                    "model_nmesh": model.nmesh,
                },
                indent=2,
            )
        )
        return 0

    home = np.array([-0.05, -0.18, 0.55], dtype=float)
    palm = home.copy()
    open_width = 0.112
    closed_width = 0.044
    palm_to_stone = np.array([0.0, 0.0, 0.110], dtype=float)
    frames = max(2, int(18 / max(args.speed, 0.05)))
    hold_dt = 0.025 / max(args.speed, 0.05)
    current_poses = {name: (pos.copy(), quat.copy()) for name, (pos, quat) in initial_poses.items()}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_poses = {name: (pos.copy(), quat.copy()) for name, (pos, quat) in initial_poses.items()}
            palm = home.copy()
            for name in wall_order:
                step_index = wall_order.index(name) + 1
                pick_pos, pick_quat = current_poses[name]
                final_pos, final_quat = committed_states[step_index][name]

                above_pick = pick_pos + np.array([0.0, 0.0, 0.20])
                at_pick = pick_pos + palm_to_stone
                above_place = final_pos + np.array([0.0, 0.0, 0.22])
                at_place = final_pos + palm_to_stone

                for p in interpolate(palm, above_pick, frames):
                    palm = p
                    render_step(model, data, viewer, current_poses, palm, open_width)
                    time.sleep(hold_dt)
                for p in interpolate(above_pick, at_pick, max(2, frames // 2)):
                    palm = p
                    render_step(model, data, viewer, current_poses, palm, open_width)
                    time.sleep(hold_dt)
                for p in interpolate(np.array([open_width]), np.array([closed_width]), max(2, frames // 3)):
                    render_step(model, data, viewer, current_poses, palm, float(p[0]))
                    time.sleep(hold_dt)

                for p in interpolate(at_pick, above_pick, max(2, frames // 2)):
                    palm = p
                    current_poses[name] = (palm - palm_to_stone, pick_quat)
                    render_step(model, data, viewer, current_poses, palm, closed_width)
                    time.sleep(hold_dt)
                for frame, p in enumerate(interpolate(above_pick, above_place, frames)):
                    t = smooth(frame / max(1, frames - 1))
                    palm = p
                    current_poses[name] = (palm - palm_to_stone, quat_lerp(pick_quat, final_quat, t))
                    render_step(model, data, viewer, current_poses, palm, closed_width)
                    time.sleep(hold_dt)
                for p in interpolate(above_place, at_place, max(2, frames // 2)):
                    palm = p
                    current_poses[name] = (palm - palm_to_stone, final_quat)
                    render_step(model, data, viewer, current_poses, palm, closed_width)
                    time.sleep(hold_dt)

                current_poses = {
                    stone_name: (pos.copy(), quat.copy())
                    for stone_name, (pos, quat) in committed_states[step_index].items()
                }
                for p in interpolate(np.array([closed_width]), np.array([open_width]), max(2, frames // 3)):
                    render_step(model, data, viewer, current_poses, palm, float(p[0]))
                    time.sleep(hold_dt)
                retreat = final_pos + np.array([0.06, -0.14, 0.24])
                for p in interpolate(palm, retreat, max(2, frames // 2)):
                    palm = p
                    render_step(model, data, viewer, current_poses, palm, open_width)
                    time.sleep(hold_dt)
                if not viewer.is_running():
                    return 0

            for p in interpolate(palm, home, frames):
                palm = p
                render_step(model, data, viewer, current_poses, palm, open_width)
                time.sleep(hold_dt)
            if not args.loop:
                while viewer.is_running():
                    render_step(model, data, viewer, current_poses, home, open_width)
                    time.sleep(0.05)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
