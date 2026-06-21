#!/usr/bin/env python3
"""Contact-grasp lift test for From-Rocks-to-Walls stones.

This is the first real gripper stage: a MuJoCo gripper with collision pads,
slide joints and position actuators closes on an irregular stone and tries to
lift it by contact friction. The gripper palm is moved by a mocap weld, which
stands in for the robot end-effector controller; the grasp itself is not an
attach constraint.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.rock_wall_stones import make_rock_wall_stones
from stone_stack.rocks import FlatStone, flatten_faces, flatten_vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stone-seed", type=int, default=17)
    parser.add_argument("--stone-index", type=int, default=1)
    parser.add_argument("--rock-irregularity", type=float, default=1.0)
    parser.add_argument("--rock-subdivisions", type=int, default=5)
    parser.add_argument("--gripper-friction", type=float, default=3.2)
    parser.add_argument("--close-extra", type=float, default=0.020)
    parser.add_argument("--lift-height", type=float, default=0.15)
    parser.add_argument("--save-xml", type=Path, default=PROJECT_ROOT / "outputs" / "contact_gripper_lift_test.xml")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "contact_gripper_lift_test.json")
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def _fmt(values) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1.0e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def stone_start_pose(stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(stone.vertices, dtype=float)
    pos = np.array([0.0, 0.0, -float(vertices[:, 2].min()) + 0.004], dtype=float)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return pos, quat


def build_lift_scene(stone: FlatStone, gripper_friction: float) -> str:
    stone_pos, stone_quat = stone_start_pose(stone)
    stone_mesh = (
        f'<mesh name="{escape(stone.name)}_mesh" '
        f'vertex="{flatten_vertices(stone.vertices)}" '
        f'face="{flatten_faces(stone.faces)}"/>'
    )
    return f'''<mujoco model="contact_gripper_lift_test">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0015" integrator="implicitfast" cone="elliptic"
          gravity="0 0 -9.81" iterations="150"/>
  <size nconmax="900" njmax="1800"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.30 0.30 0.30" diffuse="0.70 0.70 0.68"/>
  </visual>
  <default>
    <joint damping="1.8" armature="0.005"/>
    <geom solref="0.004 1" solimp="0.94 0.99 0.001"/>
  </default>
  <asset>
    <texture name="table_grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.55 0.55 0.52" rgb2="0.42 0.42 0.40"/>
    <material name="table_mat" texture="table_grid" texrepeat="4 4" reflectance="0.03"/>
    <material name="palm_mat" rgba="0.08 0.09 0.10 1"/>
    <material name="finger_mat" rgba="0.52 0.53 0.50 1"/>
    {stone_mesh}
  </asset>
  <worldbody>
    <light name="key" pos="-0.5 -0.8 1.3" dir="0 0 -1" diffuse="0.95 0.95 0.90"/>
    <camera name="overview" pos="0.42 -0.72 0.42" xyaxes="0.86 0.51 0 -0.23 0.39 0.89"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="0.55 0.45 0.025"
          material="table_mat" friction="0.90 0.012 0.0001" condim="4"/>

    <body name="gripper_mocap" mocap="true" pos="0 0 0.20"/>
    <body name="gripper_palm" pos="0 0 0.20">
      <freejoint name="gripper_free"/>
      <geom name="palm_collision" type="box" pos="0 0 0.050" size="0.070 0.030 0.022"
            material="palm_mat" mass="0.35" contype="0" conaffinity="0"/>
      <body name="left_finger" pos="0 0.095 0">
        <joint name="left_slide" type="slide" axis="0 -1 0" range="0 0.085" limited="true"/>
        <geom name="left_pad" type="box" pos="0 0 0" size="0.065 0.012 0.058"
              material="finger_mat" mass="0.12"
              friction="{gripper_friction:.6g} 0.08 0.003" condim="4"
              solref="0.003 1" solimp="0.96 0.995 0.0005"/>
      </body>
      <body name="right_finger" pos="0 -0.095 0">
        <joint name="right_slide" type="slide" axis="0 1 0" range="0 0.085" limited="true"/>
        <geom name="right_pad" type="box" pos="0 0 0" size="0.065 0.012 0.058"
              material="finger_mat" mass="0.12"
              friction="{gripper_friction:.6g} 0.08 0.003" condim="4"
              solref="0.003 1" solimp="0.96 0.995 0.0005"/>
      </body>
    </body>

    <body name="{escape(stone.name)}" pos="{_fmt(stone_pos)}" quat="{_fmt(stone_quat)}">
      <freejoint name="{escape(stone.name)}_free"/>
      <geom name="{escape(stone.name)}_geom" type="mesh" mesh="{escape(stone.name)}_mesh"
            mass="{stone.mass:.6g}" rgba="{_fmt(stone.rgba)}"
            friction="0.95 0.018 0.0001" condim="4"
            solref="0.005 1" solimp="0.92 0.99 0.001"/>
    </body>
  </worldbody>
  <equality>
    <weld name="mocap_to_palm" body1="gripper_mocap" body2="gripper_palm"
          solref="0.004 1" solimp="0.95 0.995 0.001"/>
  </equality>
  <actuator>
    <position name="left_close" joint="left_slide" kp="850" kv="35" ctrlrange="0 0.085" forcerange="-220 220"/>
    <position name="right_close" joint="right_slide" kp="850" kv="35" ctrlrange="0 0.085" forcerange="-220 220"/>
  </actuator>
</mujoco>
'''


def mocap_id(model, body_name: str) -> int:
    bid = model.body(body_name).id
    return int(model.body_mocapid[bid])


def joint_qpos_addr(model, joint_name: str) -> int:
    jid = model.joint(joint_name).id
    return int(model.jnt_qposadr[jid])


def joint_qvel_addr(model, joint_name: str) -> int:
    jid = model.joint(joint_name).id
    return int(model.jnt_dofadr[jid])


def body_pose(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = model.body(body_name).id
    return data.xpos[bid].copy(), quat_normalize(data.xquat[bid].copy())


def set_mocap(model, data, pos: np.ndarray):
    data.mocap_pos[mocap_id(model, "gripper_mocap")] = np.asarray(pos, dtype=float)
    data.mocap_quat[mocap_id(model, "gripper_mocap")] = np.array([1.0, 0.0, 0.0, 0.0])


def step_for(mujoco, model, data, seconds: float, viewer=None):
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def move_gripper(mujoco, model, data, start, end, q_start, q_end, seconds: float, viewer=None):
    steps = max(1, int(seconds / model.opt.timestep))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for i in range(steps):
        t = (i + 1) / steps
        s = t * t * (3.0 - 2.0 * t)
        pos = (1.0 - s) * start + s * end
        q = (1.0 - s) * q_start + s * q_end
        set_mocap(model, data, pos)
        data.ctrl[:] = q
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def prepare_trial(args: argparse.Namespace):
    import mujoco

    stones = make_rock_wall_stones(
        seed=args.stone_seed,
        count=max(args.stone_index + 1, 2),
        irregularity=args.rock_irregularity,
        subdivisions=args.rock_subdivisions,
    )
    stone = stones[args.stone_index]
    xml = build_lift_scene(stone, args.gripper_friction)
    args.save_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_xml.write_text(xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return mujoco, stone, model, data


def run_trial(args: argparse.Namespace, mujoco, stone: FlatStone, model, data, viewer=None) -> dict:
    stone_pos0, _ = stone_start_pose(stone)
    grasp_z = stone_pos0[2] + 0.5 * stone.thickness
    above = np.array([0.0, 0.0, grasp_z + 0.16])
    grasp = np.array([0.0, 0.0, grasp_z])
    lifted = grasp + np.array([0.0, 0.0, args.lift_height])
    open_q = 0.0
    # Initial inner gap is roughly 0.166 m. Close below stone width to create normal force.
    close_q = float(np.clip((0.166 - max(0.030, stone.width - args.close_extra)) * 0.5, 0.0, 0.085))

    set_mocap(model, data, above)
    data.ctrl[:] = open_q
    mujoco.mj_forward(model, data)
    step_for(mujoco, model, data, 0.4, viewer)
    initial_stone_pos, _ = body_pose(model, data, stone.name)

    move_gripper(mujoco, model, data, above, grasp, open_q, open_q, 0.6, viewer)
    move_gripper(mujoco, model, data, grasp, grasp, open_q, close_q, 0.8, viewer)
    step_for(mujoco, model, data, 0.4, viewer)
    closed_stone_pos, _ = body_pose(model, data, stone.name)
    move_gripper(mujoco, model, data, grasp, lifted, close_q, close_q, 1.0, viewer)
    step_for(mujoco, model, data, 0.8, viewer)
    lifted_stone_pos, _ = body_pose(model, data, stone.name)

    qvel_addr = joint_qvel_addr(model, f"{stone.name}_free")
    speed = float(np.linalg.norm(data.qvel[qvel_addr : qvel_addr + 3]))
    lift_gain = float(lifted_stone_pos[2] - initial_stone_pos[2])
    xy_error = float(np.linalg.norm(lifted_stone_pos[:2] - lifted[:2]))
    left_q = float(data.qpos[joint_qpos_addr(model, "left_slide")])
    right_q = float(data.qpos[joint_qpos_addr(model, "right_slide")])
    success = bool(lift_gain > 0.08 and xy_error < 0.08)
    result = {
        "success": success,
        "stone": {
            "name": stone.name,
            "vertices": len(stone.vertices),
            "faces": len(stone.faces),
            "length_m": stone.length,
            "width_m": stone.width,
            "thickness_m": stone.thickness,
            "mass_kg": stone.mass,
        },
        "gripper": {
            "type": "actuated_parallel_contact_gripper",
            "friction": args.gripper_friction,
            "close_q_target": close_q,
            "left_q_final": left_q,
            "right_q_final": right_q,
        },
        "initial_stone_pos": initial_stone_pos.tolist(),
        "closed_stone_pos": closed_stone_pos.tolist(),
        "lifted_stone_pos": lifted_stone_pos.tolist(),
        "lift_gain_m": lift_gain,
        "xy_error_m": xy_error,
        "final_linear_speed_m_s": speed,
        "xml": str(args.save_xml),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    mujoco, stone, model, data = prepare_trial(args)
    if args.view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            result = run_trial(args, mujoco, stone, model, data, viewer)
            while viewer.is_running():
                viewer.sync()
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1

    result = run_trial(args, mujoco, stone, model, data)
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
