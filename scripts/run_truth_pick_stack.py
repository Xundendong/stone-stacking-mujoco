#!/usr/bin/env python3
"""Truth-state pick and dry-stack prototype in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.mjcf_builder import build_truth_gripper_scene
from stone_stack.rocks import make_flat_stones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stones", type=int, default=2)
    parser.add_argument("--transport-mode", choices=("oracle", "contact"), default="oracle")
    parser.add_argument("--settle-time", type=float, default=0.8)
    parser.add_argument("--wait-time", type=float, default=8.0)
    parser.add_argument(
        "--grip-width-ratio",
        type=float,
        default=0.78,
        help="Closed inner-finger spacing as a ratio of the moving stone nominal width.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-xml", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate stones and MJCF without importing MuJoCo.")
    return parser.parse_args()


def import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MuJoCo is not installed. Create a venv and run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return mujoco


def body_mocap_id(model, name: str) -> int:
    bid = model.body(name).id
    mocapid = int(model.body_mocapid[bid])
    if mocapid < 0:
        raise ValueError(f"body {name!r} is not a mocap body")
    return mocapid


def freejoint_addresses(model, joint_name: str) -> tuple[int, int]:
    jid = model.joint(joint_name).id
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def set_free_body_pose(model, data, joint_name: str, pos, quat=(1.0, 0.0, 0.0, 0.0), zero_velocity: bool = True):
    qposadr, qveladr = freejoint_addresses(model, joint_name)
    data.qpos[qposadr : qposadr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qposadr + 3 : qposadr + 7] = np.asarray(quat, dtype=float)
    if zero_velocity:
        data.qvel[qveladr : qveladr + 6] = 0.0


def get_free_body_pose(model, data, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    qposadr, _ = freejoint_addresses(model, joint_name)
    return data.qpos[qposadr : qposadr + 3].copy(), data.qpos[qposadr + 3 : qposadr + 7].copy()


def set_gripper(model, data, center, opening: float):
    """Set two mocap fingers around a gripper center.

    Opening is the distance between the inner faces of the fingers.
    """

    half_thickness = 0.014
    left_id = body_mocap_id(model, "left_finger")
    right_id = body_mocap_id(model, "right_finger")
    palm_id = body_mocap_id(model, "palm_marker")

    center = np.asarray(center, dtype=float)
    left = center + np.array([0.0, opening * 0.5 + half_thickness, 0.0])
    right = center + np.array([0.0, -opening * 0.5 - half_thickness, 0.0])

    data.mocap_pos[left_id] = left
    data.mocap_pos[right_id] = right
    data.mocap_pos[palm_id] = center + np.array([0.0, 0.0, 0.065])

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    data.mocap_quat[left_id] = quat
    data.mocap_quat[right_id] = quat
    data.mocap_quat[palm_id] = quat


def step_for(model, data, seconds: float):
    import mujoco

    steps = int(seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)


def move_gripper(model, data, start, end, opening_start, opening_end, duration, carried_joint: str | None = None):
    import mujoco

    steps = max(1, int(duration / model.opt.timestep))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for i in range(steps):
        alpha = (i + 1) / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        center = (1.0 - smooth) * start + smooth * end
        opening = (1.0 - smooth) * opening_start + smooth * opening_end
        set_gripper(model, data, center, opening)
        if carried_joint is not None:
            set_free_body_pose(model, data, carried_joint, center, zero_velocity=True)
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)


def max_body_speed(model, data, joint_names: list[str]) -> float:
    speeds = []
    for joint_name in joint_names:
        _, qveladr = freejoint_addresses(model, joint_name)
        linear = data.qvel[qveladr : qveladr + 3]
        angular = data.qvel[qveladr + 3 : qveladr + 6]
        speeds.append(float(np.linalg.norm(linear) + 0.15 * np.linalg.norm(angular)))
    return max(speeds)


def main() -> int:
    args = parse_args()
    if args.stones < 2:
        raise SystemExit("--stones must be at least 2")

    stones = make_flat_stones(args.stones, args.seed)
    xml = build_truth_gripper_scene(stones)

    if args.save_xml is not None:
        args.save_xml.parent.mkdir(parents=True, exist_ok=True)
        args.save_xml.write_text(xml, encoding="utf-8")

    if args.dry_run:
        summary = {
            "seed": args.seed,
            "stones": [
                {
                    "name": stone.name,
                    "vertices": len(stone.vertices),
                    "triangles": len(stone.faces),
                    "length": stone.length,
                    "width": stone.width,
                    "thickness": stone.thickness,
                    "mass": stone.mass,
                }
                for stone in stones
            ],
            "mjcf_bytes": len(xml.encode("utf-8")),
        }
        print(json.dumps(summary, indent=2))
        return 0

    mujoco = import_mujoco()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    open_width = max(stone.width for stone in stones[:2]) + 0.08
    close_width = max(0.045, stones[0].width * args.grip_width_ratio)
    safe_z = 0.28

    pick_joint = f"{stones[0].name}_free"
    base_joint = f"{stones[1].name}_free"
    all_joints = [f"{stone.name}_free" for stone in stones]

    set_gripper(model, data, (-0.28, 0.0, safe_z), open_width)
    mujoco.mj_forward(model, data)
    step_for(model, data, args.settle_time)

    pick_pos, pick_quat = get_free_body_pose(model, data, pick_joint)
    base_pos, _ = get_free_body_pose(model, data, base_joint)

    grasp_z = pick_pos[2] + 0.006
    above_pick = np.array([pick_pos[0], pick_pos[1], safe_z])
    at_pick = np.array([pick_pos[0], pick_pos[1], grasp_z])

    target_center = np.array(
        [
            base_pos[0],
            base_pos[1],
            base_pos[2] + 0.5 * stones[1].thickness + 0.5 * stones[0].thickness + 0.006,
        ]
    )
    above_place = np.array([target_center[0], target_center[1], safe_z])

    move_gripper(model, data, above_pick, at_pick, open_width, open_width, 0.45)
    move_gripper(model, data, at_pick, at_pick, open_width, close_width, 0.35)

    carried_joint = pick_joint if args.transport_mode == "oracle" else None
    move_gripper(model, data, at_pick, above_pick, close_width, close_width, 0.55, carried_joint=carried_joint)
    move_gripper(model, data, above_pick, above_place, close_width, close_width, 0.70, carried_joint=carried_joint)
    move_gripper(model, data, above_place, target_center, close_width, close_width, 0.45, carried_joint=carried_joint)

    if args.transport_mode == "oracle":
        set_free_body_pose(model, data, pick_joint, target_center, pick_quat, zero_velocity=True)
        mujoco.mj_forward(model, data)

    move_gripper(model, data, target_center, target_center, close_width, open_width, 0.35)
    retreat = target_center + np.array([0.0, -0.18, 0.12])
    move_gripper(model, data, target_center, retreat, open_width, open_width, 0.45)
    step_for(model, data, args.wait_time)

    final_positions = {}
    for stone in stones:
        pos, quat = get_free_body_pose(model, data, f"{stone.name}_free")
        final_positions[stone.name] = {"pos": pos.tolist(), "quat": quat.tolist()}

    pick_final = np.array(final_positions[stones[0].name]["pos"])
    base_final = np.array(final_positions[stones[1].name]["pos"])
    horizontal_error = float(np.linalg.norm(pick_final[:2] - base_final[:2]))
    height_gain = float(pick_final[2] - base_final[2])
    speed = max_body_speed(model, data, all_joints)
    stacked = horizontal_error < 0.09 and height_gain > 0.5 * stones[1].thickness
    stable = stacked and speed < 0.035

    result = {
        "seed": args.seed,
        "transport_mode": args.transport_mode,
        "stable": stable,
        "stacked": stacked,
        "max_final_speed": speed,
        "horizontal_error_m": horizontal_error,
        "height_gain_m": height_gain,
        "moving_stone": {
            "name": stones[0].name,
            "length": stones[0].length,
            "width": stones[0].width,
            "thickness": stones[0].thickness,
            "mass": stones[0].mass,
        },
        "base_stone": {
            "name": stones[1].name,
            "length": stones[1].length,
            "width": stones[1].width,
            "thickness": stones[1].thickness,
            "mass": stones[1].mass,
        },
        "final_positions": final_positions,
    }

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
