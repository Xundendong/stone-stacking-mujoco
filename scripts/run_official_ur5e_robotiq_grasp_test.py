#!/usr/bin/env python3
"""Official UR + Robotiq contact-grasp test for one generated rock.

This script builds a MuJoCo scene from robosuite's UR5e MJCF and Robotiq 140
MJCF assets, adds one From-Rocks-to-Walls-style rock, and drives the UR joints
with position targets computed by site-position/orientation IK. The rock is
not attached to the gripper; if it lifts, it is lifted by contact and friction.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.rock_wall_stones import make_rock_wall_stones
from stone_stack.rocks import FlatStone, flatten_faces, flatten_vertices

ROBOSUITE_ASSETS = Path(
    "/home/xunden/isaac-sim/kit/python/lib/python3.11/site-packages/robosuite/models/assets"
)
UR5E_XML = ROBOSUITE_ASSETS / "robots" / "ur5e" / "robot.xml"
ROBOTIQ_140_XML = ROBOSUITE_ASSETS / "grippers" / "robotiq_gripper_140.xml"

UR_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR_CTRL_RANGES = {
    "shoulder_pan_joint": "-6.28319 6.28319",
    "shoulder_lift_joint": "-6.28319 6.28319",
    "elbow_joint": "-3.14159 3.14159",
    "wrist_1_joint": "-6.28319 6.28319",
    "wrist_2_joint": "-6.28319 6.28319",
    "wrist_3_joint": "-6.28319 6.28319",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stone-seed", type=int, default=17)
    parser.add_argument("--stone-index", type=int, default=1)
    parser.add_argument("--rock-irregularity", type=float, default=1.0)
    parser.add_argument("--rock-subdivisions", type=int, default=5)
    parser.add_argument("--grasp-yaw", type=float, default=math.pi / 2.0)
    parser.add_argument("--close", type=float, default=0.32, help="Robotiq close command in radians.")
    parser.add_argument("--lift-height", type=float, default=0.16)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--robot-visual",
        choices=("clean", "robosuite"),
        default="clean",
        help="Robot visual style. clean hides robot collision geoms while preserving robosuite dynamics and assembled visuals.",
    )
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "official_ur5e_robotiq_grasp_test.xml",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "official_ur5e_robotiq_grasp_test.json",
    )
    return parser.parse_args()


def _fmt(values) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _require_asset(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing official asset: {path}\n"
            "This script expects robosuite assets from the local Isaac/robosuite install."
        )


def _absolutize_mesh_files(root: ET.Element, base_dir: Path) -> None:
    for element in root.iter():
        file_name = element.get("file")
        if file_name and not Path(file_name).is_absolute():
            element.set("file", str((base_dir / file_name).resolve()))


def _find_body(element: ET.Element, name: str) -> ET.Element:
    if element.tag == "body" and element.get("name") == name:
        return element
    for child in list(element):
        try:
            return _find_body(child, name)
        except KeyError:
            pass
    raise KeyError(name)


def _stone_start_pose(stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(stone.vertices, dtype=float)
    z = -float(vertices[:, 2].min()) + 0.004
    return np.array([-0.20, 0.25, z], dtype=float), np.array([1.0, 0.0, 0.0, 0.0])


def _disable_body_collisions(body: ET.Element) -> None:
    for geom in body.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def _append_material(asset: ET.Element, name: str, rgba: str, specular: str = "0.35", shininess: str = "0.45") -> None:
    for material in asset.findall("material"):
        if material.get("name") == name:
            material.set("rgba", rgba)
            material.set("specular", specular)
            material.set("shininess", shininess)
            return
    ET.SubElement(
        asset,
        "material",
        {
            "name": name,
            "rgba": rgba,
            "specular": specular,
            "shininess": shininess,
        },
    )


def _clean_existing_robot_geoms(robot_body: ET.Element) -> None:
    for geom in robot_body.iter("geom"):
        name = geom.get("name", "")
        is_collision = (
            name.endswith("_col")
            or name.endswith("_col2")
            or "collision" in name
            or geom.get("group") == "0"
        )
        if is_collision:
            geom.set("group", "4")
            geom.set("rgba", "0.55 0.55 0.55 0")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def apply_clean_ur5e_visual(robot_root: ET.Element, _robot_asset_dir: Path) -> None:
    """Use cleaner UR5e visuals while keeping robosuite dynamics intact.

    The robosuite UR5e model has useful inertials, joints and actuator naming,
    but its collision and visual meshes are both visible in the composed demo.
    This keeps the original assembled visual geoms and hides only collision
    geoms, so it cannot break link mesh alignment.
    """

    asset = robot_root.find("asset")
    worldbody = robot_root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("UR5e XML must contain asset and worldbody sections")

    _append_material(asset, "LinkGrey", "0.86 0.88 0.86 1", specular="0.25", shininess="0.36")
    _append_material(asset, "URBlue", "0.04 0.36 0.68 1", specular="0.30", shininess="0.42")
    _append_material(asset, "Black", "0.055 0.060 0.065 1", specular="0.20", shininess="0.32")
    _append_material(asset, "JointGrey", "0.36 0.38 0.38 1", specular="0.22", shininess="0.34")

    robot_body = _find_body(worldbody, "base")
    _clean_existing_robot_geoms(robot_body)


def _retune_gripper_for_stones(gripper_body: ET.Element) -> None:
    for geom in gripper_body.iter("geom"):
        name = geom.get("name", "")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        if "fingerpad_collision" in name or "fingertip_collision" in name:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("friction", "3.6 0.10 0.004")
            geom.set("condim", "4")
            geom.set("solref", "0.004 1")
            geom.set("solimp", "0.96 0.995 0.0005")


def _stone_mesh_asset(stone: FlatStone) -> ET.Element:
    return ET.Element(
        "mesh",
        {
            "name": f"{stone.name}_mesh",
            "vertex": flatten_vertices(stone.vertices),
            "face": flatten_faces(stone.faces),
        },
    )


def _stone_body(stone: FlatStone) -> ET.Element:
    pos, quat = _stone_start_pose(stone)
    body = ET.Element("body", {"name": stone.name, "pos": _fmt(pos), "quat": _fmt(quat)})
    ET.SubElement(body, "freejoint", {"name": f"{stone.name}_free"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{stone.name}_geom",
            "type": "mesh",
            "mesh": f"{stone.name}_mesh",
            "mass": f"{stone.mass:.6g}",
            "rgba": _fmt(stone.rgba),
            "friction": "1.15 0.030 0.002",
            "condim": "4",
            "solref": "0.005 1",
            "solimp": "0.92 0.99 0.001",
        },
    )
    return body


def build_scene_xml(stone: FlatStone, robot_visual: str = "clean") -> str:
    _require_asset(UR5E_XML)
    _require_asset(ROBOTIQ_140_XML)

    robot_root = ET.parse(UR5E_XML).getroot()
    gripper_root = ET.parse(ROBOTIQ_140_XML).getroot()
    _absolutize_mesh_files(robot_root, UR5E_XML.parent)
    _absolutize_mesh_files(gripper_root, ROBOTIQ_140_XML.parent)
    if robot_visual == "clean":
        apply_clean_ur5e_visual(robot_root, UR5E_XML.parent)

    root = ET.Element("mujoco", {"model": "official_ur5e_robotiq_grasp_test"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.0015",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "gravity": "0 0 -9.81",
            "iterations": "150",
        },
    )
    ET.SubElement(root, "size", {"nconmax": "1600", "njmax": "3200"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": "0.30 0.30 0.30", "diffuse": "0.70 0.70 0.68", "specular": "0.12 0.12 0.12"},
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", {"damping": "1.2", "armature": "0.01"})
    ET.SubElement(default, "geom", {"solref": "0.006 1", "solimp": "0.92 0.99 0.001"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "table_grid",
            "type": "2d",
            "builtin": "checker",
            "width": "256",
            "height": "256",
            "rgb1": "0.56 0.56 0.52",
            "rgb2": "0.42 0.42 0.39",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "table_mat", "texture": "table_grid", "texrepeat": "5 4", "reflectance": "0.025"},
    )
    for source_asset in (robot_root.find("asset"), gripper_root.find("asset")):
        if source_asset is None:
            continue
        for child in list(source_asset):
            asset.append(deepcopy(child))
    asset.append(_stone_mesh_asset(stone))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key", "pos": "-0.6 -0.8 1.6", "dir": "0 0 -1", "diffuse": "0.95 0.95 0.90"},
    )
    ET.SubElement(
        worldbody,
        "light",
        {"name": "fill", "pos": "0.6 0.7 1.1", "dir": "0 0 -1", "diffuse": "0.45 0.45 0.42"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        {"name": "overview", "pos": "0.42 -0.90 0.55", "xyaxes": "0.88 0.48 0 -0.22 0.40 0.89"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table",
            "type": "box",
            "pos": "0 0 -0.025",
            "size": "0.85 0.70 0.025",
            "material": "table_mat",
            "friction": "1.0 0.02 0.001",
            "condim": "4",
        },
    )

    robot_body = deepcopy(_find_body(robot_root.find("worldbody"), "base"))
    robot_body.set("pos", "-0.55 -0.25 0")
    _disable_body_collisions(robot_body)
    right_hand = _find_body(robot_body, "right_hand")
    gripper_body = deepcopy(_find_body(gripper_root.find("worldbody"), "right_gripper"))
    _retune_gripper_for_stones(gripper_body)
    right_hand.append(gripper_body)
    worldbody.append(robot_body)
    worldbody.append(_stone_body(stone))

    actuator = ET.SubElement(root, "actuator")
    for joint_name in UR_JOINTS:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"ur_pos_{joint_name}",
                "joint": joint_name,
                "kp": "650",
                "kv": "55",
                "ctrlrange": UR_CTRL_RANGES[joint_name],
                "forcerange": "-280 280",
            },
        )
    gripper_actuator = gripper_root.find("actuator")
    if gripper_actuator is not None:
        for child in list(gripper_actuator):
            copied = deepcopy(child)
            copied.set("kp", "150")
            copied.set("kv", "5")
            copied.set("forcerange", "-220 220")
            actuator.append(copied)

    for tag in ("tendon", "equality", "sensor"):
        source = gripper_root.find(tag)
        if source is not None:
            root.append(deepcopy(source))

    return ET.tostring(root, encoding="unicode")


def top_down_gripper_rotation(yaw: float) -> np.ndarray:
    """Return a top-down Robotiq grip-site orientation.

    In the robosuite Robotiq model the finger opening direction is the
    ``grip_site`` local x-axis, and the gripper body/flange lies along local
    negative z. For a normal table-top grasp the local z-axis must point down,
    so the flange is above the fingers instead of below them.
    """

    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], dtype=float)


def grip_site_from_pad_center(pad_center: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    # Robotiq 140 fingerpad centers are about 32 mm along local -z from grip_site.
    pad_offset_in_site = np.array([0.0, 0.0, -0.032], dtype=float)
    return np.asarray(pad_center, dtype=float) - target_rot @ pad_offset_in_site


def joint_addresses(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos = []
    dof = []
    lower = []
    upper = []
    for name in UR_JOINTS:
        joint = model.joint(name)
        qpos.append(int(model.jnt_qposadr[joint.id]))
        dof.append(int(model.jnt_dofadr[joint.id]))
        lower.append(float(model.jnt_range[joint.id, 0]))
        upper.append(float(model.jnt_range[joint.id, 1]))
    return np.asarray(qpos), np.asarray(dof), np.asarray(lower), np.asarray(upper)


def set_ur_qpos(model, data, q: np.ndarray) -> None:
    qpos_addr, _, _, _ = joint_addresses(model)
    for addr, value in zip(qpos_addr, q):
        data.qpos[addr] = value


def solve_site_ik(
    mujoco,
    model,
    data,
    site_name: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q0: np.ndarray,
    iterations: int = 180,
) -> np.ndarray:
    qpos_addr, dof_addr, lower, upper = joint_addresses(model)
    site_id = model.site(site_name).id
    q = np.asarray(q0, dtype=float).copy()

    for _ in range(iterations):
        for addr, value in zip(qpos_addr, q):
            data.qpos[addr] = value
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id].copy()
        current_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        pos_error = np.asarray(target_pos, dtype=float) - current_pos
        rot_error = 0.5 * (
            np.cross(current_rot[:, 0], target_rot[:, 0])
            + np.cross(current_rot[:, 1], target_rot[:, 1])
            + np.cross(current_rot[:, 2], target_rot[:, 2])
        )
        if np.linalg.norm(pos_error) < 1.0e-4 and np.linalg.norm(rot_error) < 1.0e-3:
            break
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        orientation_weight = 0.35
        jacobian = np.vstack([jac_pos[:, dof_addr], orientation_weight * jac_rot[:, dof_addr]])
        error = np.concatenate([pos_error, orientation_weight * rot_error])
        damping = 1.0e-3
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(6),
            error,
        )
        q = np.clip(q + 0.7 * delta, lower, upper)

    return q


def body_pose(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = model.body(body_name).id
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def freejoint_qvel_addr(model, joint_name: str) -> int:
    joint = model.joint(joint_name)
    return int(model.jnt_dofadr[joint.id])


def step_seconds(mujoco, model, data, seconds: float, viewer=None, speed: float = 1.0) -> None:
    steps = max(1, int(seconds / model.opt.timestep))
    sleep_dt = model.opt.timestep / max(speed, 0.05)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(sleep_dt)


def drive_segment(
    mujoco,
    model,
    data,
    ik_data,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_q: np.ndarray,
    target_rot: np.ndarray,
    gripper_ctrl: tuple[float, float],
    seconds: float,
    viewer=None,
    speed: float = 1.0,
) -> np.ndarray:
    steps = max(1, int(seconds / model.opt.timestep))
    q = np.asarray(start_q, dtype=float).copy()
    sleep_dt = model.opt.timestep / max(speed, 0.05)
    for step in range(steps):
        t = (step + 1) / steps
        smooth = t * t * (3.0 - 2.0 * t)
        target_pos = (1.0 - smooth) * start_pos + smooth * end_pos
        q = solve_site_ik(mujoco, model, ik_data, "grip_site", target_pos, target_rot, q, iterations=45)
        data.ctrl[:6] = q
        data.ctrl[6] = gripper_ctrl[0]
        data.ctrl[7] = gripper_ctrl[1]
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(sleep_dt)
    return q


def prepare(args: argparse.Namespace):
    import mujoco

    stones = make_rock_wall_stones(
        seed=args.stone_seed,
        count=max(args.stone_index + 1, 2),
        irregularity=args.rock_irregularity,
        subdivisions=args.rock_subdivisions,
    )
    stone = stones[args.stone_index]
    xml = build_scene_xml(stone, robot_visual=args.robot_visual)
    args.save_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_xml.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    return mujoco, stone, model, data, ik_data


def run_trial(args: argparse.Namespace, mujoco, stone: FlatStone, model, data, ik_data, viewer=None) -> dict:
    target_rot = top_down_gripper_rotation(args.grasp_yaw)
    rough_stone_start, _ = _stone_start_pose(stone)
    rough_pad_center = rough_stone_start.copy()
    rough_grasp_site = grip_site_from_pad_center(rough_pad_center, target_rot)
    rough_above_site = rough_grasp_site + np.array([0.0, 0.0, 0.20])

    open_ctrl = (0.0, 0.0)
    close_value = float(np.clip(args.close, 0.0, 0.7))
    close_ctrl = (close_value, -close_value)
    # Seed IK on the elbow-up branch. The same end-effector pose also has an
    # elbow-down solution, but that makes the UR arm fold under itself visually.
    q_home = np.array([0.74, -1.30, 1.50, -1.76, -1.57, -0.83], dtype=float)
    q = solve_site_ik(mujoco, model, ik_data, "grip_site", rough_above_site, target_rot, q_home, iterations=300)

    set_ur_qpos(model, data, q)
    data.ctrl[:6] = q
    data.ctrl[6:] = open_ctrl
    mujoco.mj_forward(model, data)
    step_seconds(mujoco, model, data, 0.8, viewer, args.speed)
    initial_stone_pos, _ = body_pose(model, data, stone.name)

    grasp_pad_center = initial_stone_pos.copy()
    grasp_site = grip_site_from_pad_center(grasp_pad_center, target_rot)
    above_site = grasp_site + np.array([0.0, 0.0, 0.18])
    lift_site = grasp_site + np.array([0.0, 0.0, args.lift_height])
    retreat_site = lift_site + np.array([-0.08, -0.08, 0.04])
    q = solve_site_ik(mujoco, model, ik_data, "grip_site", above_site, target_rot, q, iterations=220)
    data.ctrl[:6] = q
    data.ctrl[6:] = open_ctrl
    step_seconds(mujoco, model, data, 0.4, viewer, args.speed)

    q = drive_segment(
        mujoco,
        model,
        data,
        ik_data,
        above_site,
        grasp_site,
        q,
        target_rot,
        open_ctrl,
        1.0,
        viewer,
        args.speed,
    )
    q = drive_segment(
        mujoco,
        model,
        data,
        ik_data,
        grasp_site,
        grasp_site,
        q,
        target_rot,
        close_ctrl,
        1.2,
        viewer,
        args.speed,
    )
    closed_stone_pos, _ = body_pose(model, data, stone.name)
    step_seconds(mujoco, model, data, 0.4, viewer, args.speed)
    q = drive_segment(
        mujoco,
        model,
        data,
        ik_data,
        grasp_site,
        lift_site,
        q,
        target_rot,
        close_ctrl,
        1.3,
        viewer,
        args.speed,
    )
    step_seconds(mujoco, model, data, 0.8, viewer, args.speed)
    lifted_stone_pos, _ = body_pose(model, data, stone.name)
    q = drive_segment(
        mujoco,
        model,
        data,
        ik_data,
        lift_site,
        retreat_site,
        q,
        target_rot,
        close_ctrl,
        0.7,
        viewer,
        args.speed,
    )
    final_stone_pos, _ = body_pose(model, data, stone.name)

    speed = float(np.linalg.norm(data.qvel[freejoint_qvel_addr(model, f"{stone.name}_free") : freejoint_qvel_addr(model, f"{stone.name}_free") + 3]))
    lift_gain = float(lifted_stone_pos[2] - initial_stone_pos[2])
    final_gain = float(final_stone_pos[2] - initial_stone_pos[2])
    xy_error = float(np.linalg.norm(final_stone_pos[:2] - retreat_site[:2]))
    success = bool(lift_gain > 0.07 and xy_error < 0.12)
    result = {
        "success": success,
        "official_models": {
            "robot": "robosuite UR5e MJCF",
            "gripper": "robosuite Robotiq 2F-140 MJCF",
            "ur_xml": str(UR5E_XML),
            "robotiq_xml": str(ROBOTIQ_140_XML),
            "robot_visual": args.robot_visual,
        },
        "stone": {
            "name": stone.name,
            "vertices": len(stone.vertices),
            "faces": len(stone.faces),
            "length_m": stone.length,
            "width_m": stone.width,
            "thickness_m": stone.thickness,
            "mass_kg": stone.mass,
        },
        "control": {
            "ur": "position targets from MuJoCo Jacobian IK",
            "gripper": "Robotiq joint position actuators, no attach/weld to stone",
            "close_command": close_ctrl,
            "grasp_yaw_rad": args.grasp_yaw,
        },
        "initial_stone_pos": initial_stone_pos.tolist(),
        "closed_stone_pos": closed_stone_pos.tolist(),
        "lifted_stone_pos": lifted_stone_pos.tolist(),
        "final_stone_pos": final_stone_pos.tolist(),
        "lift_gain_m": lift_gain,
        "final_gain_m": final_gain,
        "xy_error_m": xy_error,
        "final_linear_speed_m_s": speed,
        "xml": str(args.save_xml),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    mujoco, stone, model, data, ik_data = prepare(args)
    if args.check_only:
        print(
            json.dumps(
                {
                    "model_nbody": model.nbody,
                    "model_njnt": model.njnt,
                    "model_nu": model.nu,
                    "model_ngeom": model.ngeom,
                    "model_nmesh": model.nmesh,
                    "stone": stone.name,
                    "stone_faces": len(stone.faces),
                    "xml": str(args.save_xml),
                },
                indent=2,
            )
        )
        return 0

    if args.view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            result = run_trial(args, mujoco, stone, model, data, ik_data, viewer)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1

    result = run_trial(args, mujoco, stone, model, data, ik_data)
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
