#!/usr/bin/env python3
"""Execute the wall plan with official UR5e + Robotiq contact grasps.

This is the first multi-stone execution layer for the current reproduction
path. It uses the hybrid wall planner report for the object order and target
poses, but the transport is performed in MuJoCo by a UR5e arm and Robotiq
2F-140 gripper. Stones are not attached to the gripper; they are lifted and
carried by the Robotiq fingerpad / fingertip contact geoms.
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

from scripts.run_official_ur5e_robotiq_grasp_test import (  # noqa: E402
    ROBOTIQ_140_XML,
    UR5E_XML,
    UR_CTRL_RANGES,
    UR_JOINTS,
    _absolutize_mesh_files,
    apply_clean_ur5e_visual,
    _disable_body_collisions,
    _find_body,
    _fmt,
    _require_asset,
    _retune_gripper_for_stones,
    body_pose,
    drive_segment,
    freejoint_qvel_addr,
    grip_site_from_pad_center,
    set_ur_qpos,
    solve_site_ik,
    step_seconds,
    top_down_gripper_rotation,
)
from stone_stack.rock_wall_stones import make_rock_wall_stones  # noqa: E402
from stone_stack.rocks import FlatStone, flatten_faces, flatten_vertices  # noqa: E402


Q_HOME_ELBOW_UP = np.array([0.74, -1.30, 1.50, -1.76, -1.57, -0.83], dtype=float)
ROBOT_BASE_POS = np.array([-0.30, -0.35, 0.0], dtype=float)
FINGER_CONTACT_GEOMS = (
    "left_fingerpad_collision",
    "right_fingerpad_collision",
    "left_fingertip_collision",
    "right_fingertip_collision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "hybrid_icra_wall_planner.json")
    parser.add_argument("--max-placements", type=int, default=5)
    parser.add_argument(
        "--placement-indices",
        default="",
        help="Comma-separated indices into report['wall']; overrides --max-placements. Example: 0,1,4 builds a small two-bottom-one-top stack.",
    )
    parser.add_argument(
        "--close",
        type=float,
        default=0.32,
        help="Robotiq close command in radians. Lower values reduce sideways squeezing on irregular stones.",
    )
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--approach-height", type=float, default=0.20)
    parser.add_argument(
        "--travel-height",
        type=float,
        default=0.34,
        help="Grip-site z height used for lateral travel before descending to a stone.",
    )
    parser.add_argument(
        "--first-safe-approach",
        action="store_true",
        help="Use a high clearance approach only before the first grasp. Experimental because it can change wall contacts.",
    )
    parser.add_argument("--place-descent-time", type=float, default=0.95)
    parser.add_argument(
        "--upper-place-descent-time",
        type=float,
        default=None,
        help="Optional slower descent duration for course > 0 stones.",
    )
    parser.add_argument("--place-clearance", type=float, default=0.010)
    parser.add_argument(
        "--upper-place-clearance",
        type=float,
        default=-0.045,
        help="Optional place clearance for course > 0 stones; keeps bottom-course placement unchanged.",
    )
    parser.add_argument(
        "--contact-aware-place",
        action="store_true",
        help="For upper-course stones, stop the placement descent after sustained contact with already placed stones.",
    )
    parser.add_argument(
        "--place-contact-hold",
        type=float,
        default=0.040,
        help="Seconds of sustained support contact required before contact-aware placement stops descending.",
    )
    parser.add_argument("--settle-time", type=float, default=0.80)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--robot-visual",
        choices=("clean", "robosuite"),
        default="clean",
        help="Robot visual style. clean hides robot collision geoms while preserving robosuite assembled visuals and dynamics.",
    )
    parser.add_argument(
        "--no-reset-supply-before-pick",
        action="store_true",
        help="Do not move the next unplaced stone back to its feed pose before grasping.",
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "official_ur5e_robotiq_wall_stack.xml",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "official_ur5e_robotiq_wall_stack.json",
    )
    return parser.parse_args()


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = map(float, q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


def report_stones(report: dict) -> list[FlatStone]:
    params = report["parameters"]
    return make_rock_wall_stones(
        seed=int(params["stone_seed"]),
        count=int(params["stones"]),
        irregularity=float(params["rock_irregularity"]),
        subdivisions=int(params["rock_subdivisions"]),
    )


def parse_index_list(text: str) -> list[int]:
    if not text.strip():
        return []
    indices = [int(part.strip()) for part in text.split(",") if part.strip()]
    if any(index < 0 for index in indices):
        raise ValueError("--placement-indices must be non-negative")
    return indices


def selected_wall_entries(report: dict, args: argparse.Namespace) -> list[dict]:
    indices = parse_index_list(args.placement_indices)
    if indices:
        wall = report["wall"]
        if max(indices) >= len(wall):
            raise ValueError(f"--placement-indices contains an index beyond report wall length {len(wall)}")
        return [wall[index] for index in indices]

    max_placements = args.max_placements
    if max_placements <= 0:
        raise ValueError("--max-placements must be positive")
    return report["wall"][: min(max_placements, len(report["wall"]))]


def supply_pose(index: int, entry: dict, stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
    # Keep the supply area reachable and space stones far enough that they do
    # not knock each other while settling.
    if int(entry["course"]) > 0:
        upper_positions = [(0.18, 0.18), (-0.18, 0.18), (0.18, 0.30), (-0.18, 0.30)]
        x, y = upper_positions[int(entry["slot_index"]) % len(upper_positions)]
    else:
        bottom_positions = [(-0.36, 0.36), (0.00, 0.36), (0.30, 0.30), (-0.16, 0.48)]
        x, y = bottom_positions[int(entry["slot_index"]) % len(bottom_positions)]
    vertices = np.asarray(stone.vertices, dtype=float)
    z = -float(vertices[:, 2].min()) + 0.004
    target_yaw = yaw_from_quat(np.asarray(entry["quat"], dtype=float))
    pos = np.array([x, y, z], dtype=float)
    return pos, yaw_quat(target_yaw)


def initial_supply_pose(index: int, entry: dict, stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
    # The executable planner resets each stone to its exact reachable supply
    # pose immediately before grasping. Initial poses only need to keep all
    # queued stones visible and non-overlapping so the scene does not explode
    # before the first grasp.
    positions = [
        (-0.36, 0.36),
        (0.00, 0.36),
        (0.30, 0.30),
        (0.18, 0.18),
        (-0.18, 0.18),
        (0.54, 0.56),
        (-0.36, 0.18),
        (0.44, 0.34),
    ]
    x, y = positions[index % len(positions)]
    vertices = np.asarray(stone.vertices, dtype=float)
    z = -float(vertices[:, 2].min()) + 0.004
    target_yaw = yaw_from_quat(np.asarray(entry["quat"], dtype=float))
    return np.array([x, y, z], dtype=float), yaw_quat(target_yaw)


def stone_mesh_asset(stone: FlatStone) -> ET.Element:
    return ET.Element(
        "mesh",
        {
            "name": f"{stone.name}_mesh",
            "vertex": flatten_vertices(stone.vertices),
            "face": flatten_faces(stone.faces),
        },
    )


def stone_body(stone: FlatStone, pos: np.ndarray, quat: np.ndarray) -> ET.Element:
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


def build_wall_stack_scene(
    stones: list[FlatStone],
    initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    robot_visual: str = "clean",
) -> str:
    _require_asset(UR5E_XML)
    _require_asset(ROBOTIQ_140_XML)

    robot_root = ET.parse(UR5E_XML).getroot()
    gripper_root = ET.parse(ROBOTIQ_140_XML).getroot()
    _absolutize_mesh_files(robot_root, UR5E_XML.parent)
    _absolutize_mesh_files(gripper_root, ROBOTIQ_140_XML.parent)
    if robot_visual == "clean":
        apply_clean_ur5e_visual(robot_root, UR5E_XML.parent)

    root = ET.Element("mujoco", {"model": "official_ur5e_robotiq_wall_stack"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.0015",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "gravity": "0 0 -9.81",
            "iterations": "170",
        },
    )
    ET.SubElement(root, "size", {"nconmax": "3200", "njmax": "6400"})
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
    for stone in stones:
        asset.append(stone_mesh_asset(stone))

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
        {"name": "overview", "pos": "0.55 -1.02 0.62", "xyaxes": "0.88 0.48 0 -0.24 0.44 0.86"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table",
            "type": "box",
            "pos": "0 0 -0.025",
            "size": "0.90 0.75 0.025",
            "material": "table_mat",
            "friction": "1.0 0.02 0.001",
            "condim": "4",
        },
    )

    robot_body = deepcopy(_find_body(robot_root.find("worldbody"), "base"))
    robot_body.set("pos", _fmt(ROBOT_BASE_POS))
    _disable_body_collisions(robot_body)
    right_hand = _find_body(robot_body, "right_hand")
    gripper_body = deepcopy(_find_body(gripper_root.find("worldbody"), "right_gripper"))
    _retune_gripper_for_stones(gripper_body)
    right_hand.append(gripper_body)
    worldbody.append(robot_body)

    for stone in stones:
        pos, quat = initial_poses[stone.name]
        worldbody.append(stone_body(stone, pos, quat))

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


def prepare(args: argparse.Namespace):
    import mujoco

    report = json.loads(args.report.read_text(encoding="utf-8"))
    entries = selected_wall_entries(report, args)
    by_name = {stone.name: stone for stone in report_stones(report)}
    stones = [by_name[entry["name"]] for entry in entries]
    initial_poses = {
        entry["name"]: initial_supply_pose(index, entry, by_name[entry["name"]])
        for index, entry in enumerate(entries)
    }
    xml = build_wall_stack_scene(stones, initial_poses, robot_visual=args.robot_visual)
    args.save_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_xml.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    return mujoco, report, entries, stones, model, data, ik_data


def set_stone_contact(model, stone_name: str, enabled: bool) -> None:
    geom_id = model.geom(f"{stone_name}_geom").id
    model.geom_contype[geom_id] = 1 if enabled else 0
    model.geom_conaffinity[geom_id] = 1 if enabled else 0


def set_gripper_contact(model, enabled: bool) -> None:
    contype = 1 if enabled else 0
    conaffinity = 1 if enabled else 0
    for name in FINGER_CONTACT_GEOMS:
        geom_id = model.geom(name).id
        model.geom_contype[geom_id] = contype
        model.geom_conaffinity[geom_id] = conaffinity


def support_contact_count(model, data, stone_name: str, support_names: list[str], include_table: bool) -> int:
    stone_bid = model.body(stone_name).id
    support_bids = {model.body(name).id for name in support_names}
    count = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 != stone_bid and body2 != stone_bid:
            continue
        other_body = body2 if body1 == stone_bid else body1
        if other_body in support_bids or (include_table and other_body == 0):
            count += 1
    return count


def drive_place_descent(
    args: argparse.Namespace,
    mujoco,
    model,
    data,
    ik_data,
    stone_name: str,
    support_names: list[str],
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_q: np.ndarray,
    target_rot: np.ndarray,
    gripper_ctrl: tuple[float, float],
    seconds: float,
    viewer=None,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    steps = max(1, int(seconds / model.opt.timestep))
    q = np.asarray(start_q, dtype=float).copy()
    sleep_dt = model.opt.timestep / max(args.speed, 0.05)
    hold_steps = max(1, int(args.place_contact_hold / model.opt.timestep))
    contact_hold = 0
    max_contacts = 0
    release_site = np.asarray(start_pos, dtype=float).copy()

    for step in range(steps):
        t = (step + 1) / steps
        smooth = t * t * (3.0 - 2.0 * t)
        release_site = (1.0 - smooth) * start_pos + smooth * end_pos
        q = solve_site_ik(mujoco, model, ik_data, "grip_site", release_site, target_rot, q, iterations=45)
        data.ctrl[:6] = q
        data.ctrl[6] = gripper_ctrl[0]
        data.ctrl[7] = gripper_ctrl[1]
        mujoco.mj_step(model, data)
        contacts = support_contact_count(
            model,
            data,
            stone_name,
            support_names,
            include_table=False,
        )
        max_contacts = max(max_contacts, contacts)
        if contacts > 0:
            contact_hold += 1
            if contact_hold >= hold_steps:
                if viewer is not None:
                    viewer.sync()
                    time.sleep(sleep_dt)
                return q, release_site, True, max_contacts
        else:
            contact_hold = 0
        if viewer is not None:
            viewer.sync()
            time.sleep(sleep_dt)

    return q, np.asarray(end_pos, dtype=float).copy(), False, max_contacts


def reset_stone_to_supply(mujoco, model, data, entry: dict, stone: FlatStone, seconds: float = 0.20) -> np.ndarray:
    pos, quat = supply_pose(0, entry, stone)
    joint_id = model.joint(f"{stone.name}_free").id
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    step_seconds(mujoco, model, data, seconds)
    return body_pose(model, data, stone.name)[0]


def settle_initial_scene(mujoco, model, data, q: np.ndarray, seconds: float, viewer, speed: float) -> None:
    set_ur_qpos(model, data, q)
    data.ctrl[:6] = q
    data.ctrl[6:] = (0.0, 0.0)
    mujoco.mj_forward(model, data)
    step_seconds(mujoco, model, data, seconds, viewer, speed)


def drive_to_pick_approach(
    args: argparse.Namespace,
    mujoco,
    model,
    data,
    ik_data,
    q: np.ndarray,
    above_pick: np.ndarray,
    target_rot: np.ndarray,
    gripper_ctrl: tuple[float, float],
    viewer=None,
) -> np.ndarray:
    site_id = model.site("grip_site").id
    mujoco.mj_forward(model, data)
    current_site = data.site_xpos[site_id].copy()
    travel_z = max(float(args.travel_height), float(current_site[2]), float(above_pick[2]))

    current_high = current_site.copy()
    current_high[2] = travel_z
    above_pick_high = np.asarray(above_pick, dtype=float).copy()
    above_pick_high[2] = travel_z

    if abs(current_high[2] - current_site[2]) > 1.0e-3:
        q = drive_segment(
            mujoco,
            model,
            data,
            ik_data,
            current_site,
            current_high,
            q,
            target_rot,
            gripper_ctrl,
            0.55,
            viewer,
            args.speed,
        )
    q = drive_segment(
        mujoco,
        model,
        data,
        ik_data,
        current_high,
        above_pick_high,
        q,
        target_rot,
        gripper_ctrl,
        0.85,
        viewer,
        args.speed,
    )
    if np.linalg.norm(above_pick_high - above_pick) > 1.0e-3:
        q = drive_segment(
            mujoco,
            model,
            data,
            ik_data,
            above_pick_high,
            above_pick,
            q,
            target_rot,
            gripper_ctrl,
            0.45,
            viewer,
            args.speed,
        )
    return q


def execute_entry(
    args: argparse.Namespace,
    mujoco,
    model,
    data,
    ik_data,
    entry: dict,
    stone: FlatStone,
    placed_names: list[str],
    q: np.ndarray,
    viewer=None,
) -> tuple[np.ndarray, dict]:
    target_pos = np.asarray(entry["pos"], dtype=float)
    target_quat = np.asarray(entry["quat"], dtype=float)
    course = int(entry["course"])
    place_clearance = (
        args.place_clearance
        if course == 0 or args.upper_place_clearance is None
        else args.upper_place_clearance
    )
    place_descent_time = (
        args.place_descent_time
        if course == 0 or args.upper_place_descent_time is None
        else args.upper_place_descent_time
    )
    target_yaw = yaw_from_quat(target_quat)
    grasp_yaw = target_yaw + math.pi / 2.0
    target_rot = top_down_gripper_rotation(grasp_yaw)
    open_ctrl = (0.0, 0.0)
    close_value = float(np.clip(args.close, 0.0, 0.7))
    close_ctrl = (close_value, -close_value)
    set_gripper_contact(model, True)
    set_stone_contact(model, stone.name, True)

    if not args.no_reset_supply_before_pick:
        reset_stone_to_supply(mujoco, model, data, entry, stone)

    pick_pos, _ = body_pose(model, data, stone.name)
    pick_site = grip_site_from_pad_center(pick_pos, target_rot)
    above_pick = pick_site + np.array([0.0, 0.0, args.approach_height])
    lift_site = pick_site + np.array([0.0, 0.0, args.lift_height])

    if args.first_safe_approach and not placed_names:
        q = drive_to_pick_approach(
            args,
            mujoco,
            model,
            data,
            ik_data,
            q,
            above_pick,
            target_rot,
            open_ctrl,
            viewer,
        )
        step_seconds(mujoco, model, data, 0.15, viewer, args.speed)
    else:
        q = solve_site_ik(mujoco, model, ik_data, "grip_site", above_pick, target_rot, q, iterations=260)
        data.ctrl[:6] = q
        data.ctrl[6:] = open_ctrl
        step_seconds(mujoco, model, data, 0.25, viewer, args.speed)

    q = drive_segment(
        mujoco, model, data, ik_data, above_pick, pick_site, q, target_rot, open_ctrl, 0.85, viewer, args.speed
    )
    q = drive_segment(
        mujoco, model, data, ik_data, pick_site, pick_site, q, target_rot, close_ctrl, 1.10, viewer, args.speed
    )
    closed_pos, _ = body_pose(model, data, stone.name)
    step_seconds(mujoco, model, data, 0.25, viewer, args.speed)

    q = drive_segment(
        mujoco, model, data, ik_data, pick_site, lift_site, q, target_rot, close_ctrl, 1.15, viewer, args.speed
    )
    step_seconds(mujoco, model, data, 0.35, viewer, args.speed)
    lifted_pos, _ = body_pose(model, data, stone.name)
    lifted_site = data.site_xpos[model.site("grip_site").id].copy()
    carry_offset = lifted_pos - lifted_site

    place_center = target_pos + np.array([0.0, 0.0, place_clearance])
    place_site = place_center - carry_offset
    above_place = place_site + np.array([0.0, 0.0, args.approach_height])

    q = drive_segment(
        mujoco, model, data, ik_data, lift_site, above_place, q, target_rot, close_ctrl, 1.25, viewer, args.speed
    )
    contact_place_stopped = False
    place_support_contact_count = 0
    release_site = place_site
    if args.contact_aware_place and course > 0 and placed_names:
        q, release_site, contact_place_stopped, place_support_contact_count = drive_place_descent(
            args,
            mujoco,
            model,
            data,
            ik_data,
            stone.name,
            placed_names,
            above_place,
            place_site,
            q,
            target_rot,
            close_ctrl,
            place_descent_time,
            viewer,
        )
    else:
        q = drive_segment(
            mujoco,
            model,
            data,
            ik_data,
            above_place,
            place_site,
            q,
            target_rot,
            close_ctrl,
            place_descent_time,
            viewer,
            args.speed,
        )
    pre_release_pos, _ = body_pose(model, data, stone.name)

    q = drive_segment(
        mujoco, model, data, ik_data, release_site, release_site, q, target_rot, open_ctrl, 0.85, viewer, args.speed
    )
    step_seconds(mujoco, model, data, args.settle_time, viewer, args.speed)
    settled_pos, settled_quat = body_pose(model, data, stone.name)
    set_gripper_contact(model, False)

    slide_direction = target_rot[:, 1].copy()
    slide_direction[2] = 0.0
    slide_norm = float(np.linalg.norm(slide_direction))
    if slide_norm < 1.0e-9:
        slide_direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        slide_direction = slide_direction / slide_norm
    slide_site = release_site + 0.14 * slide_direction
    above_slide = slide_site + np.array([0.0, 0.0, args.approach_height])

    q = drive_segment(
        mujoco, model, data, ik_data, release_site, slide_site, q, target_rot, open_ctrl, 0.55, viewer, args.speed
    )
    q = drive_segment(
        mujoco, model, data, ik_data, slide_site, above_slide, q, target_rot, open_ctrl, 0.55, viewer, args.speed
    )
    step_seconds(mujoco, model, data, 0.15, viewer, args.speed)
    final_pos, final_quat = body_pose(model, data, stone.name)
    set_gripper_contact(model, True)

    qvel_addr = freejoint_qvel_addr(model, f"{stone.name}_free")
    final_speed = float(np.linalg.norm(data.qvel[qvel_addr : qvel_addr + 3]))
    lift_gain = float(lifted_pos[2] - pick_pos[2])
    target_xy_error = float(np.linalg.norm(final_pos[:2] - target_pos[:2]))
    target_z_error = float(abs(final_pos[2] - target_pos[2]))
    placed = bool(lift_gain > 0.045 and target_xy_error < 0.13 and target_z_error < 0.10)
    stacked = bool(entry["course"] == 0 or final_pos[2] > 0.075)

    result = {
        "name": stone.name,
        "course": int(entry["course"]),
        "slot_index": int(entry["slot_index"]),
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
        "place_clearance_m": float(place_clearance),
        "place_descent_time_s": float(place_descent_time),
        "release_site": release_site.tolist(),
        "contact_aware_place": bool(args.contact_aware_place and course > 0 and placed_names),
        "contact_place_stopped": contact_place_stopped,
        "place_support_contact_count": int(place_support_contact_count),
        "pick_pos": pick_pos.tolist(),
        "closed_pos": closed_pos.tolist(),
        "lifted_pos": lifted_pos.tolist(),
        "carry_offset": carry_offset.tolist(),
        "pre_release_pos": pre_release_pos.tolist(),
        "settled_pos": settled_pos.tolist(),
        "settled_quat": settled_quat.tolist(),
        "final_pos": final_pos.tolist(),
        "final_quat": final_quat.tolist(),
        "lift_gain_m": lift_gain,
        "target_xy_error_m": target_xy_error,
        "target_z_error_m": target_z_error,
        "final_linear_speed_m_s": final_speed,
        "placed": placed,
        "stacked": stacked,
    }
    return q, result


def run_execution(args: argparse.Namespace, mujoco, entries, stones, model, data, ik_data, viewer=None) -> dict:
    q = Q_HOME_ELBOW_UP.copy()
    settle_initial_scene(mujoco, model, data, q, 0.65, viewer, args.speed)

    step_results: list[dict] = []
    placed_names: list[str] = []
    for index, (entry, stone) in enumerate(zip(entries, stones), start=1):
        q, result = execute_entry(args, mujoco, model, data, ik_data, entry, stone, placed_names, q, viewer)
        step_results.append(result)
        if result["placed"]:
            placed_names.append(stone.name)
        print(
            "placement={idx} stone={name} course={course} lifted={lift:.3f} "
            "xy_error={xy:.3f} z_error={z:.3f} placed={placed}".format(
                idx=index,
                name=result["name"],
                course=result["course"],
                lift=result["lift_gain_m"],
                xy=result["target_xy_error_m"],
                z=result["target_z_error_m"],
                placed=result["placed"],
            ),
            flush=True,
        )
        if viewer is not None and not viewer.is_running():
            break

    final_height = 0.0
    for result in step_results:
        final_height = max(final_height, float(result["final_pos"][2]))
    summary = {
        "success": bool(step_results and all(result["placed"] for result in step_results)),
        "requested_placements": len(entries),
        "executed_placements": len(step_results),
        "placed_count": sum(1 for result in step_results if result["placed"]),
        "stacked_count": sum(1 for result in step_results if result["stacked"]),
        "final_center_height_m": final_height,
        "official_models": {
            "robot": "robosuite UR5e MJCF",
            "gripper": "robosuite Robotiq 2F-140 MJCF",
            "ur_xml": str(UR5E_XML),
            "robotiq_xml": str(ROBOTIQ_140_XML),
            "robot_visual": args.robot_visual,
            "visual_note": (
                "clean visual keeps robosuite assembled UR5e visual geoms, hides robot collision geoms, and keeps robosuite joints, inertials and actuators"
                if args.robot_visual == "clean"
                else "robosuite original robot visuals"
            ),
        },
        "control": {
            "ur": "position targets from MuJoCo Jacobian IK, seeded on elbow-up branch",
            "gripper": "Robotiq joint position actuators; stones are not welded/attached",
            "close_command": [float(np.clip(args.close, 0.0, 0.7)), -float(np.clip(args.close, 0.0, 0.7))],
            "retreat_collision_filter": "fingerpad/tip collision is disabled only after release settling, then re-enabled before the next grasp",
        },
        "steps": step_results,
        "xml": str(args.save_xml),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    mujoco, report, entries, stones, model, data, ik_data = prepare(args)
    if args.check_only:
        print(
            json.dumps(
                {
                    "model_nbody": model.nbody,
                    "model_njnt": model.njnt,
                    "model_nu": model.nu,
                    "model_ngeom": model.ngeom,
                    "model_nmesh": model.nmesh,
                    "requested_placements": len(entries),
                    "stones": [stone.name for stone in stones],
                    "xml": str(args.save_xml),
                    "planner_report": str(args.report),
                    "planner_final_height_m": report.get("final_height_m"),
                },
                indent=2,
            )
        )
        return 0

    if args.view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            summary = run_execution(args, mujoco, entries, stones, model, data, ik_data, viewer)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
        print(json.dumps(summary, indent=2))
        return 0 if summary["placed_count"] > 0 else 1

    summary = run_execution(args, mujoco, entries, stones, model, data, ik_data)
    print(json.dumps(summary, indent=2))
    return 0 if summary["placed_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
