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
import trimesh

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
DEFAULT_STONE_VISUAL_ROUGHNESS = 0.0025
DEFAULT_STONE_VISUAL_SUBDIVISIONS = 2
DEFAULT_STONE_VISUAL_STYLE = "paper"
PAPER_STONE_PALETTE = (
    (0.92, 0.78, 0.18),
    (0.48, 0.34, 0.58),
    (0.58, 0.61, 0.33),
    (0.70, 0.58, 0.34),
    (0.38, 0.48, 0.60),
    (0.68, 0.42, 0.34),
    (0.55, 0.52, 0.43),
    (0.82, 0.70, 0.30),
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
    parser.add_argument(
        "--grasp-retries",
        type=int,
        default=1,
        help="Retry a placement this many times with stronger gripper closure when the stone was not lifted.",
    )
    parser.add_argument(
        "--grasp-retry-close-step",
        type=float,
        default=0.06,
        help="Additional close command applied on each grasp retry.",
    )
    parser.add_argument(
        "--grasp-retry-yaw-step",
        type=float,
        default=math.pi / 2.0,
        help="Additional gripper yaw offset applied on each grasp retry.",
    )
    parser.add_argument(
        "--grasp-yaw-overrides",
        default="",
        help="Comma-separated one-based placement yaw offsets in radians, e.g. 5:1.5708.",
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
        "--align-upper-place-orientation",
        action="store_true",
        help="For upper-course stones, rotate the gripper after lift so the carried stone matches the planner target orientation.",
    )
    parser.add_argument(
        "--place-contact-hold",
        type=float,
        default=0.040,
        help="Seconds of sustained support contact required before contact-aware placement stops descending.",
    )
    parser.add_argument(
        "--place-contact-min-contacts",
        type=int,
        default=2,
        help="Minimum simultaneous support contacts required before contact-aware placement stops descending.",
    )
    parser.add_argument(
        "--place-contact-settle-depth",
        type=float,
        default=0.0,
        help="Extra downward travel after sustained support contact, in meters; useful for lightly seating rough stones.",
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
        "--stone-visual-roughness",
        type=float,
        default=DEFAULT_STONE_VISUAL_ROUGHNESS,
        help="Visual-only stone surface relief amplitude in meters. Set 0 to show the raw collision mesh.",
    )
    parser.add_argument(
        "--stone-visual-subdivisions",
        type=int,
        default=DEFAULT_STONE_VISUAL_SUBDIVISIONS,
        help="Visual-only mesh subdivision rounds before applying surface relief.",
    )
    parser.add_argument(
        "--stone-visual-style",
        choices=("paper", "natural"),
        default=DEFAULT_STONE_VISUAL_STYLE,
        help="Visual-only stone material style. paper matches the smooth colored stones used in the reference figures.",
    )
    parser.add_argument(
        "--no-reset-supply-before-pick",
        action="store_true",
        help="Do not move the next unplaced stone back to its feed pose before grasping.",
    )
    parser.add_argument(
        "--online-support-correction",
        action="store_true",
        default=False,
        help="Enable experimental XY target correction from measured lower-course support drift.",
    )
    parser.add_argument(
        "--no-online-support-correction",
        dest="online_support_correction",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--support-correction-gain",
        type=float,
        default=0.85,
        help="Gain for online XY target correction from lower-course support drift.",
    )
    parser.add_argument(
        "--max-support-correction",
        type=float,
        default=0.080,
        help="Clamp online XY support correction to this norm in meters.",
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


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return np.eye(3, dtype=float)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def report_stones(report: dict) -> list[FlatStone]:
    params = report["parameters"]
    return make_rock_wall_stones(
        seed=int(params["stone_seed"]),
        count=int(params["stones"]),
        irregularity=float(params["rock_irregularity"]),
        subdivisions=int(params["rock_subdivisions"]),
        style=str(params.get("rock_style", "paper")),
    )


def parse_index_list(text: str) -> list[int]:
    if not text.strip():
        return []
    indices = [int(part.strip()) for part in text.split(",") if part.strip()]
    if any(index < 0 for index in indices):
        raise ValueError("--placement-indices must be non-negative")
    return indices


def parse_float_override_map(text: str) -> dict[int, float]:
    overrides: dict[int, float] = {}
    if not text.strip():
        return overrides
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"override '{item}' must have the form placement:value")
        key_text, value_text = item.split(":", 1)
        key = int(key_text)
        if key <= 0:
            raise ValueError("placement override keys are one-based and must be positive")
        overrides[key] = float(value_text)
    return overrides


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
    # not knock each other while settling. Reuse the rear feed pockets for
    # upper-course stones too; resetting them near the wall makes later grasps
    # interfere with already placed stones.
    if int(entry["course"]) > 0:
        upper_positions = [(-0.36, 0.36), (0.00, 0.36), (0.30, 0.30), (-0.16, 0.48)]
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
        (-0.42, 0.54),
        (0.18, 0.54),
        (0.54, 0.38),
        (-0.54, 0.34),
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


def stable_stone_seed(stone: FlatStone) -> int:
    seed = 0
    for index, char in enumerate(stone.name):
        seed += (index + 1) * ord(char)
    seed += int(round(stone.length * 100_000.0))
    seed += int(round(stone.width * 1_000_000.0))
    seed += int(round(stone.thickness * 10_000_000.0))
    return seed % (2**32)


def stone_visual_rgba(stone: FlatStone, visual_style: str) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(stable_stone_seed(stone) + 19_381)
    if visual_style == "paper":
        palette_index = stable_stone_seed(stone) % len(PAPER_STONE_PALETTE)
        rgb = np.asarray(PAPER_STONE_PALETTE[palette_index], dtype=float)
        rgb = rgb * float(rng.uniform(0.94, 1.06)) + rng.uniform(-0.025, 0.025, size=3)
        return tuple(float(value) for value in np.clip(rgb, 0.18, 0.95)) + (1.0,)

    rgb = np.asarray(stone.rgba[:3], dtype=float)
    rgb = rgb * float(rng.uniform(0.92, 1.06)) + rng.uniform(-0.018, 0.018, size=3)
    return tuple(float(value) for value in np.clip(rgb, 0.18, 0.76)) + (1.0,)


def stone_visual_material_asset(stone: FlatStone, visual_style: str) -> ET.Element:
    if visual_style == "paper":
        specular = "0.14"
        shininess = "0.22"
    else:
        specular = "0.05"
        shininess = "0.12"
    return ET.Element(
        "material",
        {
            "name": f"{stone.name}_rough_mat",
            "rgba": _fmt(stone_visual_rgba(stone, visual_style)),
            "specular": specular,
            "shininess": shininess,
            "reflectance": "0.0",
        },
    )


def stone_visual_mesh_asset(
    stone: FlatStone,
    roughness: float,
    subdivisions: int,
    visual_style: str,
) -> ET.Element:
    roughness = max(0.0, float(roughness))
    subdivisions = max(0, int(subdivisions))
    effective_roughness = roughness
    if visual_style == "paper":
        # The paper/reference figures use smooth-shaded colored stones rather
        # than noisy surface relief. Keep only a light rounded shell.
        effective_roughness = 0.45 * roughness
    mesh = trimesh.Trimesh(
        vertices=np.asarray(stone.vertices, dtype=float),
        faces=np.asarray(stone.faces, dtype=int),
        process=False,
    )
    source_extents = np.maximum(np.ptp(mesh.vertices, axis=0), 1.0e-9)
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    if visual_style == "paper":
        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=0.36,
            iterations=7,
            volume_constraint=False,
        )
        vertices = np.asarray(mesh.vertices, dtype=float)
        center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        smoothed_extents = np.maximum(np.ptp(vertices, axis=0), 1.0e-9)
        scale = (source_extents * 1.012) / smoothed_extents
        mesh.vertices = center + (vertices - center) * scale
    mesh.fix_normals()

    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float).copy()
    normal_lengths = np.linalg.norm(normals, axis=1)
    fallback = vertices / np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1.0e-9)
    normals[normal_lengths < 1.0e-9] = fallback[normal_lengths < 1.0e-9]

    rng = np.random.default_rng(stable_stone_seed(stone) + 47_911)
    extents = np.maximum(np.ptp(vertices, axis=0), 1.0e-9)
    mean_extent = max(float(np.mean(extents)), 1.0e-9)
    directions = rng.normal(size=(5, 3))
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-9)
    if visual_style == "paper":
        weights = np.array([0.55, 0.34, 0.18, 0.0, 0.0], dtype=float)
        freqs = np.array([6.0, 9.0, 13.0, 1.0, 1.0], dtype=float)
        noise_scale = 0.04
    else:
        weights = np.array([0.62, 0.48, 0.36, 0.25, 0.18], dtype=float)
        freqs = np.array([10.0, 15.0, 22.0, 31.0, 43.0], dtype=float)
        noise_scale = 0.22
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(weights))
    relief = np.zeros(len(vertices), dtype=float)
    for direction, weight, freq, phase in zip(directions, weights, freqs, phases):
        relief += weight * np.sin(freq * (vertices @ direction) / mean_extent + phase)
    relief += rng.normal(0.0, noise_scale, size=len(vertices))
    relief -= float(np.mean(relief))
    relief /= max(float(np.max(np.abs(relief))), 1.0e-9)

    base_offset = max(0.00035, 0.62 * effective_roughness)
    displacement = np.clip(
        base_offset + effective_roughness * relief,
        0.00025,
        base_offset + 1.12 * effective_roughness,
    )
    rough_vertices = vertices + normals * displacement[:, None]

    return ET.Element(
        "mesh",
        {
            "name": f"{stone.name}_rough_visual_mesh",
            "vertex": flatten_vertices([tuple(map(float, vertex)) for vertex in rough_vertices]),
            "face": flatten_faces([tuple(map(int, face)) for face in np.asarray(mesh.faces, dtype=int)]),
            "smoothnormal": "true",
        },
    )


def stone_body(
    stone: FlatStone,
    pos: np.ndarray,
    quat: np.ndarray,
    stone_visual_roughness: float = DEFAULT_STONE_VISUAL_ROUGHNESS,
) -> ET.Element:
    body = ET.Element("body", {"name": stone.name, "pos": _fmt(pos), "quat": _fmt(quat)})
    ET.SubElement(body, "freejoint", {"name": f"{stone.name}_free"})
    collision_rgba = stone.rgba
    if stone_visual_roughness > 0.0:
        collision_rgba = (stone.rgba[0], stone.rgba[1], stone.rgba[2], 0.0)
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{stone.name}_geom",
            "type": "mesh",
            "mesh": f"{stone.name}_mesh",
            "mass": f"{stone.mass:.6g}",
            "rgba": _fmt(collision_rgba),
            "friction": "1.15 0.030 0.002",
            "condim": "4",
            "solref": "0.005 1",
            "solimp": "0.92 0.99 0.001",
        },
    )
    if stone_visual_roughness > 0.0:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{stone.name}_rough_visual_geom",
                "type": "mesh",
                "mesh": f"{stone.name}_rough_visual_mesh",
                "material": f"{stone.name}_rough_mat",
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "2",
            },
        )
    return body


def build_wall_stack_scene(
    stones: list[FlatStone],
    initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    robot_visual: str = "clean",
    stone_visual_roughness: float = DEFAULT_STONE_VISUAL_ROUGHNESS,
    stone_visual_subdivisions: int = DEFAULT_STONE_VISUAL_SUBDIVISIONS,
    stone_visual_style: str = DEFAULT_STONE_VISUAL_STYLE,
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
        if stone_visual_roughness > 0.0:
            asset.append(stone_visual_material_asset(stone, stone_visual_style))
            asset.append(
                stone_visual_mesh_asset(
                    stone,
                    stone_visual_roughness,
                    stone_visual_subdivisions,
                    stone_visual_style,
                )
            )

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
        worldbody.append(stone_body(stone, pos, quat, stone_visual_roughness))

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
    xml = build_wall_stack_scene(
        stones,
        initial_poses,
        robot_visual=args.robot_visual,
        stone_visual_roughness=args.stone_visual_roughness,
        stone_visual_subdivisions=args.stone_visual_subdivisions,
        stone_visual_style=args.stone_visual_style,
    )
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
    contact_settle_target_z: float | None = None

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
        if contact_settle_target_z is not None:
            if float(release_site[2]) <= contact_settle_target_z:
                if viewer is not None:
                    viewer.sync()
                    time.sleep(sleep_dt)
                return q, release_site, True, max_contacts
            continue
        if contacts >= args.place_contact_min_contacts:
            contact_hold += 1
            if contact_hold >= hold_steps:
                settle_depth = max(0.0, float(args.place_contact_settle_depth))
                if settle_depth > 0.0:
                    contact_settle_target_z = max(float(end_pos[2]), float(release_site[2]) - settle_depth)
                    continue
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


def corrected_entry_from_supports(
    args: argparse.Namespace,
    entry: dict,
    planned_by_slot: dict[tuple[int, int], dict],
    actual_by_slot: dict[tuple[int, int], np.ndarray],
) -> dict:
    adjusted = dict(entry)
    nominal_pos = np.asarray(entry["pos"], dtype=float)
    adjusted["nominal_pos"] = nominal_pos.copy()
    adjusted["target_correction"] = np.zeros(3, dtype=float)
    if not args.online_support_correction:
        return adjusted

    course = int(entry["course"])
    slot = int(entry["slot_index"])
    if course <= 0:
        return adjusted

    support_deltas: list[np.ndarray] = []
    for lower_slot in (slot, slot + 1):
        planned = planned_by_slot.get((course - 1, lower_slot))
        actual = actual_by_slot.get((course - 1, lower_slot))
        if planned is None or actual is None:
            continue
        planned_pos = np.asarray(planned["pos"], dtype=float)
        support_deltas.append(np.asarray(actual, dtype=float)[:2] - planned_pos[:2])
    if not support_deltas:
        return adjusted

    correction_xy = float(args.support_correction_gain) * np.mean(np.vstack(support_deltas), axis=0)
    norm = float(np.linalg.norm(correction_xy))
    max_norm = max(0.0, float(args.max_support_correction))
    if max_norm > 0.0 and norm > max_norm:
        correction_xy = correction_xy * (max_norm / max(norm, 1.0e-9))

    corrected_pos = nominal_pos.copy()
    corrected_pos[:2] += correction_xy
    correction = np.zeros(3, dtype=float)
    correction[:2] = correction_xy
    adjusted["pos"] = corrected_pos.tolist()
    adjusted["target_correction"] = correction
    return adjusted


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
    grasp_yaw = target_yaw + math.pi / 2.0 + float(entry.get("grasp_yaw_offset", 0.0))
    grasp_rot = top_down_gripper_rotation(grasp_yaw)
    place_rot = grasp_rot
    open_ctrl = (0.0, 0.0)
    close_value = float(np.clip(entry.get("close_override", args.close), 0.0, 0.7))
    close_ctrl = (close_value, -close_value)
    set_gripper_contact(model, True)
    set_stone_contact(model, stone.name, True)

    if not args.no_reset_supply_before_pick:
        reset_stone_to_supply(mujoco, model, data, entry, stone)

    pick_pos, _ = body_pose(model, data, stone.name)
    pick_site = grip_site_from_pad_center(pick_pos, grasp_rot)
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
            grasp_rot,
            open_ctrl,
            viewer,
        )
        step_seconds(mujoco, model, data, 0.15, viewer, args.speed)
    else:
        q = solve_site_ik(mujoco, model, ik_data, "grip_site", above_pick, grasp_rot, q, iterations=260)
        data.ctrl[:6] = q
        data.ctrl[6:] = open_ctrl
        step_seconds(mujoco, model, data, 0.25, viewer, args.speed)

    q = drive_segment(
        mujoco, model, data, ik_data, above_pick, pick_site, q, grasp_rot, open_ctrl, 0.85, viewer, args.speed
    )
    q = drive_segment(
        mujoco, model, data, ik_data, pick_site, pick_site, q, grasp_rot, close_ctrl, 1.10, viewer, args.speed
    )
    closed_pos, _ = body_pose(model, data, stone.name)
    step_seconds(mujoco, model, data, 0.25, viewer, args.speed)

    q = drive_segment(
        mujoco, model, data, ik_data, pick_site, lift_site, q, grasp_rot, close_ctrl, 1.15, viewer, args.speed
    )
    step_seconds(mujoco, model, data, 0.35, viewer, args.speed)
    lifted_pos, lifted_quat = body_pose(model, data, stone.name)
    lifted_site = data.site_xpos[model.site("grip_site").id].copy()
    lifted_site_rot = data.site_xmat[model.site("grip_site").id].reshape(3, 3).copy()
    carry_offset = lifted_pos - lifted_site

    place_center = target_pos + np.array([0.0, 0.0, place_clearance])
    if args.align_upper_place_orientation and course > 0:
        lifted_object_rot = quat_to_mat(lifted_quat)
        object_in_site_rot = lifted_site_rot.T @ lifted_object_rot
        site_to_object_pos = lifted_site_rot.T @ carry_offset
        place_rot = quat_to_mat(target_quat) @ object_in_site_rot.T
        place_site = place_center - place_rot @ site_to_object_pos
        carry_offset = place_center - place_site
    else:
        place_site = place_center - carry_offset
    above_place = place_site + np.array([0.0, 0.0, args.approach_height])

    q = drive_segment(
        mujoco, model, data, ik_data, lift_site, above_place, q, place_rot, close_ctrl, 1.25, viewer, args.speed
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
            place_rot,
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
            place_rot,
            close_ctrl,
            place_descent_time,
            viewer,
            args.speed,
        )
    pre_release_pos, _ = body_pose(model, data, stone.name)

    q = drive_segment(
        mujoco, model, data, ik_data, release_site, release_site, q, place_rot, open_ctrl, 0.85, viewer, args.speed
    )
    set_gripper_contact(model, False)
    step_seconds(mujoco, model, data, args.settle_time, viewer, args.speed)
    settled_pos, settled_quat = body_pose(model, data, stone.name)

    retreat_site = release_site + np.array([0.0, 0.0, args.approach_height], dtype=float)
    q = drive_segment(
        mujoco,
        model,
        data,
        ik_data,
        release_site,
        retreat_site,
        q,
        place_rot,
        open_ctrl,
        0.70,
        viewer,
        args.speed,
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
        "nominal_target_pos": np.asarray(entry.get("nominal_pos", target_pos), dtype=float).tolist(),
        "target_correction_m": np.asarray(
            entry.get("target_correction", np.zeros(3, dtype=float)),
            dtype=float,
        ).tolist(),
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
        "grasp_yaw_offset_rad": float(entry.get("grasp_yaw_offset", 0.0)),
        "place_clearance_m": float(place_clearance),
        "place_descent_time_s": float(place_descent_time),
        "close_command": [close_ctrl[0], close_ctrl[1]],
        "release_site": release_site.tolist(),
        "retreat_mode": "vertical",
        "retreat_site": retreat_site.tolist(),
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
    grasp_yaw_overrides = parse_float_override_map(args.grasp_yaw_overrides)

    step_results: list[dict] = []
    placed_names: list[str] = []
    planned_by_slot = {
        (int(entry["course"]), int(entry["slot_index"])): entry
        for entry in entries
    }
    actual_by_slot: dict[tuple[int, int], np.ndarray] = {}
    for index, (entry, stone) in enumerate(zip(entries, stones), start=1):
        execution_entry = corrected_entry_from_supports(args, entry, planned_by_slot, actual_by_slot)
        attempts: list[dict] = []
        max_attempts = 1 + max(0, int(args.grasp_retries))
        for attempt in range(max_attempts):
            attempt_entry = dict(execution_entry)
            base_yaw_offset = float(execution_entry.get("grasp_yaw_offset", 0.0)) + grasp_yaw_overrides.get(index, 0.0)
            if attempt == 0:
                close_step_count = 0
                yaw_step_count = 0
            else:
                close_step_count = 1 + (attempt - 1) // 2
                yaw_step_count = attempt // 2
            attempt_entry["close_override"] = float(args.close) + close_step_count * float(args.grasp_retry_close_step)
            attempt_entry["grasp_yaw_offset"] = base_yaw_offset + yaw_step_count * float(args.grasp_retry_yaw_step)
            q, result = execute_entry(
                args,
                mujoco,
                model,
                data,
                ik_data,
                attempt_entry,
                stone,
                placed_names,
                q,
                viewer,
            )
            result["attempt"] = attempt + 1
            attempts.append(result)
            if result["lift_gain_m"] >= 0.045 or result["placed"]:
                break
            if viewer is not None and not viewer.is_running():
                break
        result = attempts[-1]
        if len(attempts) > 1:
            result["previous_attempts"] = attempts[:-1]
        step_results.append(result)
        if result["placed"]:
            placed_names.append(stone.name)
            actual_by_slot[(int(result["course"]), int(result["slot_index"]))] = np.asarray(
                result["final_pos"],
                dtype=float,
            )
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
        "stone_visuals": {
            "style": args.stone_visual_style,
            "surface_roughness_m": float(args.stone_visual_roughness),
            "surface_subdivisions": int(args.stone_visual_subdivisions),
            "collision_note": "rough visual meshes are massless and collision-disabled; original convex meshes still provide contact, mass, and friction",
        },
        "control": {
            "ur": "position targets from MuJoCo Jacobian IK, seeded on elbow-up branch",
            "gripper": "Robotiq joint position actuators; stones are not welded/attached",
            "close_command": [float(np.clip(args.close, 0.0, 0.7)), -float(np.clip(args.close, 0.0, 0.7))],
            "retreat_collision_filter": "fingerpad/tip collision is disabled after opening at release, the stone settles independently, and contact is re-enabled before the next grasp",
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
                    "stone_visual_style": args.stone_visual_style,
                    "stone_visual_roughness_m": float(args.stone_visual_roughness),
                    "stone_visual_subdivisions": int(args.stone_visual_subdivisions),
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
