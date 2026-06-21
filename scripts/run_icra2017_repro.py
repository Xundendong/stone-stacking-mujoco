#!/usr/bin/env python3
"""MuJoCo reproduction scaffold for the ICRA 2017 stone-stacking paper.

This script is not an explanatory animation. It creates paper-style limestone
meshes, evaluates next-best object/target poses with MuJoCo contacts and the
paper cost terms, commits the selected pose to the simulated stack, and writes
the final MJCF scene for inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.paper_scene import build_icra2017_scene
from stone_stack.paper_stones import make_paper_limestones
from stone_stack.rock_wall_stones import make_rock_wall_stones
from stone_stack.rocks import FlatStone


WEIGHTS = {
    "support_area_inverse": 0.179,
    "kinetic_energy": 0.472,
    "center_distance": 0.094,
    "normal_deviation": 0.255,
}
A_MIN = 1.0e-5
E_KIN_STABLE = 20.0
THETA_INIT = math.pi / 4.0
FORCE_N = 100.0


@dataclass
class StackEntry:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    level: int
    selected_cost: float
    support_area_m2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--stone-seed", type=int, default=17)
    parser.add_argument(
        "--stone-generator",
        choices=("icra-limestone", "from-rocks-to-walls"),
        default="icra-limestone",
        help="Stone mesh source: ICRA limestone surrogate or CVPRW 2021 From Rocks to Walls generator.",
    )
    parser.add_argument(
        "--rock-irregularity",
        type=float,
        default=0.75,
        help="Zeta-like irregularity used by --stone-generator from-rocks-to-walls.",
    )
    parser.add_argument(
        "--rock-subdivisions",
        type=int,
        default=5,
        help="Mesh subdivision rounds used by --stone-generator from-rocks-to-walls.",
    )
    parser.add_argument("--available-stones", type=int, default=4)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--samples-per-stone", type=int, default=5)
    parser.add_argument("--local-iters", type=int, default=1)
    parser.add_argument("--search-time", type=float, default=0.85)
    parser.add_argument("--settle-time", type=float, default=1.4)
    parser.add_argument("--cost-contact-steps", type=int, default=10)
    parser.add_argument("--stone-friction", type=float, default=0.1)
    parser.add_argument("--force-n", type=float, default=FORCE_N)
    parser.add_argument(
        "--allow-best-effort",
        action="store_true",
        help="Commit the lowest-cost pose even if it misses a paper validity constraint.",
    )
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "icra2017_repro.json")
    parser.add_argument("--save-final-xml", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_repro_final.xml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_stones(args: argparse.Namespace) -> list[FlatStone]:
    if args.stone_generator == "icra-limestone":
        return make_paper_limestones(args.stone_seed)
    if args.stone_generator == "from-rocks-to-walls":
        return make_rock_wall_stones(
            seed=args.stone_seed,
            count=6,
            irregularity=args.rock_irregularity,
            subdivisions=args.rock_subdivisions,
        )
    raise ValueError(f"unknown stone generator: {args.stone_generator}")


def import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo is not installed. Run: source .venv/bin/activate") from exc
    return mujoco


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return quat_normalize(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=float,
        )
    )


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / norm
    half = 0.5 * angle
    return quat_normalize(np.array([math.cos(half), *(math.sin(half) * axis)], dtype=float))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def random_orientation(rng: random.Random, theta_init: float) -> np.ndarray:
    axis = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.35, 1.0)])
    angle = rng.uniform(-theta_init, theta_init)
    return axis_angle_quat(axis, angle)


def body_pose_dict(entries: list[StackEntry], extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
    poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for entry in entries:
        poses[entry.name] = (tuple(map(float, entry.pos)), tuple(map(float, entry.quat)))
    for name, (pos, quat) in (extra or {}).items():
        poses[name] = (tuple(map(float, pos)), tuple(map(float, quat_normalize(quat))))
    return poses


def build_model(mujoco, stones, poses, fixed_names, stone_friction, include_robot=False):
    xml = build_icra2017_scene(
        stones,
        body_poses=poses,
        fixed_stones=set(fixed_names),
        include_robot=include_robot,
        stone_friction=stone_friction,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data, xml


def freejoint_addresses(model, joint_name: str) -> tuple[int, int]:
    jid = model.joint(joint_name).id
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def get_body_pose(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = model.body(body_name).id
    return data.xpos[bid].copy(), quat_normalize(data.xquat[bid].copy())


def body_speed(model, data, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    _, qveladr = freejoint_addresses(model, joint_name)
    return data.qvel[qveladr : qveladr + 3].copy(), data.qvel[qveladr + 3 : qveladr + 6].copy()


def contact_points_between(model, data, candidate_name: str, stack_names: list[str]) -> list[np.ndarray]:
    candidate_bid = model.body(candidate_name).id
    stack_bids = {model.body(name).id for name in stack_names}
    points: list[np.ndarray] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if (body1 == candidate_bid and body2 in stack_bids) or (body2 == candidate_bid and body1 in stack_bids):
            points.append(np.array(contact.pos, dtype=float))
    return points


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1.0e-12:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1.0e-12:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area_2d(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def point_in_convex_polygon(point: np.ndarray, poly: np.ndarray, tol: float = 1.0e-6) -> bool:
    if len(poly) < 3:
        return False
    signs = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        signs.append(cross)
    return all(value >= -tol for value in signs) or all(value <= tol for value in signs)


def support_polygon_metrics(points: list[np.ndarray], query_pos: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {
            "area_m2": 0.0,
            "normal": np.array([0.0, 0.0, 1.0]),
            "query_inside": False,
            "contact_count": len(points),
        }
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1]
    normal = vh[2] if vh.shape[0] >= 3 else np.cross(axis_u, axis_v)
    if normal[2] < 0.0:
        normal = -normal
    coords = np.column_stack((centered @ axis_u, centered @ axis_v))
    hull = convex_hull_2d(coords)
    query = np.array([(query_pos - center) @ axis_u, (query_pos - center) @ axis_v], dtype=float)
    return {
        "area_m2": polygon_area_2d(hull),
        "normal": normal,
        "query_inside": point_in_convex_polygon(query, hull),
        "contact_count": len(points),
    }


def kinetic_energy(model, data, active_stones: list[FlatStone]) -> float:
    total = 0.0
    by_name = {stone.name: stone for stone in active_stones}
    for name, stone in by_name.items():
        joint_name = f"{name}_free"
        try:
            linear, angular = body_speed(model, data, joint_name)
        except KeyError:
            continue
        radius = 0.5 * max(stone.length, stone.width, stone.thickness)
        total += 0.5 * stone.mass * float(linear @ linear)
        total += 0.5 * stone.mass * radius * radius * float(angular @ angular)
    return total


def settle(mujoco, model, data, seconds: float):
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)


def make_initial_stack(mujoco, stones: list[FlatStone], stone_friction: float, settle_time: float) -> list[StackEntry]:
    base = max(stones, key=lambda stone: stone.length * stone.width)
    poses = {base.name: ((0.0, 0.0, base.thickness + 0.06), (1.0, 0.0, 0.0, 0.0))}
    model, data, _ = build_model(mujoco, [base], poses, fixed_names=set(), stone_friction=stone_friction)
    mujoco.mj_forward(model, data)
    settle(mujoco, model, data, settle_time)
    pos, quat = get_body_pose(model, data, base.name)
    return [StackEntry(base.name, pos, quat, 0, 0.0, 0.0)]


def evaluate_pose(
    mujoco,
    active_stones: list[FlatStone],
    stack: list[StackEntry],
    candidate: FlatStone,
    initial_pos: np.ndarray,
    initial_quat: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stack_names = [entry.name for entry in stack]
    fixed_poses = body_pose_dict(stack, {candidate.name: (initial_pos, initial_quat)})
    fixed_model, fixed_data, _ = build_model(
        mujoco,
        active_stones,
        fixed_poses,
        fixed_names=set(stack_names),
        stone_friction=args.stone_friction,
    )
    mujoco.mj_forward(fixed_model, fixed_data)

    candidate_bid = fixed_model.body(candidate.name).id
    contact_points: list[np.ndarray] = []
    contact_hold_steps = 0
    search_steps = max(1, int(args.search_time / fixed_model.opt.timestep))
    for _ in range(search_steps):
        fixed_data.xfrc_applied[candidate_bid, :3] = np.array([0.0, 0.0, -args.force_n])
        mujoco.mj_step(fixed_model, fixed_data)
        current_contacts = contact_points_between(fixed_model, fixed_data, candidate.name, stack_names)
        contact_points.extend(current_contacts)
        if len(current_contacts) >= 3:
            contact_hold_steps += 1
            if contact_hold_steps >= 6:
                break
        else:
            contact_hold_steps = 0

    fixed_data.xfrc_applied[:, :] = 0.0
    settle(mujoco, fixed_model, fixed_data, 0.18)
    contact_pos, contact_quat = get_body_pose(fixed_model, fixed_data, candidate.name)

    mobile_poses = body_pose_dict(stack, {candidate.name: (contact_pos, contact_quat)})
    mobile_model, mobile_data, _ = build_model(
        mujoco,
        active_stones,
        mobile_poses,
        fixed_names=set(),
        stone_friction=args.stone_friction,
    )
    mujoco.mj_forward(mobile_model, mobile_data)

    cost_contacts: list[np.ndarray] = []
    for _ in range(args.cost_contact_steps):
        mujoco.mj_step(mobile_model, mobile_data)
        cost_contacts.extend(contact_points_between(mobile_model, mobile_data, candidate.name, stack_names))

    candidate_pos_now, candidate_quat_now = get_body_pose(mobile_model, mobile_data, candidate.name)
    metrics = support_polygon_metrics(cost_contacts, candidate_pos_now)
    energy = kinetic_energy(mobile_model, mobile_data, active_stones)
    support = stack[-1]
    center_distance = float(np.linalg.norm(candidate_pos_now[:2] - support.pos[:2]))
    vertical_axis = np.array([0.0, 0.0, 1.0])
    normal = np.asarray(metrics["normal"], dtype=float)
    normal_deviation = float(1.0 - abs(np.dot(normal, vertical_axis) / max(np.linalg.norm(normal), 1.0e-12)))
    area = float(metrics["area_m2"])
    cost = (
        WEIGHTS["support_area_inverse"] / max(area, A_MIN)
        + WEIGHTS["kinetic_energy"] * energy
        + WEIGHTS["center_distance"] * center_distance
        + WEIGHTS["normal_deviation"] * normal_deviation
    )

    before = {entry.name: entry.pos.copy() for entry in stack}
    settle(mujoco, mobile_model, mobile_data, args.settle_time)
    final_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stone in active_stones:
        final_poses[stone.name] = get_body_pose(mobile_model, mobile_data, stone.name)
    candidate_final = final_poses[candidate.name][0]
    support_final = final_poses[support.name][0]
    previous_drift = max(float(np.linalg.norm(final_poses[name][0] - pos)) for name, pos in before.items())
    height_gain = float(candidate_final[2] - support_final[2])

    valid = bool(
        metrics["contact_count"] >= 3
        and area >= A_MIN
        and metrics["query_inside"]
        and energy <= E_KIN_STABLE
        and height_gain > 0.28 * candidate.thickness
        and previous_drift < 0.080
    )
    if not valid:
        cost += 1.0e4

    return {
        "stone": candidate.name,
        "initial_pos": initial_pos.tolist(),
        "initial_quat": initial_quat.tolist(),
        "contact_pose": {
            "pos": contact_pos.tolist(),
            "quat": contact_quat.tolist(),
        },
        "valid": valid,
        "cost": float(cost),
        "support_area_m2": area,
        "contact_count": int(metrics["contact_count"]),
        "com_projection_inside_support": bool(metrics["query_inside"]),
        "kinetic_energy_j": float(energy),
        "center_distance_m": center_distance,
        "normal_deviation": normal_deviation,
        "height_gain_m": height_gain,
        "previous_stack_drift_m": previous_drift,
        "final_poses": {
            name: {"pos": pose[0].tolist(), "quat": pose[1].tolist()} for name, pose in final_poses.items()
        },
    }


def candidate_initial_pose(
    rng: random.Random,
    stack: list[StackEntry],
    candidate: FlatStone,
    support_stone: FlatStone,
    theta_init: float,
    xy_offset: tuple[float, float] = (0.0, 0.0),
    yaw_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    support = stack[-1]
    support_rot = quat_to_matrix(support.quat)
    normal = support_rot @ np.array([0.0, 0.0, 1.0])
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / max(np.linalg.norm(normal), 1.0e-12)
    tangent_x = support_rot @ np.array([1.0, 0.0, 0.0])
    tangent_x = tangent_x - normal * float(tangent_x @ normal)
    if np.linalg.norm(tangent_x) < 1.0e-9:
        tangent_x = np.array([1.0, 0.0, 0.0])
    tangent_x = tangent_x / np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y = tangent_y / max(np.linalg.norm(tangent_y), 1.0e-12)

    clearance = 0.030
    pos = (
        support.pos
        + tangent_x * xy_offset[0]
        + tangent_y * xy_offset[1]
        + normal * (0.5 * support_stone.thickness + 0.5 * candidate.thickness + clearance)
    )
    yaw_about_support = axis_angle_quat(normal, yaw_offset)
    quat = quat_mul(quat_mul(yaw_about_support, support.quat), random_orientation(rng, theta_init))
    return pos, quat


def local_search_candidate(
    mujoco,
    rng: random.Random,
    active_stones: list[FlatStone],
    stack: list[StackEntry],
    candidate: FlatStone,
    support_stone: FlatStone,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results = []
    base_xy = (rng.uniform(-0.018, 0.018), rng.uniform(-0.018, 0.018))
    base_yaw = rng.uniform(-math.pi, math.pi)
    pos, quat = candidate_initial_pose(rng, stack, candidate, support_stone, THETA_INIT, base_xy, base_yaw)
    best = evaluate_pose(mujoco, active_stones, stack, candidate, pos, quat, args)
    best["local_iteration"] = 0
    best["local_probe"] = "initial"
    results.append(best)

    step_xy = 0.012
    step_yaw = 0.20
    current_xy = np.array(base_xy, dtype=float)
    current_yaw = base_yaw
    for iteration in range(1, args.local_iters + 1):
        probes = [
            ("x+", np.array([step_xy, 0.0]), 0.0),
            ("x-", np.array([-step_xy, 0.0]), 0.0),
            ("y+", np.array([0.0, step_xy]), 0.0),
            ("y-", np.array([0.0, -step_xy]), 0.0),
            ("yaw+", np.array([0.0, 0.0]), step_yaw),
            ("yaw-", np.array([0.0, 0.0]), -step_yaw),
        ]
        improved = False
        for label, delta_xy, delta_yaw in probes:
            pos, quat = candidate_initial_pose(
                rng,
                stack,
                candidate,
                support_stone,
                THETA_INIT * 0.35,
                tuple(current_xy + delta_xy),
                current_yaw + delta_yaw,
            )
            result = evaluate_pose(mujoco, active_stones, stack, candidate, pos, quat, args)
            result["local_iteration"] = iteration
            result["local_probe"] = label
            results.append(result)
            if result["cost"] < best["cost"]:
                best = result
                current_xy = current_xy + delta_xy
                current_yaw = current_yaw + delta_yaw
                improved = True
        if not improved:
            step_xy *= 0.5
            step_yaw *= 0.5
    return results


def choose_subset(stones: list[FlatStone], seed: int, count: int) -> list[FlatStone]:
    if count > len(stones):
        raise ValueError("available-stones cannot exceed the generated stone count")
    offset = seed % len(stones)
    return [stones[(offset + index) % len(stones)] for index in range(count)]


def final_scene_poses(
    all_stones: list[FlatStone],
    stack: list[StackEntry],
    subset: list[FlatStone],
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    poses = body_pose_dict(stack)
    used = {entry.name for entry in stack}
    supply_index = 0
    for stone in all_stones:
        if stone.name in used:
            continue
        x = -0.25 + 0.20 * (supply_index % 3)
        y = 0.38 + 0.15 * (supply_index // 3)
        z = 0.5 * stone.thickness + 0.012
        poses[stone.name] = ((x, y, z), tuple(yaw_quat(0.35 * supply_index)))
        supply_index += 1
    return poses


def to_stack_entry(name: str, pose: dict[str, list[float]], level: int, cost: float, area: float) -> StackEntry:
    return StackEntry(
        name=name,
        pos=np.asarray(pose["pos"], dtype=float),
        quat=quat_normalize(np.asarray(pose["quat"], dtype=float)),
        level=level,
        selected_cost=cost,
        support_area_m2=area,
    )


def main() -> int:
    args = parse_args()
    if args.available_stones < 2:
        raise SystemExit("--available-stones must be at least 2")
    if args.levels < 1:
        raise SystemExit("--levels must be at least 1")

    all_stones = make_stones(args)
    subset = choose_subset(all_stones, args.seed, args.available_stones)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stone_generator": args.stone_generator,
                    "rock_irregularity": args.rock_irregularity,
                    "rock_subdivisions": args.rock_subdivisions,
                    "stones": [
                        {
                            "name": stone.name,
                            "vertices": len(stone.vertices),
                            "faces": len(stone.faces),
                            "length": stone.length,
                            "width": stone.width,
                            "thickness": stone.thickness,
                            "mass": stone.mass,
                        }
                        for stone in all_stones
                    ],
                    "available_subset": [stone.name for stone in subset],
                },
                indent=2,
            )
        )
        return 0

    mujoco = import_mujoco()
    rng = random.Random(args.seed)
    by_name = {stone.name: stone for stone in subset}
    stack = make_initial_stack(mujoco, subset, args.stone_friction, args.settle_time)
    remaining = [stone for stone in subset if stone.name != stack[0].name]
    level_logs: list[dict[str, Any]] = []
    print(f"base={stack[0].name} subset={[stone.name for stone in subset]}")

    for level in range(1, min(args.levels, args.available_stones)):
        support_stone = by_name[stack[-1].name]
        level_candidates: list[dict[str, Any]] = []
        for candidate in list(remaining):
            active = [by_name[entry.name] for entry in stack] + [candidate]
            for _ in range(args.samples_per_stone):
                level_candidates.extend(
                    local_search_candidate(mujoco, rng, active, stack, candidate, support_stone, args)
                )

        stable_candidates = [item for item in level_candidates if item["valid"]]
        if stable_candidates:
            selected = min(stable_candidates, key=lambda item: item["cost"])
        elif args.allow_best_effort and level_candidates:
            selected = min(level_candidates, key=lambda item: item["cost"])
        else:
            level_logs.append(
                {
                    "level": level,
                    "selected": None,
                    "candidate_count": len(level_candidates),
                    "valid_count": len(stable_candidates),
                    "candidates": level_candidates,
                }
            )
            print(f"level={level} no_valid_candidate candidates={len(level_candidates)}")
            break

        # Commit the selected final poses from the MuJoCo cost/settle run.
        updated_stack: list[StackEntry] = []
        for entry in stack:
            updated_stack.append(
                to_stack_entry(
                    entry.name,
                    selected["final_poses"][entry.name],
                    entry.level,
                    entry.selected_cost,
                    entry.support_area_m2,
                )
            )
        new_entry = to_stack_entry(
            selected["stone"],
            selected["final_poses"][selected["stone"]],
            level,
            selected["cost"],
            selected["support_area_m2"],
        )
        updated_stack.append(new_entry)
        stack = updated_stack
        remaining = [stone for stone in remaining if stone.name != selected["stone"]]
        level_logs.append(
            {
                "level": level,
                "selected": {
                    key: value
                    for key, value in selected.items()
                    if key not in {"final_poses"}
                },
                "candidate_count": len(level_candidates),
                "valid_count": len(stable_candidates),
                "candidates": [
                    {key: value for key, value in item.items() if key not in {"final_poses"}}
                    for item in sorted(level_candidates, key=lambda item: item["cost"])[:12]
                ],
            }
        )
        print(
            "level={level} selected={stone} valid={valid} cost={cost:.6g} "
            "area={area:.6g} contacts={contacts} height_gain={height:.4f}".format(
                level=level,
                stone=selected["stone"],
                valid=selected["valid"],
                cost=selected["cost"],
                area=selected["support_area_m2"],
                contacts=selected["contact_count"],
                height=selected["height_gain_m"],
            )
        )

    final_height = max(float(entry.pos[2] + 0.5 * by_name.get(entry.name, all_stones[0]).thickness) for entry in stack)
    final_poses = final_scene_poses(all_stones, stack, subset)
    final_xml = build_icra2017_scene(
        all_stones,
        body_poses=final_poses,
        fixed_stones=set(),
        include_robot=True,
        stone_friction=args.stone_friction,
    )
    args.save_final_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_final_xml.write_text(final_xml, encoding="utf-8")

    result = {
        "paper": {
            "title": "Autonomous Robotic Stone Stacking with Online next Best Object Target Pose Planning",
            "venue": "ICRA 2017",
            "target_hardware": "UR10 arm, Robotiq 3-Finger gripper, FT150 force-torque sensor, Intel RealSense SR300",
            "asset_status": (
                "The original scanned stones are not present locally. This run uses deterministic "
                f"'{args.stone_generator}' mesh stones, a local UR10-scale visual rig, and a Robotiq "
                "three-finger visual gripper. The stacking physics and next-best pose scoring are run "
                "in MuJoCo; arm IK and contact-grasp execution are not yet simulated."
            ),
        },
        "parameters": {
            "seed": args.seed,
            "stone_seed": args.stone_seed,
            "stone_generator": args.stone_generator,
            "rock_irregularity": args.rock_irregularity,
            "rock_subdivisions": args.rock_subdivisions,
            "available_stones": args.available_stones,
            "levels_requested": args.levels,
            "samples_per_stone": args.samples_per_stone,
            "local_iters": args.local_iters,
            "stone_friction_mu": args.stone_friction,
            "weights": WEIGHTS,
            "Amin_m2": A_MIN,
            "force_n": args.force_n,
            "Ekin_stable_j": E_KIN_STABLE,
            "theta_init_rad": THETA_INIT,
        },
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
            for stone in all_stones
        ],
        "available_subset": [stone.name for stone in subset],
        "stack": [
            {
                "level": entry.level,
                "name": entry.name,
                "pos": entry.pos.tolist(),
                "quat": entry.quat.tolist(),
                "selected_cost": entry.selected_cost,
                "support_area_m2": entry.support_area_m2,
            }
            for entry in stack
        ],
        "levels": level_logs,
        "stacked_count": len(stack),
        "final_height_m": final_height,
        "final_xml": str(args.save_final_xml),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stacked_count": len(stack), "final_height_m": final_height, "final_xml": str(args.save_final_xml)}, indent=2))
    return 0 if len(stack) >= min(args.levels, args.available_stones) else 1


if __name__ == "__main__":
    raise SystemExit(main())
