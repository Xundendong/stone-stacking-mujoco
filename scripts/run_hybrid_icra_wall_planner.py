#!/usr/bin/env python3
"""Hybrid reproduction: From-Rocks-to-Walls stones + ICRA pose planner.

This is the current main research path:

* From Rocks to Walls supplies synthetic irregular convex-hull stone meshes.
* ICRA 2017 supplies the truth-state next-best object/target-pose planning
  structure and MuJoCo contact/stability evaluation.
* The target task is dry-stone wall construction, not DQN training.

The script plans a wall course by course. For every empty wall slot it evaluates
each remaining stone at several target-pose samples, simulates contact/settle in
MuJoCo, scores ICRA-style stability terms, and commits the best stable
object-pose pair.
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

from scripts.run_icra2017_repro import (  # noqa: E402
    A_MIN,
    E_KIN_STABLE,
    FORCE_N,
    WEIGHTS,
    axis_angle_quat,
    build_model,
    get_body_pose,
    kinetic_energy,
    quat_mul,
    quat_normalize,
    settle,
    support_polygon_metrics,
    yaw_quat,
)
from stone_stack.paper_scene import build_icra2017_scene  # noqa: E402
from stone_stack.rock_wall_stones import make_rock_wall_stones  # noqa: E402
from stone_stack.rocks import FlatStone  # noqa: E402


@dataclass
class WallSlot:
    course: int
    index: int
    target_xy: np.ndarray


@dataclass
class WallEntry:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    course: int
    slot_index: int
    target_xy: np.ndarray
    selected_cost: float
    support_area_m2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--stone-seed", type=int, default=17)
    parser.add_argument("--stones", type=int, default=12)
    parser.add_argument("--courses", default="4,3,2", help="Comma-separated stones per wall course.")
    parser.add_argument("--wall-width", type=float, default=0.72)
    parser.add_argument("--wall-y", type=float, default=0.0)
    parser.add_argument("--rock-irregularity", type=float, default=1.0)
    parser.add_argument("--rock-subdivisions", type=int, default=5)
    parser.add_argument("--samples-per-stone", type=int, default=8)
    parser.add_argument("--slot-jitter", type=float, default=0.030)
    parser.add_argument("--yaw-jitter", type=float, default=0.22)
    parser.add_argument("--tilt-jitter", type=float, default=0.10)
    parser.add_argument("--search-time", type=float, default=0.55)
    parser.add_argument("--settle-time", type=float, default=0.9)
    parser.add_argument("--cost-contact-steps", type=int, default=12)
    parser.add_argument("--stone-friction", type=float, default=0.60)
    parser.add_argument("--force-n", type=float, default=FORCE_N)
    parser.add_argument("--max-previous-drift", type=float, default=0.090)
    parser.add_argument("--target-tolerance", type=float, default=0.120)
    parser.add_argument("--allow-best-effort", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "hybrid_icra_wall_planner.json")
    parser.add_argument("--save-final-xml", type=Path, default=PROJECT_ROOT / "outputs" / "hybrid_icra_wall_planner_final.xml")
    return parser.parse_args()


def import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo is not installed. Run: source .venv/bin/activate") from exc
    return mujoco


def parse_courses(text: str) -> list[int]:
    courses = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not courses or any(count <= 0 for count in courses):
        raise ValueError("--courses must contain positive integers")
    for before, after in zip(courses, courses[1:]):
        if after > before:
            raise ValueError("upper courses cannot contain more stones than the course below")
    return courses


def contact_points_for_support(
    model,
    data,
    candidate_name: str,
    support_names: list[str],
    include_floor: bool,
) -> list[np.ndarray]:
    candidate_bid = model.body(candidate_name).id
    support_bids = {model.body(name).id for name in support_names}
    points: list[np.ndarray] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        candidate_in_contact = body1 == candidate_bid or body2 == candidate_bid
        if not candidate_in_contact:
            continue
        other_body = body2 if body1 == candidate_bid else body1
        if other_body in support_bids or (include_floor and other_body == 0):
            points.append(np.array(contact.pos, dtype=float))
    return points


def body_pose_dict(entries: list[WallEntry], extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
    poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for entry in entries:
        poses[entry.name] = (tuple(map(float, entry.pos)), tuple(map(float, entry.quat)))
    for name, (pos, quat) in (extra or {}).items():
        poses[name] = (tuple(map(float, pos)), tuple(map(float, quat_normalize(quat))))
    return poses


def stack_top_height(entries: list[WallEntry], by_name: dict[str, FlatStone]) -> float:
    if not entries:
        return 0.0
    return max(float(entry.pos[2] + 0.5 * by_name[entry.name].thickness) for entry in entries)


def local_support_height(entries: list[WallEntry], by_name: dict[str, FlatStone], x: float, radius: float = 0.20) -> float:
    nearby = [
        float(entry.pos[2] + 0.5 * by_name[entry.name].thickness)
        for entry in entries
        if abs(float(entry.pos[0]) - x) <= radius
    ]
    if nearby:
        return max(nearby)
    return stack_top_height(entries, by_name)


def sampled_target_pose(
    rng: random.Random,
    slot: WallSlot,
    candidate: FlatStone,
    placed: list[WallEntry],
    by_name: dict[str, FlatStone],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    x = float(slot.target_xy[0] + rng.uniform(-args.slot_jitter, args.slot_jitter))
    y = float(slot.target_xy[1] + rng.uniform(-0.45 * args.slot_jitter, 0.45 * args.slot_jitter))
    support = local_support_height(placed, by_name, x)
    z = support + candidate.thickness + 0.075
    base_yaw = 0.0 if slot.course % 2 == 0 else math.pi
    yaw = base_yaw + rng.uniform(-args.yaw_jitter, args.yaw_jitter)
    quat = yaw_quat(yaw)
    if args.tilt_jitter > 0.0:
        tilt_axis = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0], dtype=float)
        quat = quat_mul(axis_angle_quat(tilt_axis, rng.uniform(-args.tilt_jitter, args.tilt_jitter)), quat)
    return np.array([x, y, z], dtype=float), quat


def evaluate_wall_pose(
    mujoco,
    active_stones: list[FlatStone],
    placed: list[WallEntry],
    candidate: FlatStone,
    initial_pos: np.ndarray,
    initial_quat: np.ndarray,
    slot: WallSlot,
    args: argparse.Namespace,
) -> dict[str, Any]:
    placed_names = [entry.name for entry in placed]
    include_floor_support = slot.course == 0
    fixed_poses = body_pose_dict(placed, {candidate.name: (initial_pos, initial_quat)})
    fixed_model, fixed_data, _ = build_model(
        mujoco,
        active_stones,
        fixed_poses,
        fixed_names=set(placed_names),
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
        current_contacts = contact_points_for_support(
            fixed_model,
            fixed_data,
            candidate.name,
            placed_names,
            include_floor=include_floor_support,
        )
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

    mobile_poses = body_pose_dict(placed, {candidate.name: (contact_pos, contact_quat)})
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
        cost_contacts.extend(
            contact_points_for_support(
                mobile_model,
                mobile_data,
                candidate.name,
                placed_names,
                include_floor=include_floor_support,
            )
        )

    candidate_pos_now, candidate_quat_now = get_body_pose(mobile_model, mobile_data, candidate.name)
    metrics = support_polygon_metrics(cost_contacts, candidate_pos_now)
    energy = kinetic_energy(mobile_model, mobile_data, active_stones)
    target_distance = float(np.linalg.norm(candidate_pos_now[:2] - slot.target_xy))
    vertical_axis = np.array([0.0, 0.0, 1.0])
    normal = np.asarray(metrics["normal"], dtype=float)
    normal_deviation = float(1.0 - abs(np.dot(normal, vertical_axis) / max(np.linalg.norm(normal), 1.0e-12)))
    area = float(metrics["area_m2"])
    cost = (
        WEIGHTS["support_area_inverse"] / max(area, A_MIN)
        + WEIGHTS["kinetic_energy"] * energy
        + WEIGHTS["center_distance"] * target_distance
        + WEIGHTS["normal_deviation"] * normal_deviation
    )

    before = {entry.name: entry.pos.copy() for entry in placed}
    settle(mujoco, mobile_model, mobile_data, args.settle_time)
    final_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stone in active_stones:
        final_poses[stone.name] = get_body_pose(mobile_model, mobile_data, stone.name)
    candidate_final = final_poses[candidate.name][0]
    previous_drift = 0.0
    if before:
        previous_drift = max(float(np.linalg.norm(final_poses[name][0] - pos)) for name, pos in before.items())

    target_distance_final = float(np.linalg.norm(candidate_final[:2] - slot.target_xy))
    min_course_height = 0.010 if slot.course == 0 else 0.040 + slot.course * 0.030
    valid = bool(
        metrics["contact_count"] >= 3
        and area >= A_MIN
        and metrics["query_inside"]
        and energy <= E_KIN_STABLE
        and previous_drift < args.max_previous_drift
        and target_distance_final <= args.target_tolerance
        and candidate_final[2] >= min_course_height
    )
    if not valid:
        cost += 1.0e4

    return {
        "stone": candidate.name,
        "course": slot.course,
        "slot_index": slot.index,
        "target_xy": slot.target_xy.tolist(),
        "initial_pos": initial_pos.tolist(),
        "initial_quat": initial_quat.tolist(),
        "contact_pose": {"pos": contact_pos.tolist(), "quat": contact_quat.tolist()},
        "valid": valid,
        "cost": float(cost),
        "support_area_m2": area,
        "contact_count": int(metrics["contact_count"]),
        "com_projection_inside_support": bool(metrics["query_inside"]),
        "kinetic_energy_j": float(energy),
        "normal_deviation": normal_deviation,
        "target_distance_m": target_distance_final,
        "previous_wall_drift_m": previous_drift,
        "final_z_m": float(candidate_final[2]),
        "final_poses": {
            name: {"pos": pose[0].tolist(), "quat": pose[1].tolist()} for name, pose in final_poses.items()
        },
    }


def first_course_slots(count: int, wall_width: float, wall_y: float) -> list[WallSlot]:
    spacing = wall_width / count
    x0 = -0.5 * wall_width + 0.5 * spacing
    return [
        WallSlot(0, index, np.array([x0 + index * spacing, wall_y], dtype=float))
        for index in range(count)
    ]


def next_course_slots(course: int, count: int, lower_entries: list[WallEntry], wall_y: float) -> list[WallSlot]:
    lower_sorted = sorted(lower_entries, key=lambda entry: float(entry.pos[0]))
    centers = [0.5 * (lower_sorted[i].pos[0] + lower_sorted[i + 1].pos[0]) for i in range(len(lower_sorted) - 1)]
    if len(centers) < count:
        xmin = min(float(entry.pos[0]) for entry in lower_sorted)
        xmax = max(float(entry.pos[0]) for entry in lower_sorted)
        centers = list(np.linspace(xmin, xmax, count + 2)[1:-1])
    return [
        WallSlot(course, index, np.array([float(centers[index]), wall_y], dtype=float))
        for index in range(count)
    ]


def to_wall_entry(selected: dict[str, Any], slot: WallSlot) -> WallEntry:
    pose = selected["final_poses"][selected["stone"]]
    return WallEntry(
        name=selected["stone"],
        pos=np.asarray(pose["pos"], dtype=float),
        quat=quat_normalize(np.asarray(pose["quat"], dtype=float)),
        course=slot.course,
        slot_index=slot.index,
        target_xy=slot.target_xy.copy(),
        selected_cost=float(selected["cost"]),
        support_area_m2=float(selected["support_area_m2"]),
    )


def update_existing_entries(entries: list[WallEntry], selected: dict[str, Any]) -> list[WallEntry]:
    updated: list[WallEntry] = []
    for entry in entries:
        pose = selected["final_poses"][entry.name]
        updated.append(
            WallEntry(
                name=entry.name,
                pos=np.asarray(pose["pos"], dtype=float),
                quat=quat_normalize(np.asarray(pose["quat"], dtype=float)),
                course=entry.course,
                slot_index=entry.slot_index,
                target_xy=entry.target_xy.copy(),
                selected_cost=entry.selected_cost,
                support_area_m2=entry.support_area_m2,
            )
        )
    return updated


def final_scene_poses(
    all_stones: list[FlatStone],
    wall: list[WallEntry],
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    poses = body_pose_dict(wall)
    used = {entry.name for entry in wall}
    supply_index = 0
    for stone in all_stones:
        if stone.name in used:
            continue
        x = -0.36 + 0.18 * (supply_index % 5)
        y = 0.44 + 0.14 * (supply_index // 5)
        z = 0.5 * stone.thickness + 0.014
        poses[stone.name] = ((x, y, z), tuple(yaw_quat(0.22 * supply_index)))
        supply_index += 1
    return poses


def serialize_scene_state(
    step_index: int,
    label: str,
    all_stones: list[FlatStone],
    wall: list[WallEntry],
) -> dict[str, Any]:
    poses = final_scene_poses(all_stones, wall)
    return {
        "step": step_index,
        "label": label,
        "placed": [
            {
                "course": entry.course,
                "slot_index": entry.slot_index,
                "name": entry.name,
            }
            for entry in wall
        ],
        "poses": {
            name: {"pos": list(map(float, pos)), "quat": list(map(float, quat))}
            for name, (pos, quat) in poses.items()
        },
    }


def main() -> int:
    args = parse_args()
    courses = parse_courses(args.courses)
    if args.stones < sum(courses):
        raise SystemExit("--stones must be at least the total requested wall slots")

    all_stones = make_rock_wall_stones(
        seed=args.stone_seed,
        count=args.stones,
        irregularity=args.rock_irregularity,
        subdivisions=args.rock_subdivisions,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "pipeline": "from-rocks-to-walls-stones + icra2017-next-best-pose-wall-planner",
                    "courses": courses,
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
                },
                indent=2,
            )
        )
        return 0

    mujoco = import_mujoco()
    rng = random.Random(args.seed)
    by_name = {stone.name: stone for stone in all_stones}
    remaining = list(all_stones)
    wall: list[WallEntry] = []
    course_entries: list[list[WallEntry]] = []
    step_logs: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = [
        serialize_scene_state(0, "initial_candidates", all_stones, wall)
    ]

    for course_index, count in enumerate(courses):
        if course_index == 0:
            slots = first_course_slots(count, args.wall_width, args.wall_y)
        else:
            slots = next_course_slots(course_index, count, course_entries[-1], args.wall_y)
        current_course: list[WallEntry] = []

        for slot in slots:
            candidate_results: list[dict[str, Any]] = []
            for candidate in list(remaining):
                active = [by_name[entry.name] for entry in wall] + [candidate]
                for _ in range(args.samples_per_stone):
                    pos, quat = sampled_target_pose(rng, slot, candidate, wall, by_name, args)
                    candidate_results.append(
                        evaluate_wall_pose(mujoco, active, wall, candidate, pos, quat, slot, args)
                    )

            stable = [item for item in candidate_results if item["valid"]]
            if stable:
                selected = min(stable, key=lambda item: item["cost"])
            elif args.allow_best_effort and candidate_results:
                selected = min(candidate_results, key=lambda item: item["cost"])
            else:
                step_logs.append(
                    {
                        "course": slot.course,
                        "slot_index": slot.index,
                        "target_xy": slot.target_xy.tolist(),
                        "selected": None,
                        "candidate_count": len(candidate_results),
                        "valid_count": len(stable),
                        "best_candidates": [
                            {key: value for key, value in item.items() if key != "final_poses"}
                            for item in sorted(candidate_results, key=lambda item: item["cost"])[:10]
                        ],
                    }
                )
                print(f"course={slot.course} slot={slot.index} no_valid_candidate candidates={len(candidate_results)}")
                final_xml = build_icra2017_scene(
                    all_stones,
                    body_poses=final_scene_poses(all_stones, wall),
                    fixed_stones=set(),
                    include_robot=True,
                    stone_friction=args.stone_friction,
                    model_name="hybrid_icra_wall_planner_partial",
                )
                args.save_final_xml.parent.mkdir(parents=True, exist_ok=True)
                args.save_final_xml.write_text(final_xml, encoding="utf-8")
                break

            wall = update_existing_entries(wall, selected)
            new_entry = to_wall_entry(selected, slot)
            wall.append(new_entry)
            current_course.append(new_entry)
            remaining = [stone for stone in remaining if stone.name != selected["stone"]]
            trajectory.append(
                serialize_scene_state(
                    len(trajectory),
                    f"placed_{selected['stone']}_course_{slot.course}_slot_{slot.index}",
                    all_stones,
                    wall,
                )
            )
            step_logs.append(
                {
                    "course": slot.course,
                    "slot_index": slot.index,
                    "target_xy": slot.target_xy.tolist(),
                    "selected": {key: value for key, value in selected.items() if key != "final_poses"},
                    "candidate_count": len(candidate_results),
                    "valid_count": len(stable),
                    "best_candidates": [
                        {key: value for key, value in item.items() if key != "final_poses"}
                        for item in sorted(candidate_results, key=lambda item: item["cost"])[:10]
                    ],
                }
            )
            print(
                "course={course} slot={slot} selected={stone} valid={valid} "
                "cost={cost:.6g} area={area:.6g} contacts={contacts} target_dist={dist:.4f}".format(
                    course=slot.course,
                    slot=slot.index,
                    stone=selected["stone"],
                    valid=selected["valid"],
                    cost=selected["cost"],
                    area=selected["support_area_m2"],
                    contacts=selected["contact_count"],
                    dist=selected["target_distance_m"],
                )
            )

        if len(current_course) != count:
            break
        course_entries.append(current_course)

    final_poses = final_scene_poses(all_stones, wall)
    final_xml = build_icra2017_scene(
        all_stones,
        body_poses=final_poses,
        fixed_stones=set(),
        include_robot=True,
        stone_friction=args.stone_friction,
        model_name="hybrid_icra_wall_planner",
    )
    args.save_final_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_final_xml.write_text(final_xml, encoding="utf-8")

    final_height = stack_top_height(wall, by_name)
    result = {
        "pipeline": {
            "name": "From-Rocks-to-Walls synthetic stones + ICRA 2017 next-best-pose wall planner",
            "stone_generation": (
                "Rectangular prism, truncated-normal vertex displacement, subdivision, convex hull, "
                "OBB centering/alignment, random density."
            ),
            "planner": (
                "Truth-state online next-best object/target-pose planning with MuJoCo contact search, "
                "support polygon area, kinetic energy, normal deviation, target distance and wall drift checks."
            ),
            "excluded": "No DQN/RL training in this path.",
        },
        "parameters": {
            "seed": args.seed,
            "stone_seed": args.stone_seed,
            "stones": args.stones,
            "courses": courses,
            "wall_width": args.wall_width,
            "rock_irregularity": args.rock_irregularity,
            "rock_subdivisions": args.rock_subdivisions,
            "samples_per_stone": args.samples_per_stone,
            "search_time": args.search_time,
            "settle_time": args.settle_time,
            "stone_friction": args.stone_friction,
            "weights": WEIGHTS,
            "Amin_m2": A_MIN,
            "Ekin_stable_j": E_KIN_STABLE,
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
        "wall": [
            {
                "course": entry.course,
                "slot_index": entry.slot_index,
                "name": entry.name,
                "target_xy": entry.target_xy.tolist(),
                "pos": entry.pos.tolist(),
                "quat": entry.quat.tolist(),
                "selected_cost": entry.selected_cost,
                "support_area_m2": entry.support_area_m2,
            }
            for entry in wall
        ],
        "trajectory": trajectory,
        "steps": step_logs,
        "placed_count": len(wall),
        "requested_count": sum(courses),
        "final_height_m": final_height,
        "final_xml": str(args.save_final_xml),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "placed_count": len(wall),
                "requested_count": sum(courses),
                "final_height_m": final_height,
                "output_json": str(args.output_json),
                "final_xml": str(args.save_final_xml),
            },
            indent=2,
        )
    )
    return 0 if len(wall) == sum(courses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
