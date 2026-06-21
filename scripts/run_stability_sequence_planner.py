#!/usr/bin/env python3
"""Stability-based sequence planner for robotic dry-stone stacking.

This script is the planning layer for the current reproduction path:

* From Rocks to Walls supplies synthetic irregular convex-hull stones.
* The ICRA 2017 scaffold supplies truth-state object/target-pose search and
  MuJoCo contact-settling evaluation.
* The RA-L 2023 Stability-Based Sequence Planning paper contributes the core
  idea used here: for every candidate sequence step, filter with masonry-style
  support heuristics, then run a simulated disturbance test and prefer the
  candidate that survives longer.

It does not run the robot arm. It writes a wall-plan report compatible with
``scripts/run_official_ur5e_robotiq_wall_stack.py --report ...``.
"""

from __future__ import annotations

import argparse
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

from scripts.run_hybrid_icra_wall_planner import (  # noqa: E402
    A_MIN,
    E_KIN_STABLE,
    FORCE_N,
    WEIGHTS,
    WallEntry,
    WallSlot,
    body_pose_dict,
    contact_points_for_support,
    evaluate_wall_pose,
    final_scene_poses,
    first_course_slots,
    import_mujoco,
    next_course_slots,
    parse_courses,
    sampled_target_pose,
    serialize_scene_state,
    stack_top_height,
    to_wall_entry,
    update_existing_entries,
)
from scripts.run_icra2017_repro import build_model, get_body_pose, quat_normalize, settle  # noqa: E402
from stone_stack.paper_scene import build_icra2017_scene  # noqa: E402
from stone_stack.rock_wall_stones import make_rock_wall_stones  # noqa: E402
from stone_stack.rocks import FlatStone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stone-seed", type=int, default=17)
    parser.add_argument("--stones", type=int, default=7)
    parser.add_argument(
        "--courses",
        default="3,2,1",
        help="Comma-separated stones per wall course. Example: 3,2 or 4,3,2.",
    )
    parser.add_argument("--wall-width", type=float, default=0.62)
    parser.add_argument("--wall-y", type=float, default=0.0)
    parser.add_argument("--rock-irregularity", type=float, default=1.0)
    parser.add_argument("--rock-subdivisions", type=int, default=5)
    parser.add_argument(
        "--max-grasp-mass",
        type=float,
        default=3.20,
        help="Exclude stones heavier than this from the robot-executable plan. Use <=0 to disable.",
    )
    parser.add_argument("--samples-per-stone", type=int, default=3)
    parser.add_argument("--slot-jitter", type=float, default=0.030)
    parser.add_argument("--yaw-jitter", type=float, default=0.24)
    parser.add_argument("--tilt-jitter", type=float, default=0.10)
    parser.add_argument("--search-time", type=float, default=0.48)
    parser.add_argument("--settle-time", type=float, default=0.80)
    parser.add_argument("--cost-contact-steps", type=int, default=12)
    parser.add_argument("--stone-friction", type=float, default=0.75)
    parser.add_argument("--force-n", type=float, default=FORCE_N)
    parser.add_argument("--max-previous-drift", type=float, default=0.080)
    parser.add_argument("--target-tolerance", type=float, default=0.120)
    parser.add_argument("--min-support-area", type=float, default=A_MIN)
    parser.add_argument("--max-normal-deviation", type=float, default=0.42)
    parser.add_argument(
        "--require-com-inside-support",
        action="store_true",
        help="Keep the old ICRA-style hard COM-in-support-polygon filter. Off by default because the shake test is the stronger stability filter.",
    )
    parser.add_argument(
        "--min-support-bodies",
        type=int,
        default=1,
        help="Minimum unique lower stones contacted by non-ground-course candidates.",
    )
    parser.add_argument("--shake-time", type=float, default=0.70)
    parser.add_argument("--shake-accel", type=float, default=2.0)
    parser.add_argument("--shake-frequency", type=float, default=2.4)
    parser.add_argument("--shake-axis-mix", type=float, default=0.35)
    parser.add_argument("--collapse-xy", type=float, default=0.070)
    parser.add_argument("--collapse-z", type=float, default=0.055)
    parser.add_argument(
        "--min-shake-survival",
        type=float,
        default=0.0,
        help="Hard survival threshold in seconds. Default keeps this as a score, not a filter.",
    )
    parser.add_argument(
        "--shake-invalid-candidates",
        action="store_true",
        help="Also run the disturbance test on candidates that fail the cheaper support filters.",
    )
    parser.add_argument("--shake-weight", type=float, default=85.0)
    parser.add_argument("--area-weight", type=float, default=0.025)
    parser.add_argument("--normal-weight", type=float, default=35.0)
    parser.add_argument("--interlock-weight", type=float, default=18.0)
    parser.add_argument("--allow-best-effort", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stability_sequence_planner.json",
    )
    parser.add_argument(
        "--save-final-xml",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stability_sequence_planner_final.xml",
    )
    return parser.parse_args()


def support_body_names(
    model,
    data,
    candidate_name: str,
    support_names: list[str],
    include_floor: bool,
) -> set[str]:
    candidate_bid = model.body(candidate_name).id
    support_bids = {model.body(name).id: name for name in support_names}
    names: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 != candidate_bid and body2 != candidate_bid:
            continue
        other_body = body2 if body1 == candidate_bid else body1
        if other_body in support_bids:
            names.add(support_bids[other_body])
        elif include_floor and other_body == 0:
            names.add("table")
    return names


def final_pose_map(selected: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (
            np.asarray(pose["pos"], dtype=float),
            quat_normalize(np.asarray(pose["quat"], dtype=float)),
        )
        for name, pose in selected["final_poses"].items()
    }


def build_mobile_model_from_result(
    mujoco,
    active_stones: list[FlatStone],
    selected: dict[str, Any],
    stone_friction: float,
):
    poses = final_pose_map(selected)
    model, data, _ = build_model(
        mujoco,
        active_stones,
        poses,
        fixed_names=set(),
        stone_friction=stone_friction,
    )
    mujoco.mj_forward(model, data)
    return model, data


def count_candidate_supports(
    mujoco,
    active_stones: list[FlatStone],
    placed: list[WallEntry],
    selected: dict[str, Any],
    slot: WallSlot,
    args: argparse.Namespace,
) -> tuple[int, list[str], int]:
    model, data = build_mobile_model_from_result(mujoco, active_stones, selected, args.stone_friction)
    placed_names = [entry.name for entry in placed]
    include_floor_support = slot.course == 0
    contact_count = 0
    support_names: set[str] = set()
    for _ in range(10):
        mujoco.mj_step(model, data)
        support_names.update(
            support_body_names(
                model,
                data,
                selected["stone"],
                placed_names,
                include_floor=include_floor_support,
            )
        )
        contact_count += len(
            contact_points_for_support(
                model,
                data,
                selected["stone"],
                placed_names,
                include_floor=include_floor_support,
            )
        )
    return len(support_names), sorted(support_names), contact_count


def shake_survival_time(
    mujoco,
    active_stones: list[FlatStone],
    selected: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Approximate the paper's shake-table test with horizontal inertial loads."""

    if args.shake_time <= 0.0:
        return {
            "survival_s": 0.0,
            "collapsed": False,
            "collapse_reason": "disabled",
            "max_xy_drift_m": 0.0,
            "max_z_drop_m": 0.0,
            "final_height_m": 0.0,
        }

    model, data = build_mobile_model_from_result(mujoco, active_stones, selected, args.stone_friction)
    settle(mujoco, model, data, 0.08)

    stone_by_name = {stone.name: stone for stone in active_stones}
    body_ids = {stone.name: model.body(stone.name).id for stone in active_stones}
    initial_positions = {
        name: get_body_pose(model, data, name)[0]
        for name in stone_by_name
    }
    steps = max(1, int(args.shake_time / model.opt.timestep))
    check_interval = max(1, int(0.010 / model.opt.timestep))
    max_xy_drift = 0.0
    max_z_drop = 0.0
    collapse_reason = ""
    survived = args.shake_time

    for step_index in range(steps):
        t = step_index * model.opt.timestep
        phase = 2.0 * math.pi * args.shake_frequency * t
        horizontal = np.array(
            [
                math.sin(phase),
                args.shake_axis_mix * math.sin(phase + 0.5 * math.pi),
                0.0,
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(horizontal[:2]))
        if norm > 1.0e-12:
            horizontal = horizontal / norm
        data.xfrc_applied[:, :] = 0.0
        for stone in active_stones:
            data.xfrc_applied[body_ids[stone.name], :3] = stone.mass * args.shake_accel * horizontal
        mujoco.mj_step(model, data)

        if step_index % check_interval != 0 and step_index != steps - 1:
            continue

        for name in stone_by_name:
            pos = get_body_pose(model, data, name)[0]
            start = initial_positions[name]
            xy_drift = float(np.linalg.norm(pos[:2] - start[:2]))
            z_drop = float(max(0.0, start[2] - pos[2]))
            max_xy_drift = max(max_xy_drift, xy_drift)
            max_z_drop = max(max_z_drop, z_drop)
            if xy_drift > args.collapse_xy:
                survived = t
                collapse_reason = f"{name} xy drift {xy_drift:.4f} m"
                break
            if z_drop > args.collapse_z:
                survived = t
                collapse_reason = f"{name} z drop {z_drop:.4f} m"
                break
        if collapse_reason:
            break

    data.xfrc_applied[:, :] = 0.0
    final_height = max(float(get_body_pose(model, data, name)[0][2]) for name in stone_by_name)
    return {
        "survival_s": float(survived),
        "collapsed": bool(collapse_reason),
        "collapse_reason": collapse_reason or "survived_full_test",
        "max_xy_drift_m": float(max_xy_drift),
        "max_z_drop_m": float(max_z_drop),
        "final_height_m": final_height,
    }


def add_stability_terms(
    mujoco,
    active_stones: list[FlatStone],
    placed: list[WallEntry],
    candidate: FlatStone,
    selected: dict[str, Any],
    slot: WallSlot,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enriched = dict(selected)
    support_count, support_names, support_contact_count = count_candidate_supports(
        mujoco,
        active_stones,
        placed,
        selected,
        slot,
        args,
    )
    area = float(selected["support_area_m2"])
    normal_deviation = float(selected["normal_deviation"])
    contact_count = int(selected["contact_count"])
    energy = float(selected["kinetic_energy_j"])
    previous_drift = float(selected["previous_wall_drift_m"])
    target_distance = float(selected["target_distance_m"])
    min_course_height = 0.010 if slot.course == 0 else 0.040 + slot.course * 0.030
    pose_filter_valid = bool(
        contact_count >= 3
        and area >= args.min_support_area
        and energy <= E_KIN_STABLE
        and previous_drift < args.max_previous_drift
        and target_distance <= args.target_tolerance
        and float(selected["final_z_m"]) >= min_course_height
    )
    if args.require_com_inside_support:
        pose_filter_valid = bool(pose_filter_valid and selected["com_projection_inside_support"])

    interlock_ok = bool(slot.course == 0 or support_count >= args.min_support_bodies)
    area_ok = bool(area >= args.min_support_area)
    normal_ok = bool(normal_deviation <= args.max_normal_deviation)
    shake = {
        "survival_s": 0.0,
        "collapsed": True,
        "collapse_reason": "skipped_failed_support_filters",
        "max_xy_drift_m": 0.0,
        "max_z_drop_m": 0.0,
        "final_height_m": 0.0,
    }
    if pose_filter_valid or args.shake_invalid_candidates:
        shake = shake_survival_time(mujoco, active_stones, selected, args)

    shake_ok = bool(float(shake["survival_s"]) >= args.min_shake_survival)
    stability_valid = bool(pose_filter_valid and interlock_ok and area_ok and normal_ok and shake_ok)

    shake_loss = max(0.0, args.shake_time - float(shake["survival_s"]))
    missing_supports = max(0, args.min_support_bodies - support_count) if slot.course > 0 else 0
    base_pose_cost = (
        WEIGHTS["support_area_inverse"] / max(area, A_MIN)
        + WEIGHTS["kinetic_energy"] * energy
        + WEIGHTS["center_distance"] * target_distance
        + WEIGHTS["normal_deviation"] * normal_deviation
    )
    stability_cost = (
        base_pose_cost
        + args.shake_weight * shake_loss
        + args.area_weight / max(area, args.min_support_area)
        + args.normal_weight * normal_deviation
        + args.interlock_weight * missing_supports
    )
    if not stability_valid:
        stability_cost += 1.0e4

    enriched.update(
        {
            "stone_length_m": float(candidate.length),
            "stone_width_m": float(candidate.width),
            "stone_thickness_m": float(candidate.thickness),
            "stability_valid": stability_valid,
            "valid": stability_valid,
            "base_pose_valid": pose_filter_valid,
            "icra_strict_valid": bool(selected["valid"]),
            "icra_strict_cost": float(selected["cost"]),
            "pose_cost": float(base_pose_cost),
            "stability_cost": float(stability_cost),
            "cost": float(stability_cost),
            "support_body_count": int(support_count),
            "support_bodies": support_names,
            "support_contact_observations": int(support_contact_count),
            "interlock_valid": interlock_ok,
            "area_valid": area_ok,
            "normal_valid": normal_ok,
            "shake_valid": shake_ok,
            "shake": shake,
        }
    )
    return enriched


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(item["shake"]["survival_s"]),
        float(item["cost"]),
        -float(item["support_area_m2"]),
        float(item["target_distance_m"]),
    )


def candidate_log_without_final_poses(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "final_poses"}


def plan_slot(
    mujoco,
    rng: random.Random,
    all_stones_by_name: dict[str, FlatStone],
    remaining: list[FlatStone],
    wall: list[WallEntry],
    slot: WallSlot,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_results: list[dict[str, Any]] = []
    for candidate in list(remaining):
        active = [all_stones_by_name[entry.name] for entry in wall] + [candidate]
        for _ in range(args.samples_per_stone):
            pos, quat = sampled_target_pose(rng, slot, candidate, wall, all_stones_by_name, args)
            base_result = evaluate_wall_pose(mujoco, active, wall, candidate, pos, quat, slot, args)
            candidate_results.append(
                add_stability_terms(mujoco, active, wall, candidate, base_result, slot, args)
            )

    stable = [item for item in candidate_results if item["valid"]]
    selected: dict[str, Any] | None
    if stable:
        selected = sorted(stable, key=candidate_sort_key)[0]
    elif args.allow_best_effort and candidate_results:
        selected = sorted(candidate_results, key=candidate_sort_key)[0]
    else:
        selected = None
    return selected, candidate_results, stable


def write_final_xml(
    all_stones: list[FlatStone],
    wall: list[WallEntry],
    args: argparse.Namespace,
    model_name: str,
) -> None:
    final_xml = build_icra2017_scene(
        all_stones,
        body_poses=final_scene_poses(all_stones, wall),
        fixed_stones=set(),
        include_robot=True,
        stone_friction=args.stone_friction,
        model_name=model_name,
    )
    args.save_final_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_final_xml.write_text(final_xml, encoding="utf-8")


def main() -> int:
    args = parse_args()
    courses = parse_courses(args.courses)
    if args.stones < sum(courses):
        raise SystemExit("--stones must be at least the total requested wall slots")
    if args.samples_per_stone <= 0:
        raise SystemExit("--samples-per-stone must be positive")

    all_stones = make_rock_wall_stones(
        seed=args.stone_seed,
        count=args.stones,
        irregularity=args.rock_irregularity,
        subdivisions=args.rock_subdivisions,
    )
    if args.max_grasp_mass > 0.0:
        planning_stones = [stone for stone in all_stones if stone.mass <= args.max_grasp_mass]
    else:
        planning_stones = list(all_stones)
    if len(planning_stones) < sum(courses):
        raise SystemExit(
            f"only {len(planning_stones)} stones pass --max-grasp-mass; "
            f"{sum(courses)} are required"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "pipeline": (
                        "from-rocks-to-walls-stones + icra2017-pose-search + "
                        "stability-based sequence planning"
                    ),
                    "courses": courses,
                    "shake_test": {
                        "time_s": args.shake_time,
                        "accel_m_s2": args.shake_accel,
                        "frequency_hz": args.shake_frequency,
                        "collapse_xy_m": args.collapse_xy,
                        "collapse_z_m": args.collapse_z,
                    },
                    "graspability_filter": {
                        "max_grasp_mass_kg": args.max_grasp_mass,
                        "available_for_planning": [stone.name for stone in planning_stones],
                        "excluded": [
                            stone.name for stone in all_stones if stone.name not in {item.name for item in planning_stones}
                        ],
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
                },
                indent=2,
            )
        )
        return 0

    mujoco = import_mujoco()
    rng = random.Random(args.seed)
    by_name = {stone.name: stone for stone in all_stones}
    remaining = list(planning_stones)
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
            selected, candidate_results, stable = plan_slot(
                mujoco,
                rng,
                by_name,
                remaining,
                wall,
                slot,
                args,
            )
            if selected is None:
                step_logs.append(
                    {
                        "course": slot.course,
                        "slot_index": slot.index,
                        "target_xy": slot.target_xy.tolist(),
                        "selected": None,
                        "candidate_count": len(candidate_results),
                        "valid_count": len(stable),
                        "best_candidates": [
                            candidate_log_without_final_poses(item)
                            for item in sorted(candidate_results, key=candidate_sort_key)[:10]
                        ],
                    }
                )
                print(
                    f"course={slot.course} slot={slot.index} "
                    f"no_valid_candidate candidates={len(candidate_results)}",
                    flush=True,
                )
                write_final_xml(all_stones, wall, args, "stability_sequence_planner_partial")
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
                    "selected": candidate_log_without_final_poses(selected),
                    "candidate_count": len(candidate_results),
                    "valid_count": len(stable),
                    "best_candidates": [
                        candidate_log_without_final_poses(item)
                        for item in sorted(candidate_results, key=candidate_sort_key)[:10]
                    ],
                }
            )
            print(
                "course={course} slot={slot} selected={stone} valid={valid} "
                "shake={shake:.3f}/{shake_time:.3f}s cost={cost:.6g} "
                "area={area:.6g} supports={supports} target_dist={dist:.4f}".format(
                    course=slot.course,
                    slot=slot.index,
                    stone=selected["stone"],
                    valid=selected["valid"],
                    shake=float(selected["shake"]["survival_s"]),
                    shake_time=args.shake_time,
                    cost=selected["cost"],
                    area=selected["support_area_m2"],
                    supports=selected["support_bodies"],
                    dist=selected["target_distance_m"],
                ),
                flush=True,
            )

        if len(current_course) != count:
            break
        course_entries.append(current_course)

    write_final_xml(all_stones, wall, args, "stability_sequence_planner")
    final_height = stack_top_height(wall, by_name)
    result = {
        "pipeline": {
            "name": (
                "From-Rocks-to-Walls synthetic stones + ICRA 2017 pose search + "
                "Stability-Based sequence planning"
            ),
            "stone_generation": (
                "Rectangular prism, truncated-normal vertex displacement, subdivision, "
                "convex hull, OBB centering/alignment, random density."
            ),
            "planner": (
                "Truth-state online sequence planning. Candidates are settled in MuJoCo, "
                "filtered by support area, support normal, COM projection, target error, "
                "previous-wall drift and interlock contact count, then ranked by simulated "
                "disturbance survival time."
            ),
            "shake_model": (
                "Horizontal inertial force approximation applied to every active stone. "
                "This is a MuJoCo approximation of the paper's shake-table stability test; "
                "it is not yet a moving-table model."
            ),
            "robot_execution": "Use scripts/run_official_ur5e_robotiq_wall_stack.py with this report.",
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
            "support_filters": {
                "min_support_area_m2": args.min_support_area,
                "max_normal_deviation": args.max_normal_deviation,
                "min_support_bodies": args.min_support_bodies,
                "require_com_inside_support": args.require_com_inside_support,
                "max_previous_drift_m": args.max_previous_drift,
                "target_tolerance_m": args.target_tolerance,
            },
            "graspability": {
                "max_grasp_mass_kg": args.max_grasp_mass,
                "planning_stones": [stone.name for stone in planning_stones],
                "excluded_stones": [
                    stone.name for stone in all_stones if stone.name not in {item.name for item in planning_stones}
                ],
            },
            "shake": {
                "time_s": args.shake_time,
                "accel_m_s2": args.shake_accel,
                "frequency_hz": args.shake_frequency,
                "axis_mix": args.shake_axis_mix,
                "collapse_xy_m": args.collapse_xy,
                "collapse_z_m": args.collapse_z,
                "min_survival_s": args.min_shake_survival,
            },
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
                "planner_available": stone.name in {item.name for item in planning_stones},
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
        ),
        flush=True,
    )
    return 0 if len(wall) == sum(courses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
