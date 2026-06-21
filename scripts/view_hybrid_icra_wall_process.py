#!/usr/bin/env python3
"""Replay hybrid ICRA wall-planning states in MuJoCo viewer.

This opens an interactive MuJoCo viewer and replays the committed planner
states saved by ``run_hybrid_icra_wall_planner.py``. It is not a rendered video:
the states are applied to the live MuJoCo model so the user can inspect the
wall-building process in the simulation environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.paper_scene import build_icra2017_scene
from stone_stack.rock_wall_stones import make_rock_wall_stones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "hybrid_icra_wall_planner.json")
    parser.add_argument("--seconds-per-step", type=float, default=1.2)
    parser.add_argument("--interpolate-frames", type=int, default=40)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1.0e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def quat_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    if float(a @ b) < 0.0:
        b = -b
    return quat_normalize((1.0 - t) * a + t * b)


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def freejoint_qpos_addr(model, stone_name: str) -> int:
    jid = model.joint(f"{stone_name}_free").id
    return int(model.jnt_qposadr[jid])


def set_pose(model, data, stone_name: str, pos: np.ndarray, quat: np.ndarray):
    qadr = freejoint_qpos_addr(model, stone_name)
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = quat_normalize(quat)


def report_stones(report: dict):
    params = report["parameters"]
    return make_rock_wall_stones(
        seed=int(params["stone_seed"]),
        count=int(params["stones"]),
        irregularity=float(params["rock_irregularity"]),
        subdivisions=int(params["rock_subdivisions"]),
    )


def state_pose(state: dict, stone_name: str) -> tuple[np.ndarray, np.ndarray]:
    pose = state["poses"][stone_name]
    return np.asarray(pose["pos"], dtype=float), np.asarray(pose["quat"], dtype=float)


def apply_state(model, data, state: dict, stone_names: list[str]):
    import mujoco

    for name in stone_names:
        pos, quat = state_pose(state, name)
        set_pose(model, data, name, pos, quat)
    mujoco.mj_forward(model, data)


def interpolate_state(model, data, start: dict, end: dict, stone_names: list[str], frames: int, frame_dt: float):
    import mujoco

    for frame in range(frames):
        t = smooth(frame / max(1, frames - 1))
        for name in stone_names:
            p0, q0 = state_pose(start, name)
            p1, q1 = state_pose(end, name)
            set_pose(model, data, name, (1.0 - t) * p0 + t * p1, quat_lerp(q0, q1, t))
        mujoco.mj_forward(model, data)
        yield
        time.sleep(frame_dt)


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    trajectory = report.get("trajectory")
    if not trajectory:
        raise SystemExit(
            "Report does not contain trajectory states. Re-run: python scripts/run_hybrid_icra_wall_planner.py"
        )

    import mujoco
    import mujoco.viewer

    stones = report_stones(report)
    initial_poses = {
        name: (pose["pos"], pose["quat"])
        for name, pose in trajectory[0]["poses"].items()
    }
    xml = build_icra2017_scene(
        stones,
        body_poses=initial_poses,
        fixed_stones=set(),
        include_robot=True,
        stone_friction=float(report["parameters"].get("stone_friction", 0.6)),
        model_name="hybrid_icra_wall_process_replay",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    stone_names = [stone.name for stone in stones]

    frame_dt = max(0.001, args.seconds_per_step / max(1, args.interpolate_frames))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for index, state in enumerate(trajectory):
                apply_state(model, data, state, stone_names)
                viewer.sync()
                time.sleep(max(0.15, 0.35 * args.seconds_per_step))
                if index + 1 >= len(trajectory):
                    continue
                for _ in interpolate_state(
                    model,
                    data,
                    state,
                    trajectory[index + 1],
                    stone_names,
                    args.interpolate_frames,
                    frame_dt,
                ):
                    viewer.sync()
                    if not viewer.is_running():
                        return 0
            if not args.loop:
                while viewer.is_running():
                    apply_state(model, data, trajectory[-1], stone_names)
                    viewer.sync()
                    time.sleep(0.05)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
