#!/usr/bin/env python3
"""ICRA 2017-style next-best stone pose demo in MuJoCo.

This is a short-term reproduction of the paper's core planning idea under
truth-state assumptions: known stone geometry, known poses, sampled candidate
target poses, physics-based release evaluation, and greedy selection of the next
best object/pose.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stone_stack.mjcf_builder import build_truth_gripper_scene
from stone_stack.rocks import FlatStone, make_flat_stones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--stones", type=int, default=4)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--samples-per-stone", type=int, default=5)
    parser.add_argument("--eval-time", type=float, default=3.0)
    parser.add_argument("--stable-speed", type=float, default=0.045)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "paper_pose_demo.json")
    parser.add_argument("--output-html", type=Path, default=PROJECT_ROOT / "outputs" / "paper_pose_demo.html")
    parser.add_argument("--save-final-xml", type=Path, default=PROJECT_ROOT / "outputs" / "paper_pose_demo_final.xml")
    return parser.parse_args()


def import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo is not installed. Run: source .venv/bin/activate") from exc
    return mujoco


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def quat_yaw(quat: np.ndarray | list[float]) -> float:
    w, _x, _y, z = quat
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def freejoint_addresses(model, joint_name: str) -> tuple[int, int]:
    jid = model.joint(joint_name).id
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def set_free_body_pose(model, data, joint_name: str, pos, quat, zero_velocity: bool = True):
    qposadr, qveladr = freejoint_addresses(model, joint_name)
    data.qpos[qposadr : qposadr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qposadr + 3 : qposadr + 7] = np.asarray(quat, dtype=float)
    if zero_velocity:
        data.qvel[qveladr : qveladr + 6] = 0.0


def get_free_body_pose(model, data, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    qposadr, _ = freejoint_addresses(model, joint_name)
    return data.qpos[qposadr : qposadr + 3].copy(), data.qpos[qposadr + 3 : qposadr + 7].copy()


def max_body_speed(model, data, joint_names: list[str]) -> float:
    speeds = []
    for joint_name in joint_names:
        _, qveladr = freejoint_addresses(model, joint_name)
        linear = data.qvel[qveladr : qveladr + 3]
        angular = data.qvel[qveladr + 3 : qveladr + 6]
        speeds.append(float(np.linalg.norm(linear) + 0.15 * np.linalg.norm(angular)))
    return max(speeds) if speeds else 0.0


def rectangle(center: np.ndarray, length: float, width: float, yaw: float) -> list[np.ndarray]:
    hx = 0.5 * length
    hy = 0.5 * width
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return [center[:2] + rot @ np.asarray(point) for point in local]


def polygon_area(poly: list[np.ndarray]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, point in enumerate(poly):
        nxt = poly[(i + 1) % len(poly)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(0.5 * area)


def _inside(point: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> bool:
    edge = edge_b - edge_a
    rel = point - edge_a
    return edge[0] * rel[1] - edge[1] * rel[0] >= -1e-10


def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py])


def polygon_clip(subject: list[np.ndarray], clip: list[np.ndarray]) -> list[np.ndarray]:
    output = subject
    for i, edge_a in enumerate(clip):
        edge_b = clip[(i + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        prev = input_poly[-1]
        for current in input_poly:
            if _inside(current, edge_a, edge_b):
                if not _inside(prev, edge_a, edge_b):
                    output.append(_line_intersection(prev, current, edge_a, edge_b))
                output.append(current)
            elif _inside(prev, edge_a, edge_b):
                output.append(_line_intersection(prev, current, edge_a, edge_b))
            prev = current
    return output


def overlap_area(a_center, a_stone: FlatStone, a_yaw: float, b_center, b_stone: FlatStone, b_yaw: float) -> float:
    a_rect = rectangle(np.asarray(a_center), a_stone.length, a_stone.width, a_yaw)
    b_rect = rectangle(np.asarray(b_center), b_stone.length, b_stone.width, b_yaw)
    return polygon_area(polygon_clip(a_rect, b_rect))


def build_model_with_poses(mujoco, stones: list[FlatStone], poses: dict[str, tuple[np.ndarray, np.ndarray]]):
    body_poses = {name: (tuple(pos), tuple(quat)) for name, (pos, quat) in poses.items()}
    xml = build_truth_gripper_scene(stones, body_poses=body_poses, include_gripper=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def simulate_candidate(mujoco, stones, stack, candidate, target_pos, target_yaw, eval_time, stable_speed):
    poses = {entry["name"]: (np.asarray(entry["pos"], dtype=float), np.asarray(entry["quat"], dtype=float)) for entry in stack}
    poses[candidate.name] = (np.asarray(target_pos, dtype=float), np.asarray(yaw_quat(target_yaw), dtype=float))

    model, data = build_model_with_poses(mujoco, stones, poses)
    mujoco.mj_forward(model, data)
    for _ in range(int(eval_time / model.opt.timestep)):
        mujoco.mj_step(model, data)

    used_names = [entry["name"] for entry in stack] + [candidate.name]
    final = {}
    for name in used_names:
        pos, quat = get_free_body_pose(model, data, f"{name}_free")
        final[name] = {"pos": pos.tolist(), "quat": quat.tolist()}

    speed = max_body_speed(model, data, [f"{name}_free" for name in used_names])
    support = stack[-1]
    candidate_pos = np.asarray(final[candidate.name]["pos"], dtype=float)
    support_pos = np.asarray(final[support["name"]]["pos"], dtype=float)
    horizontal_error = float(np.linalg.norm(candidate_pos[:2] - support_pos[:2]))
    height_gain = float(candidate_pos[2] - support_pos[2])
    previous_drift = 0.0
    for entry in stack:
        before = np.asarray(entry["pos"], dtype=float)
        after = np.asarray(final[entry["name"]]["pos"], dtype=float)
        previous_drift = max(previous_drift, float(np.linalg.norm(after - before)))

    support_stone = stones_by_name(stones)[support["name"]]
    support_yaw = quat_yaw(support["quat"])
    area = overlap_area(candidate_pos, candidate, target_yaw, support_pos, support_stone, support_yaw)
    stable = bool(
        speed < stable_speed
        and previous_drift < 0.035
        and horizontal_error < 0.10
        and height_gain > 0.45 * support_stone.thickness
        and area > 1.0e-4
    )
    cost = 0.179 / max(area, 1.0e-5) + 0.472 * speed + 0.094 * horizontal_error
    if not stable:
        cost += 50.0

    return {
        "stone": candidate.name,
        "target_pos": list(map(float, target_pos)),
        "target_yaw": target_yaw,
        "cost": float(cost),
        "stable": bool(stable),
        "support_area_proxy": float(area),
        "max_speed": float(speed),
        "horizontal_error_m": horizontal_error,
        "height_gain_m": height_gain,
        "previous_stack_drift_m": previous_drift,
        "final": final,
    }


def stones_by_name(stones: list[FlatStone]) -> dict[str, FlatStone]:
    return {stone.name: stone for stone in stones}


def make_initial_stack(mujoco, stones: list[FlatStone]) -> list[dict]:
    base = max(stones, key=lambda stone: stone.length * stone.width)
    poses = {base.name: (np.array([0.0, 0.0, 0.15]), np.array([1.0, 0.0, 0.0, 0.0]))}
    model, data = build_model_with_poses(mujoco, stones, poses)
    mujoco.mj_forward(model, data)
    for _ in range(int(1.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    pos, quat = get_free_body_pose(model, data, f"{base.name}_free")
    return [{"name": base.name, "pos": pos.tolist(), "quat": quat.tolist(), "level": 0, "selected_cost": 0.0}]


def export_demo_html(path: Path, result: dict, stones: list[FlatStone]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result)
    stone_meta = {
        stone.name: {
            "length": stone.length,
            "width": stone.width,
            "thickness": stone.thickness,
            "rgba": stone.rgba,
        }
        for stone in stones
    }
    meta = json.dumps(stone_meta)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MuJoCo Stone Stacking Paper Demo</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f4f1eb; color: #222; }}
    header {{ padding: 18px 24px; background: #27323a; color: white; }}
    main {{ display: grid; grid-template-columns: 1fr 380px; gap: 18px; padding: 18px; }}
    svg {{ width: 100%; height: 640px; background: #fbfaf7; border: 1px solid #d7d0c6; }}
    aside {{ background: white; border: 1px solid #d7d0c6; padding: 14px; }}
    button {{ padding: 8px 12px; margin-right: 8px; }}
    .metric {{ margin: 8px 0; }}
    pre {{ white-space: pre-wrap; font-size: 12px; background: #f7f7f7; padding: 10px; max-height: 360px; overflow: auto; }}
  </style>
</head>
<body>
  <header>
    <h2>ICRA 2017-style next-best stone pose search, MuJoCo truth-state demo</h2>
    <div>Known stones, sampled candidate poses, physics release evaluation, greedy object/pose selection.</div>
  </header>
  <main>
    <svg id="scene" viewBox="-260 -120 520 360"></svg>
    <aside>
      <button onclick="prev()">Prev</button>
      <button onclick="next()">Next</button>
      <button onclick="play()">Play</button>
      <div id="metrics"></div>
      <pre id="log"></pre>
    </aside>
  </main>
  <script>
    const result = {payload};
    const stones = {meta};
    let step = 0;
    const svg = document.getElementById('scene');
    function color(name) {{
      const rgba = stones[name].rgba;
      return `rgb(${{Math.round(rgba[0]*255)}}, ${{Math.round(rgba[1]*255)}}, ${{Math.round(rgba[2]*255)}})`;
    }}
    function draw() {{
      svg.innerHTML = '';
      const stack = result.stack.slice(0, step + 1);
      const supply = result.available_by_step[step] || [];
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      svg.appendChild(g);
      const floor = document.createElementNS(svg.namespaceURI, 'line');
      floor.setAttribute('x1', -220); floor.setAttribute('x2', 220);
      floor.setAttribute('y1', 260); floor.setAttribute('y2', 260);
      floor.setAttribute('stroke', '#777'); floor.setAttribute('stroke-width', 2);
      g.appendChild(floor);
      for (const entry of stack) {{
        const s = stones[entry.name];
        const x = entry.pos[0] * 700;
        const y = 260 - entry.pos[2] * 1400;
        const w = s.length * 700;
        const h = Math.max(8, s.thickness * 1400);
        const r = document.createElementNS(svg.namespaceURI, 'rect');
        r.setAttribute('x', x - w/2); r.setAttribute('y', y - h/2);
        r.setAttribute('width', w); r.setAttribute('height', h);
        r.setAttribute('rx', 4);
        r.setAttribute('fill', color(entry.name));
        r.setAttribute('stroke', '#252525');
        r.setAttribute('transform', `rotate(${{entry.yaw_deg || 0}} ${{x}} ${{y}})`);
        g.appendChild(r);
      }}
      let sy = 30;
      for (const name of supply) {{
        const s = stones[name];
        const r = document.createElementNS(svg.namespaceURI, 'rect');
        r.setAttribute('x', -235); r.setAttribute('y', sy);
        r.setAttribute('width', s.length * 420); r.setAttribute('height', Math.max(8, s.thickness * 900));
        r.setAttribute('fill', color(name)); r.setAttribute('stroke', '#444'); r.setAttribute('opacity', '0.65');
        g.appendChild(r);
        const t = document.createElementNS(svg.namespaceURI, 'text');
        t.setAttribute('x', -235); t.setAttribute('y', sy - 5); t.textContent = name;
        t.setAttribute('font-size', '10'); g.appendChild(t);
        sy += 42;
      }}
      const sel = result.decisions[Math.max(0, step - 1)];
      document.getElementById('metrics').innerHTML = `
        <div class="metric"><b>Step:</b> ${{step}} / ${{result.stack.length - 1}}</div>
        <div class="metric"><b>Final height:</b> ${{result.final_height_m.toFixed(3)}} m</div>
        <div class="metric"><b>Stable:</b> ${{result.stable}}</div>
        <div class="metric"><b>Last selected:</b> ${{sel ? sel.selected.stone : result.stack[0].name}}</div>
        <div class="metric"><b>Last cost:</b> ${{sel ? sel.selected.cost.toFixed(3) : 'base'}}</div>`;
      document.getElementById('log').textContent = JSON.stringify(sel || result.stack[0], null, 2);
    }}
    function next() {{ step = Math.min(step + 1, result.stack.length - 1); draw(); }}
    function prev() {{ step = Math.max(step - 1, 0); draw(); }}
    function play() {{ step = 0; draw(); const id = setInterval(() => {{ if (step >= result.stack.length - 1) clearInterval(id); else next(); }}, 900); }}
    draw();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.stones < 2:
        raise SystemExit("--stones must be at least 2")

    mujoco = import_mujoco()
    rng = random.Random(args.seed)
    stones = make_flat_stones(args.stones, args.seed)
    name_to_stone = stones_by_name(stones)
    stack = make_initial_stack(mujoco, stones)
    remaining = [stone.name for stone in stones if stone.name != stack[0]["name"]]
    decisions = []
    available_by_step = [remaining.copy()]

    while remaining and len(stack) < args.levels:
        support = stack[-1]
        support_stone = name_to_stone[support["name"]]
        support_pos = np.asarray(support["pos"], dtype=float)
        candidates = []
        for stone_name in remaining:
            stone = name_to_stone[stone_name]
            for sample in range(args.samples_per_stone):
                yaw = rng.uniform(-math.pi, math.pi)
                offset_scale = min(support_stone.length, support_stone.width) * 0.12
                offset = np.array([rng.uniform(-offset_scale, offset_scale), rng.uniform(-offset_scale, offset_scale), 0.0])
                target_pos = support_pos + offset
                target_pos[2] = support_pos[2] + 0.5 * support_stone.thickness + 0.5 * stone.thickness + 0.006
                candidates.append(
                    simulate_candidate(
                        mujoco=mujoco,
                        stones=stones,
                        stack=stack,
                        candidate=stone,
                        target_pos=target_pos,
                        target_yaw=yaw,
                        eval_time=args.eval_time,
                        stable_speed=args.stable_speed,
                    )
                )

        stable_candidates = [candidate for candidate in candidates if candidate["stable"]]
        chosen = min(stable_candidates or candidates, key=lambda candidate: candidate["cost"])
        final_pose = chosen["final"][chosen["stone"]]
        yaw_deg = math.degrees(quat_yaw(final_pose["quat"]))
        stack.append(
            {
                "name": chosen["stone"],
                "pos": final_pose["pos"],
                "quat": final_pose["quat"],
                "level": len(stack),
                "selected_cost": chosen["cost"],
                "yaw_deg": yaw_deg,
            }
        )
        remaining.remove(chosen["stone"])
        decisions.append(
            {
                "level": len(stack) - 1,
                "selected": chosen,
                "candidate_summary": sorted(
                    [
                        {
                            "stone": candidate["stone"],
                            "cost": candidate["cost"],
                            "stable": candidate["stable"],
                            "support_area_proxy": candidate["support_area_proxy"],
                            "max_speed": candidate["max_speed"],
                        }
                        for candidate in candidates
                    ],
                    key=lambda item: item["cost"],
                )[:8],
            }
        )
        available_by_step.append(remaining.copy())
        if not chosen["stable"]:
            break

    final_height = max(entry["pos"][2] + 0.5 * name_to_stone[entry["name"]].thickness for entry in stack)
    result = {
        "paper": "Furrer et al., ICRA 2017, Autonomous robotic stone stacking with online next best object target pose planning",
        "reproduction_scope": "truth-state MuJoCo next-best object/pose search; perception and real arm execution omitted",
        "seed": args.seed,
        "stones": args.stones,
        "levels_requested": args.levels,
        "samples_per_stone": args.samples_per_stone,
        "stable": all(decision["selected"]["stable"] for decision in decisions),
        "stacked_count": len(stack),
        "final_height_m": float(final_height),
        "stone_metadata": {
            stone.name: {
                "length": stone.length,
                "width": stone.width,
                "thickness": stone.thickness,
                "mass": stone.mass,
                "rgba": stone.rgba,
            }
            for stone in stones
        },
        "stack": stack,
        "available_by_step": available_by_step,
        "decisions": decisions,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    export_demo_html(args.output_html, result, stones)

    final_poses = {
        entry["name"]: (tuple(entry["pos"]), tuple(entry["quat"]))
        for entry in stack
    }
    args.save_final_xml.parent.mkdir(parents=True, exist_ok=True)
    args.save_final_xml.write_text(
        build_truth_gripper_scene(stones, body_poses=final_poses, include_gripper=False),
        encoding="utf-8",
    )

    print(json.dumps({
        "stable": result["stable"],
        "stacked_count": result["stacked_count"],
        "final_height_m": result["final_height_m"],
        "output_json": str(args.output_json),
        "output_html": str(args.output_html),
        "final_xml": str(args.save_final_xml),
    }, indent=2))
    return 0 if result["stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
