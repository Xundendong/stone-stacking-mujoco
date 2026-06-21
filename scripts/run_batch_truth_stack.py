#!/usr/bin/env python3
"""Batch-run truth-state flat-stone stacking over multiple random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--transport-mode", choices=("oracle", "contact"), default="oracle")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "batch_truth_stack.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    runner = PROJECT_ROOT / "scripts" / "run_truth_pick_stack.py"

    for seed in range(args.start_seed, args.start_seed + args.trials):
        command = [
            sys.executable,
            str(runner),
            "--seed",
            str(seed),
            "--transport-mode",
            args.transport_mode,
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = {
                "seed": seed,
                "transport_mode": args.transport_mode,
                "stable": False,
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
        result["returncode"] = completed.returncode
        results.append(result)
        print(
            f"seed={seed:03d} stable={result.get('stable')} "
            f"stacked={result.get('stacked')} height_gain={result.get('height_gain_m')}"
        )

    stable_count = sum(1 for result in results if result.get("stable"))
    summary = {
        "transport_mode": args.transport_mode,
        "start_seed": args.start_seed,
        "trials": args.trials,
        "stable_count": stable_count,
        "success_rate": stable_count / args.trials if args.trials else 0.0,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if stable_count == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())

