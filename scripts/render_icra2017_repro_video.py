#!/usr/bin/env python3
"""Render the generated ICRA 2017 MuJoCo scene to an MP4 video."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_repro_final.xml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_repro_mujoco.mp4")
    parser.add_argument("--frames-dir", type=Path, default=PROJECT_ROOT / "outputs" / "icra2017_repro_video_frames")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write MP4 video")

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    for old in args.frames_dir.glob("frame_*.png"):
        old.unlink()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.00, -0.04, 0.23]
    camera.distance = 1.05
    camera.elevation = -22.0

    total_frames = max(1, int(args.fps * args.seconds))
    for frame in range(total_frames):
        t = frame / max(1, total_frames - 1)
        camera.azimuth = 138.0 + 52.0 * math.sin(math.pi * (t - 0.15))
        camera.distance = 1.02 + 0.08 * math.sin(math.tau * t)
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        Image.fromarray(image).save(args.frames_dir / f"frame_{frame:05d}.png")

    renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(args.frames_dir / "frame_%05d.png"),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(args.output),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
