#!/usr/bin/env python3
"""Render a presentation MP4 for the ICRA-style stone stacking demo.

This renderer deliberately does not use MuJoCo OpenGL. It reads the JSON output
from ``run_paper_pose_demo.py`` and creates a clean 2D side-view animation that
is robust on machines without a working GL context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=PROJECT_ROOT / "reports" / "paper_pose_demo.json")
    parser.add_argument("--output-mp4", type=Path, default=PROJECT_ROOT / "outputs" / "paper_pose_demo.mp4")
    parser.add_argument("--frames-dir", type=Path, default=PROJECT_ROOT / "outputs" / "paper_demo_frames")
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_H2 = load_font(24, bold=True)
FONT_BODY = load_font(20)
FONT_SMALL = load_font(16)
FONT_MONO = load_font(15)


def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def color_from_rgba(rgba) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * 255))) for c in rgba[:3])


def projected_width(stone_meta: dict, yaw_deg: float) -> float:
    # Approximate side-view width from yaw.
    import math

    yaw = math.radians(yaw_deg)
    return abs(stone_meta["length"] * math.cos(yaw)) + abs(stone_meta["width"] * math.sin(yaw))


class Renderer:
    def __init__(self, result: dict, width: int = 1280, height: int = 720):
        self.result = result
        self.width = width
        self.height = height
        self.meta = result["stone_metadata"]
        self.scale = 1700.0
        self.origin_x = width // 2
        self.floor_y = 560

    def world_to_screen(self, x: float, z: float) -> tuple[int, int]:
        return int(self.origin_x + x * self.scale), int(self.floor_y - z * self.scale)

    def base_canvas(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (self.width, self.height), (246, 242, 235))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.width, 84), fill=(39, 50, 58))
        draw.text((30, 18), "ICRA 2017-style Robotic Stone Stacking Demo", fill=(255, 255, 255), font=FONT_TITLE)
        draw.text(
            (32, 57),
            "Truth-state MuJoCo reproduction: next-best object/pose search + physics release evaluation",
            fill=(218, 226, 231),
            font=FONT_SMALL,
        )
        draw.line((130, self.floor_y, self.width - 120, self.floor_y), fill=(120, 111, 100), width=3)
        draw.text((132, self.floor_y + 12), "table / ground", fill=(108, 100, 92), font=FONT_SMALL)
        return img, draw

    def draw_stone(self, draw, entry: dict, alpha: float = 1.0, override_pos=None, outline=(35, 35, 35)):
        name = entry["name"]
        meta = self.meta[name]
        pos = override_pos if override_pos is not None else entry["pos"]
        yaw_deg = entry.get("yaw_deg", 0.0)
        cx, cy = self.world_to_screen(pos[0], pos[2])
        w = max(18, projected_width(meta, yaw_deg) * self.scale)
        h = max(12, meta["thickness"] * self.scale)
        fill = color_from_rgba(meta["rgba"])
        if alpha < 1.0:
            fill = tuple(int(lerp(246, c, alpha)) for c in fill)
        x0, y0 = cx - w / 2, cy - h / 2
        x1, y1 = cx + w / 2, cy + h / 2
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=fill, outline=outline, width=2)
        draw.text((x0 + 6, y0 + 4), name, fill=(20, 20, 20), font=FONT_SMALL)

    def draw_stack(self, draw, stack_entries: list[dict], moving: dict | None = None):
        for entry in stack_entries:
            self.draw_stone(draw, entry)
        if moving is not None:
            self.draw_stone(draw, moving["entry"], override_pos=moving["pos"], outline=(25, 82, 160))
            cx, cy = self.world_to_screen(moving["pos"][0], moving["pos"][2])
            draw.line((cx, cy - 90, cx, cy - 42), fill=(25, 82, 160), width=3)
            draw.rectangle((cx - 60, cy - 112, cx + 60, cy - 90), outline=(25, 82, 160), width=3)
            draw.text((cx - 50, cy - 137), "oracle gripper", fill=(25, 82, 160), font=FONT_SMALL)

    def draw_side_panel(self, draw, step_idx: int, selected: dict | None):
        x0 = 895
        draw.rounded_rectangle((x0, 112, 1228, 620), radius=10, fill=(255, 255, 255), outline=(212, 204, 192), width=2)
        draw.text((x0 + 20, 132), "Paper-style loop", fill=(30, 35, 38), font=FONT_H2)
        lines = [
            "1. Known stone models and poses",
            "2. Sample candidate target poses",
            "3. Simulate release in MuJoCo",
            "4. Score support area and motion",
            "5. Commit the best next stone",
        ]
        y = 176
        for line in lines:
            draw.text((x0 + 20, y), line, fill=(60, 60, 60), font=FONT_SMALL)
            y += 28
        draw.line((x0 + 20, y + 10, x0 + 310, y + 10), fill=(220, 220, 220), width=2)
        y += 28
        draw.text((x0 + 20, y), f"Stack level: {step_idx}", fill=(30, 35, 38), font=FONT_H2)
        y += 42
        draw.text((x0 + 20, y), f"Final height: {self.result['final_height_m']:.3f} m", fill=(30, 35, 38), font=FONT_BODY)
        y += 30
        draw.text((x0 + 20, y), f"Stones stacked: {self.result['stacked_count']}", fill=(30, 35, 38), font=FONT_BODY)
        y += 30
        draw.text((x0 + 20, y), f"Stable: {self.result['stable']}", fill=(30, 35, 38), font=FONT_BODY)
        y += 40
        if selected:
            draw.text((x0 + 20, y), "Selected candidate", fill=(30, 35, 38), font=FONT_H2)
            y += 34
            info = selected["selected"]
            detail = [
                f"stone: {info['stone']}",
                f"cost: {info['cost']:.3f}",
                f"support proxy: {info['support_area_proxy']:.5f} m2",
                f"residual speed: {info['max_speed']:.4f}",
                f"height gain: {info['height_gain_m']:.3f} m",
            ]
            for line in detail:
                draw.text((x0 + 24, y), line, fill=(50, 50, 50), font=FONT_MONO)
                y += 23

    def intro_frame(self):
        img, draw = self.base_canvas()
        draw.text((95, 155), "Short-term demo for advisor", fill=(38, 45, 50), font=FONT_TITLE)
        bullets = [
            "Reproduces the core planning idea from the ICRA stone-stacking paper.",
            "Uses generated flat irregular stones and true object poses.",
            "Searches next best object and target pose by MuJoCo physics rollouts.",
            "Shows a stable 4-stone tower in a fast, reproducible simulation.",
        ]
        y = 220
        for bullet in bullets:
            draw.text((118, y), "- " + bullet, fill=(54, 54, 54), font=FONT_BODY)
            y += 42
        draw.rounded_rectangle((120, 430, 760, 535), radius=12, fill=(255, 255, 255), outline=(210, 202, 190), width=2)
        draw.text((145, 452), f"Result: {self.result['stacked_count']} stones stacked", fill=(20, 70, 40), font=FONT_H2)
        draw.text((145, 488), f"Final height: {self.result['final_height_m']:.3f} m    Stable: {self.result['stable']}", fill=(20, 70, 40), font=FONT_BODY)
        return img

    def render_frames(self, frames_dir: Path, fps: int):
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)
        frame_index = 0

        def save(img):
            nonlocal frame_index
            img.save(frames_dir / f"frame_{frame_index:05d}.png")
            frame_index += 1

        for _ in range(int(2.0 * fps)):
            save(self.intro_frame())

        stack = self.result["stack"]
        decisions = self.result["decisions"]
        committed = [stack[0]]

        for _ in range(int(1.0 * fps)):
            img, draw = self.base_canvas()
            self.draw_stack(draw, committed)
            self.draw_side_panel(draw, 0, None)
            save(img)

        supply_x = -0.31
        for level, decision in enumerate(decisions, start=1):
            moving_entry = stack[level].copy()
            start_z = 0.19 + 0.035 * (level % 2)
            end_pos = moving_entry["pos"]
            hover_pos = [end_pos[0], end_pos[1], end_pos[2] + 0.13]
            start_pos = [supply_x, 0.0, start_z]

            phases = [
                (start_pos, hover_pos, 1.2),
                (hover_pos, end_pos, 0.8),
            ]
            for a, b, duration in phases:
                n = int(duration * fps)
                for i in range(n):
                    t = smooth((i + 1) / n)
                    pos = [lerp(a[j], b[j], t) for j in range(3)]
                    img, draw = self.base_canvas()
                    self.draw_stack(draw, committed, moving={"entry": moving_entry, "pos": pos})
                    self.draw_side_panel(draw, level, decision)
                    save(img)

            committed.append(moving_entry)
            for _ in range(int(0.9 * fps)):
                img, draw = self.base_canvas()
                self.draw_stack(draw, committed)
                self.draw_side_panel(draw, level, decision)
                draw.text((150, 115), "release -> MuJoCo physics settles the dry stack", fill=(42, 96, 56), font=FONT_H2)
                save(img)

        for _ in range(int(2.0 * fps)):
            img, draw = self.base_canvas()
            self.draw_stack(draw, committed)
            self.draw_side_panel(draw, len(committed) - 1, decisions[-1] if decisions else None)
            draw.text((150, 115), "Final stable stack", fill=(42, 96, 56), font=FONT_TITLE)
            save(img)

        return frame_index


def main() -> int:
    args = parse_args()
    result = json.loads(args.input_json.read_text(encoding="utf-8"))
    if "stone_metadata" not in result:
        raise SystemExit("input JSON has no stone_metadata; rerun scripts/run_paper_pose_demo.py first")

    renderer = Renderer(result)
    frame_count = renderer.render_frames(args.frames_dir, args.fps)
    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(args.frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output_mp4),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode
    print(json.dumps({"output_mp4": str(args.output_mp4), "frames": frame_count, "fps": args.fps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

