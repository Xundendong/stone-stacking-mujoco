"""Procedural flat stone meshes for MuJoCo experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class FlatStone:
    """A low-poly, mostly-convex, flattened stone mesh."""

    name: str
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    rgba: tuple[float, float, float, float]
    mass: float
    length: float
    width: float
    thickness: float


def _fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def flatten_vertices(vertices: list[tuple[float, float, float]]) -> str:
    """Serialize vertices for MJCF ``mesh vertex=...``."""

    return _fmt([coord for vertex in vertices for coord in vertex])


def flatten_faces(faces: list[tuple[int, int, int]]) -> str:
    """Serialize triangular faces for MJCF ``mesh face=...``."""

    return " ".join(str(index) for face in faces for index in face)


def generate_flat_stone(
    name: str,
    rng: random.Random,
    length: float | None = None,
    width: float | None = None,
    thickness: float | None = None,
    perimeter_vertices: int = 14,
) -> FlatStone:
    """Generate a dry-stacking-friendly flattened irregular convex stone.

    The shape is intentionally biased toward a flat rounded slab. MuJoCo uses a
    convex hull for mesh collision, so this generator keeps the visual shape
    close to its collision shape.
    """

    if perimeter_vertices < 8:
        raise ValueError("perimeter_vertices must be >= 8")

    length = length if length is not None else rng.uniform(0.18, 0.26)
    width = width if width is not None else rng.uniform(0.105, 0.155)
    thickness = thickness if thickness is not None else rng.uniform(0.032, 0.052)

    top_center_index = 0
    bottom_center_index = 1
    vertices: list[tuple[float, float, float]] = [
        (0.0, 0.0, thickness * 0.5 * rng.uniform(0.92, 1.04)),
        (0.0, 0.0, -thickness * 0.5 * rng.uniform(0.92, 1.04)),
    ]

    top_ring: list[int] = []
    bottom_ring: list[int] = []

    for i in range(perimeter_vertices):
        angle = 2.0 * math.pi * i / perimeter_vertices
        # Superellipse-like outline with local dents, still mostly convex.
        radial = rng.uniform(0.88, 1.10)
        x = 0.5 * length * radial * math.cos(angle)
        y = 0.5 * width * radial * math.sin(angle)
        top_z = 0.5 * thickness * rng.uniform(0.84, 1.10)
        bottom_z = -0.5 * thickness * rng.uniform(0.84, 1.10)

        top_ring.append(len(vertices))
        vertices.append((x, y, top_z))
        bottom_ring.append(len(vertices))
        vertices.append((x * rng.uniform(0.97, 1.03), y * rng.uniform(0.97, 1.03), bottom_z))

    faces: list[tuple[int, int, int]] = []
    for i in range(perimeter_vertices):
        j = (i + 1) % perimeter_vertices
        top_i = top_ring[i]
        top_j = top_ring[j]
        bottom_i = bottom_ring[i]
        bottom_j = bottom_ring[j]

        faces.append((top_center_index, top_i, top_j))
        faces.append((bottom_center_index, bottom_j, bottom_i))
        faces.append((top_i, bottom_i, bottom_j))
        faces.append((top_i, bottom_j, top_j))

    rgba = (
        rng.uniform(0.33, 0.48),
        rng.uniform(0.30, 0.42),
        rng.uniform(0.25, 0.35),
        1.0,
    )
    # Use lightweight lab stones for the first grasping prototype. Heavier
    # natural stones can be introduced after the contact-grasp baseline works.
    density = rng.uniform(550.0, 900.0)
    mass = density * length * width * thickness * 0.72
    return FlatStone(
        name=name,
        vertices=vertices,
        faces=faces,
        rgba=rgba,
        mass=mass,
        length=length,
        width=width,
        thickness=thickness,
    )


def make_flat_stones(count: int, seed: int) -> list[FlatStone]:
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    return [generate_flat_stone(f"stone_{index:02d}", rng) for index in range(count)]
