"""Synthetic irregular rocks following the From Rocks to Walls generator.

The CVPRW 2021 paper "From Rocks to Walls" generates irregular rocks by
starting from a rectangular prism, repeatedly subdividing the mesh, perturbing
vertices with truncated-normal noise, and using the convex hull as the final
collision-friendly rock. This module implements that procedure for the MuJoCo
demo while keeping the final oriented bounding box at a controlled scale.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm
import trimesh

from .rocks import FlatStone


# Demo-scale rectangular prisms in meters. These are intentionally close to the
# existing ICRA limestone surrogates so the pose-search demo remains feasible,
# while the mesh generation follows the "From Rocks to Walls" procedure.
ROCK_WALL_TEMPLATES: tuple[tuple[float, float, float], ...] = (
    (0.176, 0.118, 0.078),
    (0.160, 0.112, 0.088),
    (0.150, 0.126, 0.072),
    (0.168, 0.106, 0.082),
    (0.140, 0.114, 0.092),
    (0.190, 0.130, 0.070),
)


def _truncated_normal_offsets(
    rng: np.random.Generator,
    count: int,
    scale: np.ndarray,
    truncation: float = 2.0,
) -> np.ndarray:
    values = truncnorm.rvs(
        -truncation,
        truncation,
        loc=0.0,
        scale=1.0,
        size=(count, 3),
        random_state=rng,
    )
    return np.asarray(values, dtype=float) * scale


def _align_to_target_obb(mesh: trimesh.Trimesh, target_extents: np.ndarray) -> trimesh.Trimesh:
    """Center the rock on its OBB and align longest/middle/shortest axes.

    Trimesh gives an oriented bounding box transform, but the returned axis
    order is not guaranteed. The paper stores each mesh with its OBB centered at
    the model origin, largest axis first and smallest axis third; we apply that
    convention here and then rescale the final OBB to the requested dimensions.
    """

    obb = mesh.bounding_box_oriented
    vertices = trimesh.transform_points(mesh.vertices, np.linalg.inv(obb.primitive.transform))
    extents = np.ptp(vertices, axis=0)
    order = np.argsort(extents)[::-1]
    vertices = vertices[:, order]

    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    vertices = vertices - 0.5 * (lower + upper)
    extents = np.maximum(np.ptp(vertices, axis=0), 1.0e-9)
    vertices = vertices * (target_extents / extents)

    aligned = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=True)
    aligned.fix_normals()
    return aligned


def generate_rock_wall_stone(
    name: str,
    seed: int,
    length: float,
    width: float,
    thickness: float,
    irregularity: float = 0.75,
    subdivisions: int = 5,
    density_range: tuple[float, float] = (1800.0, 2700.0),
) -> FlatStone:
    """Generate one convex-hull irregular rock.

    ``irregularity`` corresponds to the paper's zeta-like shape parameter. The
    paper uses values from 0.5 to 1.0; lower values stay closer to the source
    prism, higher values produce more faceted, uneven rocks.
    """

    if not 0.0 < irregularity <= 1.5:
        raise ValueError("irregularity should be in (0, 1.5]")
    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")
    if density_range[0] <= 0.0 or density_range[1] <= density_range[0]:
        raise ValueError("density_range must be positive and increasing")

    rng = np.random.default_rng(seed)
    target_extents = np.array([length, width, thickness], dtype=float)
    mesh = trimesh.creation.box(extents=target_extents)

    base_scale = target_extents * (0.09 * irregularity)
    mesh.vertices = mesh.vertices + _truncated_normal_offsets(rng, len(mesh.vertices), base_scale)

    for level in range(subdivisions):
        old_vertex_count = len(mesh.vertices)
        mesh = mesh.subdivide()
        vertices = mesh.vertices.copy()
        scale = base_scale / (2.0 ** (level + 1))
        new_vertex_count = len(vertices) - old_vertex_count
        if new_vertex_count > 0:
            vertices[old_vertex_count:] = (
                vertices[old_vertex_count:]
                + _truncated_normal_offsets(rng, new_vertex_count, scale)
            )
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    hull = mesh.convex_hull
    hull.fix_normals()
    aligned = _align_to_target_obb(hull, target_extents)
    volume = max(abs(float(aligned.volume)), 1.0e-9)
    density = float(rng.uniform(*density_range))
    mass = density * volume

    shade = float(rng.uniform(0.34, 0.58))
    warmth = float(rng.uniform(-0.035, 0.045))
    rgba = (
        min(0.72, max(0.22, shade + warmth)),
        min(0.70, max(0.22, shade + 0.5 * warmth)),
        min(0.64, max(0.18, shade * rng.uniform(0.78, 0.96))),
        1.0,
    )

    return FlatStone(
        name=name,
        vertices=[tuple(map(float, vertex)) for vertex in aligned.vertices],
        faces=[tuple(map(int, face)) for face in aligned.faces],
        rgba=rgba,
        mass=mass,
        length=length,
        width=width,
        thickness=thickness,
    )


def generate_natural_wall_rock(
    name: str,
    seed: int,
    length: float,
    width: float,
    thickness: float,
    irregularity: float = 1.0,
    subdivisions: int = 5,
    density_range: tuple[float, float] = (1900.0, 2750.0),
) -> FlatStone:
    """Generate a more natural-looking dry-wall rock.

    This is still compatible with the paper's synthetic-rock idea: a box is
    repeatedly subdivided, vertices are displaced with truncated-normal noise,
    and the final rock is a convex hull. The difference from
    ``generate_rock_wall_stone`` is that every subdivision level perturbs all
    vertices, not just newly inserted ones. That breaks up large planar box
    faces and produces faceted stones that look closer to natural wall rocks.
    """

    if not 0.0 < irregularity <= 1.5:
        raise ValueError("irregularity should be in (0, 1.5]")
    if subdivisions < 1:
        raise ValueError("subdivisions must be at least 1")

    rng = np.random.default_rng(seed)
    target_extents = np.array([length, width, thickness], dtype=float)
    mesh = trimesh.creation.box(extents=target_extents)

    # Keep the long/side faces visibly irregular, but avoid over-perturbing the
    # bedding direction. Natural dry-wall stones are rough, not spherical; they
    # still need broad-ish upper/lower faces to be stackable by contact.
    base_scale = target_extents * np.array([0.12, 0.13, 0.095]) * irregularity
    vertices = mesh.vertices.copy()
    vertices = vertices + _truncated_normal_offsets(rng, len(vertices), base_scale)
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    for level in range(subdivisions):
        mesh = mesh.subdivide()
        vertices = mesh.vertices.copy()
        scale = base_scale / (1.65 ** (level + 1))
        offsets = _truncated_normal_offsets(rng, len(vertices), scale)

        # Add very low-frequency surface relief. This keeps the stones from
        # reading as chamfered boxes after the convex hull step.
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        relief = (
            np.sin(17.0 * vertices[:, 0] / max(length, 1.0e-9) + phase[0])
            + 0.7 * np.cos(19.0 * vertices[:, 1] / max(width, 1.0e-9) + phase[1])
            + 0.5 * np.sin(23.0 * vertices[:, 2] / max(thickness, 1.0e-9) + phase[2])
        )
        offsets += vertices * (0.010 * irregularity * relief[:, None] / (level + 1))
        vertices = vertices + offsets
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    hull = mesh.convex_hull
    hull.fix_normals()
    aligned = _align_to_target_obb(hull, target_extents)
    volume = max(abs(float(aligned.volume)), 1.0e-9)
    density = float(rng.uniform(*density_range))
    mass = density * volume

    shade = float(rng.uniform(0.34, 0.58))
    warmth = float(rng.uniform(-0.04, 0.08))
    rgba = (
        min(0.72, max(0.22, shade + warmth + rng.uniform(-0.025, 0.025))),
        min(0.70, max(0.22, shade + 0.45 * warmth + rng.uniform(-0.025, 0.025))),
        min(0.66, max(0.20, shade - 0.030 + rng.uniform(-0.035, 0.020))),
        1.0,
    )

    return FlatStone(
        name=name,
        vertices=[tuple(map(float, vertex)) for vertex in aligned.vertices],
        faces=[tuple(map(int, face)) for face in aligned.faces],
        rgba=rgba,
        mass=mass,
        length=length,
        width=width,
        thickness=thickness,
    )


def make_rock_wall_stones(
    seed: int = 17,
    count: int = 6,
    irregularity: float = 0.75,
    subdivisions: int = 5,
    style: str = "paper",
) -> list[FlatStone]:
    """Create a deterministic set of From-Rocks-to-Walls-style stones."""

    if count <= 0:
        raise ValueError("count must be positive")
    normalized_style = style.strip().lower().replace("_", "-")
    if normalized_style in {"natural", "rough", "wall-rocks"}:
        return make_natural_wall_rocks(
            seed=seed,
            count=count,
            irregularity=irregularity,
            subdivisions=subdivisions,
        )
    if normalized_style not in {"paper", "from-rocks-to-walls", "convex"}:
        raise ValueError("style must be one of: paper, natural")

    stones: list[FlatStone] = []
    dim_rng = np.random.default_rng(seed * 6151 + 97)
    for index in range(count):
        if index < len(ROCK_WALL_TEMPLATES):
            dims = ROCK_WALL_TEMPLATES[index]
        else:
            length = float(dim_rng.uniform(0.135, 0.195))
            width = float(dim_rng.uniform(0.095, 0.135))
            thickness = float(dim_rng.uniform(0.062, 0.095))
            dims = (length, width, thickness)

        stone_seed = seed * 100_003 + index * 9_973 + 31
        stones.append(
            generate_rock_wall_stone(
                f"rock_wall_{index + 1:02d}",
                stone_seed,
                *dims,
                irregularity=irregularity,
                subdivisions=subdivisions,
            )
        )
    return stones


def make_natural_wall_rocks(
    seed: int = 23,
    count: int = 9,
    irregularity: float = 1.0,
    subdivisions: int = 5,
) -> list[FlatStone]:
    """Create realistic, flat wall rocks for dry-stacking experiments."""

    if count <= 0:
        raise ValueError("count must be positive")

    rng = np.random.default_rng(seed * 7757 + 211)
    stones: list[FlatStone] = []
    for index in range(count):
        length = float(rng.uniform(0.150, 0.220))
        width = float(rng.uniform(0.095, 0.138))
        thickness = float(rng.uniform(0.062, 0.095))
        if index < 4:
            length *= 1.08
            width *= 1.04
            thickness *= 1.05
        elif index >= 7:
            length *= 0.88
            width *= 0.98
            thickness *= 0.97

        stone_seed = seed * 150_001 + index * 12_989 + 53
        stones.append(
            generate_natural_wall_rock(
                f"wall_rock_{index + 1:02d}",
                stone_seed,
                length,
                width,
                thickness,
                irregularity=irregularity,
                subdivisions=subdivisions,
            )
        )
    return stones
