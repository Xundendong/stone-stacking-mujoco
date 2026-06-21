"""Paper-style limestone meshes for the ICRA 2017 reproduction path.

The paper used six pre-scanned natural lime stones and reduced each mesh to
about 500 faces for pose search. We do not have the original scanned meshes, so
this module creates six deterministic, irregular, mostly-convex limestone
surrogates with similar face count and dry-stacking scale.
"""

from __future__ import annotations

import math
import random

from .rocks import FlatStone


# Six fixed template dimensions in meters. They are sized to fit a Robotiq
# 3-Finger stroke while still being large enough for meaningful contact areas.
LIMESTONE_TEMPLATES: tuple[tuple[float, float, float], ...] = (
    (0.172, 0.116, 0.074),
    (0.158, 0.108, 0.083),
    (0.146, 0.121, 0.069),
    (0.166, 0.101, 0.079),
    (0.136, 0.111, 0.089),
    (0.186, 0.126, 0.067),
)


def generate_paper_limestone(
    name: str,
    rng: random.Random,
    length: float,
    width: float,
    thickness: float,
    radial_rings: int = 4,
    perimeter_vertices: int = 32,
) -> FlatStone:
    """Generate one deterministic paper-style limestone surrogate.

    The mesh is a high-resolution irregular rounded slab: top and bottom are
    triangulated radial surfaces, and the side wall connects their outer rings.
    With the defaults this produces 258 vertices and 512 triangular faces,
    matching the paper's "500 faces" simplification closely enough for the
    physics-search loop.
    """

    if radial_rings < 2:
        raise ValueError("radial_rings must be >= 2")
    if perimeter_vertices < 12:
        raise ValueError("perimeter_vertices must be >= 12")

    harmonics = [
        (rng.uniform(0.025, 0.065), rng.randint(2, 5), rng.uniform(0.0, math.tau)),
        (rng.uniform(0.015, 0.045), rng.randint(5, 9), rng.uniform(0.0, math.tau)),
        (rng.uniform(0.010, 0.030), rng.randint(9, 14), rng.uniform(0.0, math.tau)),
    ]
    outer_scale: list[float] = []
    top_relief: list[float] = []
    bottom_relief: list[float] = []
    for i in range(perimeter_vertices):
        angle = math.tau * i / perimeter_vertices
        scale = 1.0
        for amplitude, frequency, phase in harmonics:
            scale += amplitude * math.sin(frequency * angle + phase)
        scale += rng.uniform(-0.055, 0.055)
        outer_scale.append(max(0.78, min(1.22, scale)))
        top_relief.append(rng.uniform(-0.0018, 0.0018))
        bottom_relief.append(rng.uniform(-0.0018, 0.0018))

    top_center = 0
    bottom_center = 1
    top_z = 0.5 * thickness * rng.uniform(0.94, 1.06)
    bottom_z = -0.5 * thickness * rng.uniform(0.94, 1.06)
    top_shift = (rng.uniform(-0.006, 0.006), rng.uniform(-0.004, 0.004))
    bottom_shift = (rng.uniform(-0.010, 0.010), rng.uniform(-0.008, 0.008))
    vertices: list[tuple[float, float, float]] = [
        (0.0, 0.0, top_z + rng.uniform(-0.002, 0.002)),
        (0.0, 0.0, bottom_z + rng.uniform(-0.002, 0.002)),
    ]

    top_rings: list[list[int]] = []
    bottom_rings: list[list[int]] = []

    for ring in range(1, radial_rings + 1):
        frac = ring / radial_rings
        top_ring: list[int] = []
        bottom_ring: list[int] = []
        for i in range(perimeter_vertices):
            angle = math.tau * i / perimeter_vertices
            scale = outer_scale[i]
            local_scale = scale * (0.95 + 0.06 * frac + rng.uniform(-0.020, 0.020) * (1.0 - frac))
            shear_x = 0.018 * frac * frac * math.sin(2.0 * angle + harmonics[0][2])
            shear_y = 0.015 * frac * frac * math.cos(3.0 * angle + harmonics[1][2])
            x = 0.5 * length * frac * local_scale * math.cos(angle) + shear_x
            y = 0.5 * width * frac * local_scale * math.sin(angle) + shear_y
            top_x = x + top_shift[0] * frac + rng.uniform(-0.002, 0.002) * frac
            top_y = y + top_shift[1] * frac + rng.uniform(-0.002, 0.002) * frac
            bottom_x = x * rng.uniform(0.91, 1.09) + bottom_shift[0] * frac + rng.uniform(-0.004, 0.004) * frac
            bottom_y = y * rng.uniform(0.91, 1.09) + bottom_shift[1] * frac + rng.uniform(-0.004, 0.004) * frac

            # Broad, slightly uneven support surfaces with a chamfered outer
            # edge. MuJoCo uses convex-hull mesh collision, so a domed surface
            # would collapse dry-stack contacts to a point. These plateaus keep
            # the stone irregular without destroying the paper-style support
            # polygon calculation.
            edge = max(0.0, (frac - 0.72) / 0.28)
            relief_weight = 0.35 + 0.65 * frac
            top_surface = top_z + relief_weight * top_relief[i] - 0.0045 * edge * edge + rng.uniform(-0.0009, 0.0009)
            bottom_surface = bottom_z + relief_weight * bottom_relief[i] + 0.0042 * edge * edge + rng.uniform(-0.0009, 0.0009)
            if ring == radial_rings:
                top_surface -= rng.uniform(0.000, 0.0025)
                bottom_surface += rng.uniform(0.000, 0.0025)

            top_ring.append(len(vertices))
            vertices.append((top_x, top_y, top_surface))
            bottom_ring.append(len(vertices))
            vertices.append((bottom_x, bottom_y, bottom_surface))
        top_rings.append(top_ring)
        bottom_rings.append(bottom_ring)

    faces: list[tuple[int, int, int]] = []
    for i in range(perimeter_vertices):
        j = (i + 1) % perimeter_vertices
        faces.append((top_center, top_rings[0][i], top_rings[0][j]))
        faces.append((bottom_center, bottom_rings[0][j], bottom_rings[0][i]))

    for ring in range(radial_rings - 1):
        top_inner = top_rings[ring]
        top_outer = top_rings[ring + 1]
        bottom_inner = bottom_rings[ring]
        bottom_outer = bottom_rings[ring + 1]
        for i in range(perimeter_vertices):
            j = (i + 1) % perimeter_vertices
            faces.append((top_inner[i], top_outer[i], top_outer[j]))
            faces.append((top_inner[i], top_outer[j], top_inner[j]))
            faces.append((bottom_inner[i], bottom_outer[j], bottom_outer[i]))
            faces.append((bottom_inner[i], bottom_inner[j], bottom_outer[j]))

    top_outer = top_rings[-1]
    bottom_outer = bottom_rings[-1]
    for i in range(perimeter_vertices):
        j = (i + 1) % perimeter_vertices
        faces.append((top_outer[i], bottom_outer[i], bottom_outer[j]))
        faces.append((top_outer[i], bottom_outer[j], top_outer[j]))

    density = rng.uniform(2350.0, 2700.0)
    volume_fill = rng.uniform(0.52, 0.64)
    mass = density * length * width * thickness * volume_fill
    shade = rng.uniform(0.44, 0.62)
    rgba = (
        shade + rng.uniform(-0.05, 0.04),
        shade + rng.uniform(-0.04, 0.05),
        shade * rng.uniform(0.82, 0.96),
        1.0,
    )

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


def make_paper_limestones(seed: int = 17) -> list[FlatStone]:
    """Create the six known limestone surrogates used by the reproduction."""

    stones: list[FlatStone] = []
    for index, dims in enumerate(LIMESTONE_TEMPLATES):
        rng = random.Random(seed * 1009 + index * 7919)
        stones.append(generate_paper_limestone(f"limestone_{index + 1:02d}", rng, *dims))
    return stones
