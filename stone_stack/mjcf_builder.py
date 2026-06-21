"""MJCF construction for the flat-stone stacking prototype."""

from __future__ import annotations

from html import escape

from .rocks import FlatStone, flatten_faces, flatten_vertices


def _vec(values) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def build_truth_gripper_scene(
    stones: list[FlatStone],
    body_poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] | None = None,
    include_gripper: bool = True,
) -> str:
    """Build a MuJoCo XML scene with dynamic stones.

    ``body_poses`` maps stone name to ``(pos, quat)`` where quaternion is in
    MuJoCo's ``w x y z`` convention.
    """

    if len(stones) < 2:
        raise ValueError("at least two stones are required")

    mesh_assets = []
    for stone in stones:
        mesh_assets.append(
            f'<mesh name="{escape(stone.name)}_mesh" '
            f'vertex="{flatten_vertices(stone.vertices)}" '
            f'face="{flatten_faces(stone.faces)}"/>'
        )

    stone_bodies = []
    initial_positions = [
        (-0.28, 0.0, 0.12),
        (0.22, 0.0, 0.12),
        (-0.28, 0.20, 0.12),
        (0.22, 0.20, 0.12),
    ]
    for index, stone in enumerate(stones):
        if body_poses is not None and stone.name in body_poses:
            pos, quat = body_poses[stone.name]
            pose_attr = f'pos="{_vec(pos)}" quat="{_vec(quat)}"'
        else:
            pos = initial_positions[index % len(initial_positions)]
            yaw = 0.0 if index < 2 else 0.3 * index
            pose_attr = f'pos="{_vec(pos)}" euler="0 0 {yaw:.6g}"'
        stone_bodies.append(
            f'''
    <body name="{escape(stone.name)}" {pose_attr}>
      <freejoint name="{escape(stone.name)}_free"/>
      <geom name="{escape(stone.name)}_geom" type="mesh" mesh="{escape(stone.name)}_mesh"
            mass="{stone.mass:.6g}" rgba="{_vec(stone.rgba)}"
            friction="1.65 0.035 0.002" condim="4"
            solref="0.006 1" solimp="0.92 0.99 0.001"/>
    </body>'''
        )

    gripper_xml = ""
    if include_gripper:
        gripper_xml = '''
    <body name="palm_marker" mocap="true" pos="0 0 0.35">
      <geom name="palm_visual" type="box" size="0.085 0.012 0.012"
            rgba="0.15 0.35 0.85 0.35" contype="0" conaffinity="0"/>
    </body>

    <body name="left_finger" mocap="true" pos="0 0.09 0.25">
      <geom name="left_finger_pad" type="box" size="0.085 0.014 0.046"
            rgba="0.05 0.08 0.11 1" friction="2.8 0.05 0.002"
            condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    </body>
    <body name="right_finger" mocap="true" pos="0 -0.09 0.25">
      <geom name="right_finger_pad" type="box" size="0.085 0.014 0.046"
            rgba="0.05 0.08 0.11 1" friction="2.8 0.05 0.002"
            condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    </body>'''

    return f'''<mujoco model="truth_flat_stone_stack">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" iterations="80" gravity="0 0 -9.81"/>
  <size nconmax="600" njmax="1200"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.22 0.22 0.22" rgb2="0.30 0.30 0.30"/>
    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.05"/>
    {' '.join(mesh_assets)}
  </asset>

  <worldbody>
    <light name="key" pos="0 -1.2 1.8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="0 -1.05 0.72" xyaxes="1 0 0 0 0.45 0.89"/>
    <geom name="floor" type="plane" size="1.0 1.0 0.05" material="floor_mat"
          friction="1.4 0.025 0.001" condim="4"/>

    {gripper_xml}

    {''.join(stone_bodies)}
  </worldbody>
</mujoco>
'''
