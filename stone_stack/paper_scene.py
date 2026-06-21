"""MJCF scene builder for the ICRA 2017 stone-stacking reproduction."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .rocks import FlatStone, flatten_faces, flatten_vertices


UR10_MESH_DIR = Path(
    "/home/xunden/isaacsim/extscache/"
    "isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310/"
    "data/urdf/robots/ur10/meshes"
)

UR10_MESH_FILES = {
    "base": "ur10_base.obj",
    "shoulder": "ur10_shoulder.obj",
    "upper_arm": "ur10_upper_arm.obj",
    "forearm": "ur10_forearm.obj",
    "wrist_1": "ur10_wrist_1.obj",
    "wrist_2": "ur10_wrist_2.obj",
    "wrist_3": "ur10_wrist_3.obj",
}

ROBOTIQ_MESH_DIR = Path(
    "/home/xunden/isaac-sim/kit/python/lib/python3.11/site-packages/"
    "robosuite/models/assets/grippers/meshes/robotiq_s_gripper"
)

ROBOTIQ_MESH_FILES = {
    "palm_vis": "palm_vis.stl",
    "link_0_vis": "link_0_vis.stl",
    "link_1_vis": "link_1_vis.stl",
    "link_2_vis": "link_2_vis.stl",
    "link_3_vis": "link_3_vis.stl",
}


def _vec(values) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _stone_supply_pose(index: int, stone: FlatStone) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    x = -0.20 + 0.20 * (index % 3)
    y = 0.34 + 0.16 * (index // 3)
    z = 0.5 * stone.thickness + 0.015
    return (x, y, z), (1.0, 0.0, 0.0, 0.0)


def _has_ur10_meshes() -> bool:
    return all((UR10_MESH_DIR / file_name).is_file() for file_name in UR10_MESH_FILES.values())


def _has_robotiq_meshes() -> bool:
    return all((ROBOTIQ_MESH_DIR / file_name).is_file() for file_name in ROBOTIQ_MESH_FILES.values())


def _robot_asset_xml() -> str:
    return """
    <material name="ur10_white" rgba="0.86 0.88 0.88 1"/>
    <material name="ur10_blue" rgba="0.05 0.25 0.58 1"/>
    <material name="ur10_dark" rgba="0.08 0.09 0.10 1"/>
    <material name="ur10_metal" rgba="0.64 0.64 0.60 1"/>
"""


def _robotiq_s_model_visual_xml() -> str:
    if not _has_robotiq_meshes():
        return """
                    <body name="robotiq_3finger_visual" pos="0 0 0.075">
                      <geom name="robotiq_3finger_palm" type="cylinder" pos="0 0 0"
                            euler="1.5708 0 0" size="0.060 0.026"
                            material="ur10_dark" contype="0" conaffinity="0"/>
                      <geom name="robotiq_3finger_knuckle_a" type="box" pos="0 -0.050 -0.060"
                            euler="0.22 0 0" size="0.016 0.012 0.050"
                            material="ur10_metal" contype="0" conaffinity="0"/>
                      <geom name="robotiq_3finger_knuckle_b" type="box" pos="-0.043 0.027 -0.060"
                            euler="-0.20 0 0.70" size="0.016 0.012 0.050"
                            material="ur10_metal" contype="0" conaffinity="0"/>
                      <geom name="robotiq_3finger_knuckle_c" type="box" pos="0.043 0.027 -0.060"
                            euler="-0.20 0 -0.70" size="0.016 0.012 0.050"
                            material="ur10_metal" contype="0" conaffinity="0"/>
                      <site name="tool0" pos="0 0 -0.155" size="0.012" rgba="1 0.1 0.1 1"/>
                    </body>
"""

    return """
                    <body name="robotiq_s_palm_visual" pos="0 0 0.045"
                          quat="-0.49921826 -0.50133955 0.50133955 0.49921826">
                      <geom name="robotiq_s_palm_mesh_geom" type="mesh" mesh="robotiq_s_palm_vis_mesh"
                            material="ur10_dark" contype="0" conaffinity="0"/>
                      <site name="tool0" pos="0 0.15 0" size="0.012" rgba="1 0.1 0.1 1"/>

                      <body name="robotiq_s_finger_1_l0_visual" pos="-0.0455 0.0214 0.036"
                            quat="-2.59838e-06 0.706825 0.707388 2.59631e-06">
                        <geom name="robotiq_s_f1_l0_mesh_geom" type="mesh" mesh="robotiq_s_link_0_vis_mesh"
                              pos="0.02 0 0" material="ur10_dark" contype="0" conaffinity="0"/>
                        <body name="robotiq_s_finger_1_l1_visual" pos="0.02 0 0">
                          <geom name="robotiq_s_f1_l1_mesh_geom" type="mesh" mesh="robotiq_s_link_1_vis_mesh"
                                pos="0.05 -0.028 0" quat="0.96639 0 0 -0.257081"
                                material="ur10_dark" contype="0" conaffinity="0"/>
                          <body name="robotiq_s_finger_1_l2_visual" pos="0.05 -0.028 0"
                                quat="0.96639 0 0 -0.257081">
                            <geom name="robotiq_s_f1_l2_mesh_geom" type="mesh" mesh="robotiq_s_link_2_vis_mesh"
                                  pos="0.039 0 0.0075" material="ur10_dark"
                                  contype="0" conaffinity="0"/>
                            <body name="robotiq_s_finger_1_l3_visual" pos="0.039 0 0">
                              <geom name="robotiq_s_f1_l3_mesh_geom" type="mesh" mesh="robotiq_s_link_3_vis_mesh"
                                    quat="0.96639 0 0 0.257081" material="ur10_white"
                                    contype="0" conaffinity="0"/>
                            </body>
                          </body>
                        </body>
                      </body>

                      <body name="robotiq_s_finger_2_l0_visual" pos="-0.0455 0.0214 -0.036"
                            quat="-2.59838e-06 0.706825 0.707388 2.59631e-06">
                        <geom name="robotiq_s_f2_l0_mesh_geom" type="mesh" mesh="robotiq_s_link_0_vis_mesh"
                              pos="0.02 0 0" material="ur10_dark" contype="0" conaffinity="0"/>
                        <body name="robotiq_s_finger_2_l1_visual" pos="0.02 0 0">
                          <geom name="robotiq_s_f2_l1_mesh_geom" type="mesh" mesh="robotiq_s_link_1_vis_mesh"
                                pos="0.05 -0.028 0" quat="0.96639 0 0 -0.257081"
                                material="ur10_dark" contype="0" conaffinity="0"/>
                          <body name="robotiq_s_finger_2_l2_visual" pos="0.05 -0.028 0"
                                quat="0.96639 0 0 -0.257081">
                            <geom name="robotiq_s_f2_l2_mesh_geom" type="mesh" mesh="robotiq_s_link_2_vis_mesh"
                                  pos="0.039 0 0.0075" material="ur10_dark"
                                  contype="0" conaffinity="0"/>
                            <body name="robotiq_s_finger_2_l3_visual" pos="0.039 0 0">
                              <geom name="robotiq_s_f2_l3_mesh_geom" type="mesh" mesh="robotiq_s_link_3_vis_mesh"
                                    quat="0.96639 0 0 0.257081" material="ur10_white"
                                    contype="0" conaffinity="0"/>
                            </body>
                          </body>
                        </body>
                      </body>

                      <body name="robotiq_s_middle_l0_visual" pos="0.0455 0.0214 0"
                            quat="0.707388 0 0 0.706825">
                        <geom name="robotiq_s_f3_l0_mesh_geom" type="mesh" mesh="robotiq_s_link_0_vis_mesh"
                              pos="0.02 0 0" material="ur10_dark" contype="0" conaffinity="0"/>
                        <body name="robotiq_s_middle_l1_visual" pos="0.02 0 0">
                          <geom name="robotiq_s_f3_l1_mesh_geom" type="mesh" mesh="robotiq_s_link_1_vis_mesh"
                                pos="0.05 -0.028 0" quat="0.96639 0 0 -0.257081"
                                material="ur10_dark" contype="0" conaffinity="0"/>
                          <body name="robotiq_s_middle_l2_visual" pos="0.05 -0.028 0"
                                quat="0.96639 0 0 -0.257081">
                            <geom name="robotiq_s_f3_l2_mesh_geom" type="mesh" mesh="robotiq_s_link_2_vis_mesh"
                                  pos="0.039 0 0.0075" material="ur10_dark"
                                  contype="0" conaffinity="0"/>
                            <body name="robotiq_s_middle_l3_visual" pos="0.039 0 0">
                              <geom name="robotiq_s_f3_l3_mesh_geom" type="mesh" mesh="robotiq_s_link_3_vis_mesh"
                                    quat="0.96639 0 0 0.257081" material="ur10_white"
                                    contype="0" conaffinity="0"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
"""


def _fallback_robot_visual_xml() -> str:
    return """
    <body name="ur10_robot_visual">
      <geom name="ur10_base_link" type="cylinder" pos="-0.62 0 0.055"
            size="0.095 0.055" material="ur10_dark"
            contype="0" conaffinity="0"/>
      <geom name="ur10_shoulder_link" type="capsule"
            fromto="-0.62 0 0.12 -0.62 0 0.27" size="0.07"
            material="ur10_blue" contype="0" conaffinity="0"/>
      <geom name="ur10_upper_arm_link" type="capsule"
            fromto="-0.62 0 0.27 -0.31 0 0.62" size="0.052"
            material="ur10_blue" contype="0" conaffinity="0"/>
      <geom name="ur10_forearm_link" type="capsule"
            fromto="-0.31 0 0.62 0.02 0 0.42" size="0.045"
            material="ur10_blue" contype="0" conaffinity="0"/>
      <geom name="ur10_wrist_1_link" type="sphere" pos="0.02 0 0.42"
            size="0.060" material="ur10_dark" contype="0" conaffinity="0"/>
      <geom name="ur10_wrist_2_link" type="capsule"
            fromto="0.02 0 0.42 0.02 0 0.30" size="0.038"
            material="ur10_dark" contype="0" conaffinity="0"/>
    </body>
"""


def _robot_visual_xml() -> str:
    """Return a clean UR10-scale visual rig with a three-finger gripper."""

    return """
    <body name="ur10_robot_visual">
      <geom name="ur10_base" type="cylinder" pos="-0.54 -0.36 0.070"
            size="0.115 0.070" material="ur10_dark" contype="0" conaffinity="0"/>
      <geom name="ur10_base_ring" type="cylinder" pos="-0.54 -0.36 0.145"
            size="0.090 0.025" material="ur10_blue" contype="0" conaffinity="0"/>

      <geom name="ur10_shoulder_joint" type="cylinder" pos="-0.54 -0.36 0.245"
            euler="1.5708 0 0" size="0.085 0.070"
            material="ur10_blue" contype="0" conaffinity="0"/>
      <geom name="ur10_shoulder_cap" type="sphere" pos="-0.54 -0.36 0.245"
            size="0.087" material="ur10_blue" contype="0" conaffinity="0"/>

      <geom name="ur10_upper_arm" type="capsule"
            fromto="-0.50 -0.32 0.285 -0.33 -0.14 0.585"
            size="0.052" material="ur10_white" contype="0" conaffinity="0"/>
      <geom name="ur10_upper_arm_shadow_side" type="capsule"
            fromto="-0.57 -0.34 0.280 -0.40 -0.16 0.580"
            size="0.032" material="ur10_blue" contype="0" conaffinity="0"/>

      <geom name="ur10_elbow_joint" type="sphere" pos="-0.33 -0.14 0.585"
            size="0.078" material="ur10_blue" contype="0" conaffinity="0"/>
      <geom name="ur10_forearm" type="capsule"
            fromto="-0.33 -0.14 0.585 -0.06 -0.045 0.410"
            size="0.045" material="ur10_white" contype="0" conaffinity="0"/>
      <geom name="ur10_forearm_side" type="capsule"
            fromto="-0.37 -0.12 0.565 -0.10 -0.025 0.390"
            size="0.027" material="ur10_blue" contype="0" conaffinity="0"/>

      <geom name="ur10_wrist_1" type="cylinder" pos="-0.06 -0.045 0.410"
            euler="0.35 1.5708 0" size="0.055 0.050"
            material="ur10_dark" contype="0" conaffinity="0"/>
      <geom name="ur10_wrist_2" type="cylinder" pos="-0.010 -0.025 0.365"
            euler="1.5708 0.35 0" size="0.045 0.045"
            material="ur10_blue" contype="0" conaffinity="0"/>
      <geom name="ur10_wrist_3" type="cylinder" pos="0.030 -0.010 0.330"
            euler="0.30 1.5708 0" size="0.042 0.040"
            material="ur10_dark" contype="0" conaffinity="0"/>

      <geom name="ft150_sensor_visual" type="cylinder" pos="0.065 0.000 0.305"
            euler="0.30 1.5708 0" size="0.048 0.018"
            material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="robotiq_3finger_palm" type="cylinder" pos="0.095 0.012 0.282"
            euler="0.30 1.5708 0" size="0.060 0.030"
            material="ur10_dark" contype="0" conaffinity="0"/>

      <geom name="robotiq_3finger_finger_a" type="capsule"
            fromto="0.105 -0.042 0.262 0.090 -0.090 0.198"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="robotiq_3finger_tip_a" type="box" pos="0.083 -0.104 0.180"
            euler="0.45 0.20 0.08" size="0.017 0.010 0.024"
            material="ur10_dark" contype="0" conaffinity="0"/>
      <geom name="robotiq_3finger_finger_b" type="capsule"
            fromto="0.082 0.050 0.260 0.052 0.090 0.198"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="robotiq_3finger_tip_b" type="box" pos="0.042 0.102 0.180"
            euler="-0.45 0.20 0.35" size="0.017 0.010 0.024"
            material="ur10_dark" contype="0" conaffinity="0"/>
      <geom name="robotiq_3finger_finger_c" type="capsule"
            fromto="0.145 0.030 0.260 0.178 0.064 0.198"
            size="0.012" material="ur10_metal" contype="0" conaffinity="0"/>
      <geom name="robotiq_3finger_tip_c" type="box" pos="0.190 0.072 0.180"
            euler="-0.45 -0.20 -0.35" size="0.017 0.010 0.024"
            material="ur10_dark" contype="0" conaffinity="0"/>
      <site name="tool0" pos="0.105 0.000 0.175" size="0.012" rgba="1 0.1 0.1 1"/>
    </body>
"""


def build_icra2017_scene(
    stones: list[FlatStone],
    body_poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] | None = None,
    fixed_stones: set[str] | None = None,
    include_robot: bool = True,
    stone_friction: float = 0.1,
    model_name: str = "icra2017_stone_stack_repro",
) -> str:
    """Build a MuJoCo scene for paper-style stone stacking.

    ``body_poses`` maps stone name to ``(pos, quat)`` in MuJoCo ``w x y z``
    convention. Stones listed in ``fixed_stones`` are welded to the world, which
    mirrors the paper's pose-search phase where the existing stack is immobile.
    """

    fixed_stones = fixed_stones or set()
    body_poses = body_poses or {}

    mesh_assets = []
    for stone in stones:
        mesh_assets.append(
            f'<mesh name="{escape(stone.name)}_mesh" '
            f'vertex="{flatten_vertices(stone.vertices)}" '
            f'face="{flatten_faces(stone.faces)}"/>'
        )

    stone_bodies = []
    for index, stone in enumerate(stones):
        if stone.name in body_poses:
            pos, quat = body_poses[stone.name]
        else:
            pos, quat = _stone_supply_pose(index, stone)
        fixed = stone.name in fixed_stones
        joint_xml = "" if fixed else f'<freejoint name="{escape(stone.name)}_free"/>'
        mass_attr = "" if fixed else f'mass="{stone.mass:.6g}"'
        stone_bodies.append(
            f'''
    <body name="{escape(stone.name)}" pos="{_vec(pos)}" quat="{_vec(quat)}">
      {joint_xml}
      <geom name="{escape(stone.name)}_geom" type="mesh" mesh="{escape(stone.name)}_mesh"
            {mass_attr} rgba="{_vec(stone.rgba)}"
            friction="{stone_friction:.6g} 0.004 0.0001" condim="4"
            solref="0.006 1" solimp="0.90 0.99 0.001"/>
    </body>'''
        )

    robot_xml = _robot_visual_xml() if include_robot else ""
    return f'''<mujoco model="{escape(model_name)}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0025" integrator="implicitfast" cone="elliptic"
          iterations="120" gravity="0 0 -9.81"/>
  <size nconmax="1200" njmax="2400"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <map force="0.15"/>
  </visual>

  <asset>
    <texture name="table_grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.55 0.55 0.52" rgb2="0.45 0.45 0.42"/>
    <material name="table_mat" texture="table_grid" texrepeat="4 4" reflectance="0.04"/>
    {_robot_asset_xml()}
    {' '.join(mesh_assets)}
  </asset>

  <worldbody>
    <light name="key" pos="-0.5 -1.0 1.6" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="0.6 0.8 1.2" dir="0 0 -1" diffuse="0.35 0.35 0.35"/>
    <camera name="overview" pos="0.08 -1.14 0.72" xyaxes="1 0 0 0 0.50 0.87"/>
    <camera name="robot_view" pos="-0.82 -0.92 0.64" xyaxes="0.84 -0.54 0 0.29 0.45 0.84"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="0.85 0.65 0.025"
          material="table_mat" friction="0.6 0.01 0.0001" condim="4"/>

    {robot_xml}

    {''.join(stone_bodies)}
  </worldbody>
</mujoco>
'''
