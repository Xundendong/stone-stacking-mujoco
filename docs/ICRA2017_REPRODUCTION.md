# ICRA 2017 Stone Stacking Reproduction Notes

Target paper:

`Autonomous Robotic Stone Stacking with Online next Best Object Target Pose Planning`, ICRA 2017.

## What The Current Demo Reproduces

The current MuJoCo demo focuses on the short-term result needed for a lab demo:

- six known limestone-like stones, matching the paper's six pre-scanned objects;
- each generated stone has 258 vertices and 512 triangular faces, close to the paper's 500-face simplified mesh;
- a subset of four stones is selected for one stacking run;
- stone friction defaults to `mu_stone = 0.1`, matching the paper;
- candidate placement uses the previous top stone pose and its local support normal;
- each available object gets `x = 5` randomized local pose searches by default;
- MuJoCo contacts are collected over 10 simulation steps;
- contact points are projected with PCA to estimate support polygon area;
- cost uses the paper weights:
  - `w1 = 0.179` for inverse support area;
  - `w2 = 0.472` for kinetic energy;
  - `w3 = 0.094` for center distance;
  - `w4 = 0.255` for normal deviation;
- validity checks include at least 3 contacts, `Amin = 1e-5 m^2`, CoM projection inside the support polygon, and `Ekin <= 20 J`;
- the final stack is saved as a MuJoCo MJCF scene with local UR10 visual meshes and local Robotiq S-Model three-finger visual meshes.

Run:

```bash
cd /home/xunden/stone-stacking-mujoco
source .venv/bin/activate
python scripts/run_icra2017_repro.py
```

or:

```bash
./scripts/make_icra2017_repro_demo.sh
```

Expected default result at the time of writing:

```text
base=limestone_06 subset=['limestone_05', 'limestone_06', 'limestone_01', 'limestone_02']
level=1 selected=limestone_01 valid=True cost=96.6646 area=0.00185177 contacts=27 height_gain=0.0768
level=2 selected=limestone_05 valid=True cost=151.693 area=0.00118002 contacts=32 height_gain=0.0844
level=3 selected=limestone_02 valid=True cost=297.2 area=0.000602294 contacts=33 height_gain=0.0921
stacked_count=4
final_height_m=0.33093837056134756
```

Outputs:

- `reports/icra2017_repro.json`
- `outputs/icra2017_repro_final.xml`
- `outputs/icra2017_repro_mujoco.mp4`
- `outputs/icra2017_repro_overview.png`
- `outputs/icra2017_repro_robot_view.png`

Render the MP4:

```bash
python scripts/render_icra2017_repro_video.py
```

Open the resulting MuJoCo scene:

```bash
python scripts/view_icra2017_repro.py
```

Use this viewer for screen recording. It loads the real generated MJCF scene;
it does not play the older explanatory animation.

## What Is Still Approximate

This is not yet a full one-to-one hardware reproduction.

The original paper used:

- real scanned lime stone meshes;
- measured mass, center of mass, and inertia for each stone;
- UR10 arm;
- Robotiq 3-Finger gripper;
- FT150 force-torque sensor;
- Intel RealSense SR300;
- MoveIt collision-free motion planning;
- object detection and pose estimation from RGB-D point clouds.

The current demo uses:

- procedural limestone surrogates because the original scanned stone meshes are not available in the folder;
- approximate mass from limestone density and generated volume;
- local UR10 visual meshes from the Isaac URDF asset directory, assembled in a fixed display pose;
- local Robotiq S-Model three-finger visual meshes from robosuite, mounted at the UR10 wrist;
- truth-state object poses and direct placement into candidate target poses for the physics search;
- no RGB-D detection, no MoveIt, and no real arm IK yet.

This distinction matters: the current demo is a valid short-term reproduction of the paper's core next-best pose planning loop in MuJoCo, but it is not yet a complete reproduction of the physical robot system.

## Next Engineering Steps

1. Replace procedural limestone surrogates with scanned stone meshes.
2. Add measured mass, CoM, and inertia for each real stone.
3. Convert or rebuild the UR10 as a valid MuJoCo articulated dynamics model.
4. Connect the Robotiq S-Model visual mesh to actuated gripper joints in the full arm model.
5. Add grasp candidates and IK-driven pick/place motion.
6. Add contact or force-triggered release instead of direct truth placement.
7. Add perception later: point cloud pose estimation, ICP, and online re-planning from measured post-placement poses.
