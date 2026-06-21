# Hybrid ICRA Wall Reproduction

This is the active reproduction path for the project.

The goal is not to reproduce the DQN policy from **From Rocks to Walls**.
Instead, we use that paper only for synthetic irregular rock generation, and
use the **ICRA 2017** next-best object/target-pose planning idea for robotic
dry stacking.

## Scope

Stone generation follows **From Rocks to Walls**:

```text
rectangular prism
-> truncated-normal vertex displacement
-> mesh subdivision
-> convex hull
-> OBB centering/alignment
-> random density
```

Planning follows the ICRA 2017 truth-state structure:

```text
known stone geometry and poses
-> select next object from available stones
-> sample target poses
-> MuJoCo contact search
-> release and settle
-> score contact/stability terms
-> commit best stable object/pose
```

The target task is changed from a vertical stack to a dry-stone wall. The
current wall plan fills a `4,3,2` set of courses. Upper-course target slots are
derived from the settled positions of stones in the course below.

## Main Command

```bash
cd /home/xunden/stone-stacking-mujoco
source .venv/bin/activate
python scripts/run_hybrid_icra_wall_planner.py
```

Verified output:

```text
placed_count: 9
requested_count: 9
final_height_m: 0.2584144921622992
```

The script writes:

```text
reports/hybrid_icra_wall_planner.json
outputs/hybrid_icra_wall_planner_final.xml
```

Replay the committed planning process in MuJoCo viewer:

```bash
python scripts/view_hybrid_icra_wall_process.py
```

Replay the planner result as a UR10-scale pick-place execution:

```bash
python scripts/view_hybrid_icra_arm_execution.py
```

This shows the arm moving to each selected stone, closing the gripper, carrying
the stone to the selected target pose, releasing it, and syncing to the
MuJoCo-settled planner state. The current execution is kinematic; true
contact-grasp with actuated Robotiq fingers is still the next implementation
step.

The replay uses the `trajectory` stored in
`reports/hybrid_icra_wall_planner.json`. It shows the wall after each committed
planner decision, from zero placed stones to the final `4,3,2` wall.

The current default stone generation parameters are:

```text
rock_irregularity: 1.0
rock_subdivisions: 5
```

These are intentionally more irregular than the initial low-zeta debugging
configuration, which looked too close to rectangular blocks.

## Current Status

Implemented:

- From-Rocks-to-Walls-style synthetic convex-hull stones.
- Course-by-course wall target generation.
- Online next-best object/target-pose selection.
- MuJoCo contact search with existing stones fixed.
- Full release/settle with all stones mobile.
- ICRA-style support polygon area, COM support, kinetic energy, normal
  deviation, target-distance and wall-drift checks.

Not implemented yet:

- Full UR10 articulated dynamics.
- Camera perception and pose estimation.
- Force-triggered placement controller.
- Real scanned stone dataset.
- DQN/RL training. This is intentionally excluded from the current path.

Partially implemented:

- Contact gripper lift test with collision pads, slide joints and position
  actuators:

```bash
python scripts/run_contact_gripper_lift_test.py --stone-index 1
python scripts/run_contact_gripper_lift_test.py --stone-index 1 --view
```

This is not an attach constraint. The gripper closes on the irregular rock and
lifts it by MuJoCo contact friction. The current end-effector pose is still
driven by a mocap weld, so the next step is integrating this gripper module
with the wall planner execution sequence and then replacing the mocap wrist
with full UR10 kinematics/dynamics.
