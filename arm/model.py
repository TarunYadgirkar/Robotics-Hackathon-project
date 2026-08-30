"""Joint model, soft limits and speed caps — mirrored from the in-repo YAM driver.

PROVENANCE, because none of it was measured by this module:

* Joint names, order and limits come from `yam.arm.ARM_JOINTS` (i2rt yam_pro_v1),
  converted rad -> deg here because the trajectory JSONs and the sim animation
  are far more readable in degrees. `hw_backend.verify_against_yam()` re-checks
  these numbers against the live `yam.arm` module at connect time, so a drift
  between this file and the driver is caught before anything moves.
* MAX_VEL_DEG_S is derived from `yam.arm.SafetyLimits.max_step_per_tick` (0.02
  rad) at the driver's 100 Hz control tick — i.e. the arm's own slew clamp — and
  the 30% cap in this module sits underneath it. It is not an invented number.
* HOME_POSE_DEG is the measured resting pose from `scripts/plan_and_run.py`
  (stable with the motors off), with joint2/joint3 clamped up to their 0.0 lower
  bound: at rest they read ~0.01 deg below it, which `yam.arm` notes and clamps
  on the first command anyway.
* The gripper is SIM-ONLY. `yam.arm.ARM_JOINTS` deliberately excludes it (jaw
  travel depends on the jaws fitted and is uncalibrated on this robot), so the
  hardware backend commands the six arm joints and refuses gripper motion. In
  the simulator the gripper channel carries percent-open, not degrees, so the
  pick-and-place trajectory has something to grasp with.

NOT verified here: self-collision. `yam.arm`'s own notes say ~10% of in-limit
poses self-collide and nothing in the driver checks it. The poses in
arm/gestures/*.json were authored conservatively (small joint2, lifting done by
joint3, everything near home) but they have NOT been run through
`yam.environment.ArmSafetyChecker` — mujoco is not installed and the i2rt URDF
is not on this machine. That check is a hard prerequisite in hw_backend.
"""

import math

LIMITS_SOURCE = "yam.arm.ARM_JOINTS (i2rt yam_pro_v1), rad->deg; gripper channel is sim-only"

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")
ARM_JOINT_NAMES = JOINT_NAMES[:6]
GRIPPER_NAME = "gripper"
N_JOINTS = len(JOINT_NAMES)

UNITS_LABEL = "deg (joint1-joint6), percent_open (gripper)"

# rad limits copied from yam.arm.ARM_JOINTS, converted to degrees.
_YAM_LIMITS_RAD = {
    "joint1": (-2.61799, 3.14159),
    "joint2": (0.0, 3.66519),
    "joint3": (0.0, 3.14159),
    "joint4": (-1.69297, 1.5708),
    "joint5": (-1.5708, 1.5708),
    "joint6": (-2.0944, 2.0944),
}

LIMITS_DEG = {name: (math.degrees(lo), math.degrees(hi)) for name, (lo, hi) in _YAM_LIMITS_RAD.items()}
LIMITS_DEG[GRIPPER_NAME] = (0.0, 100.0)  # percent open, sim-only

# yam.arm.SafetyLimits: 0.02 rad per tick at the driver's 100 Hz control rate.
YAM_MAX_STEP_PER_TICK_RAD = 0.02
YAM_TICK_HZ = 100.0
_YAM_SLEW_DEG_S = math.degrees(YAM_MAX_STEP_PER_TICK_RAD * YAM_TICK_HZ)  # ~114.6 deg/s

MAX_VEL_DEG_S = {name: _YAM_SLEW_DEG_S for name in ARM_JOINT_NAMES}
MAX_VEL_DEG_S[GRIPPER_NAME] = 120.0  # percent/s, sim-only

VELOCITY_CAP_FRACTION = 0.30
LOW_SPEED_FRACTION = 0.25  # homing after an abort
CONTROL_HZ = 30.0

# scripts/plan_and_run.py HOME, rad -> deg, joint2/joint3 clamped to their bound.
HOME_POSE_DEG = (2.854, 0.0, 0.0, -5.191, 4.205, 67.070, 20.0)

# yam.arm: joint2 self-collides against the base past roughly +105 deg from a
# folded home pose. Nothing here plans around it, so authored poses stay well
# below this and the smoke test asserts it.
JOINT2_SELF_COLLISION_DEG = 105.0
JOINT2_AUTHORED_CEILING_DEG = 45.0


def velocity_cap_deg_s() -> dict[str, float]:
    return {j: MAX_VEL_DEG_S[j] * VELOCITY_CAP_FRACTION for j in JOINT_NAMES}


def to_yam_radians(positions) -> list[float]:
    """The six arm-joint targets `yam.arm.YamArm.command_positions` expects."""
    return [math.radians(v) for v in positions[:6]]


def check_limits(positions) -> list[str]:
    """Return a list of human-readable soft-limit violations (empty == clean)."""
    if len(positions) != N_JOINTS:
        return [f"expected {N_JOINTS} channels {JOINT_NAMES}, got {len(positions)}"]
    out = []
    for name, value in zip(JOINT_NAMES, positions):
        lo, hi = LIMITS_DEG[name]
        if not lo <= value <= hi:
            out.append(f"{name}={value:.2f} outside soft limit [{lo:.2f}, {hi:.2f}]")
    return out
