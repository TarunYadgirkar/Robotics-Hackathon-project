"""Joint model, soft limits and speed caps — mirrored from the in-repo YAM driver.

PROVENANCE, because none of it was measured by this module:

* Joint names, order and limits come from `yam.arm.ARM_JOINTS` (i2rt yam_pro_v1),
  converted rad -> deg here because the trajectory JSONs and the sim animation
  are far more readable in degrees. `hw_backend.verify_against_yam()` re-checks
  these numbers against the live `yam.arm` module at connect time, so a drift
  between this file and the driver is caught before anything moves.
* MAX_VEL_DEG_S is derived from `yam.arm.SafetyLimits.max_joint_speed` (2.0
  rad/s) — the arm's own speed ceiling — and the 30% cap here sits underneath
  it. It is not an invented number. NOTE: this used to be derived from
  `max_step_per_tick` (0.02 rad/tick at 100 Hz). Boris's sync replaced that with
  a time-based limit precisely because a per-tick cap made the real speed a
  function of loop rate, and `verify_against_yam()` caught the drift on the
  first hardware run — the numeric cap is unchanged (0.02 rad x 100 Hz = 2.0
  rad/s), but it is now anchored to the constant that actually governs.
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

# yam.arm.SafetyLimits, current semantics: a rad/s speed ceiling plus a per-tick
# anti-lunge ceiling that stops a late tick authorising a jump.
YAM_MAX_JOINT_SPEED_RAD_S = 2.0
YAM_MAX_STEP_PER_TICK_RAD = 0.05
YAM_TICK_HZ = 100.0
_YAM_SPEED_DEG_S = math.degrees(YAM_MAX_JOINT_SPEED_RAD_S)  # ~114.6 deg/s

MAX_VEL_DEG_S = {name: _YAM_SPEED_DEG_S for name in ARM_JOINT_NAMES}
MAX_VEL_DEG_S[GRIPPER_NAME] = 120.0  # percent/s; the sim branch's ceiling

# -- hardware-mode constraints (operator's rule at the arm, enforced in code) --
# Checked against the prepared setpoint stream in hw_backend BEFORE anything is
# sent, so replaying a sim trajectory on hardware by mistake is refused rather
# than swept through space.
#
# AUTHORIZATION, 2026-08-30, user present at the arm, verbatim: "you can move the
# whole arm but do it slow and dont hit anything." This replaced the earlier
# gripper-only rule (5 deg), which the user judged too subtle to read as motion
# on stage. The envelope that replaces it is: bigger excursions on the axes that
# cannot fold the arm into itself, still-tight limits on the two that can, and a
# hard slow-speed ceiling well under the velocity cap.
HW_MAX_EXCURSION_DEG = 30.0      # default per arm joint, from the pose at motion start
#: joint2 is yam.arm's documented base-collision trap — it self-collides past
#: ~105 deg from a folded home and moves the tip DOWN, not up — so it keeps a
#: tight budget. joint3 is the joint that actually lifts and yam.arm records it
#: as "self-collision-free across its whole range from home", which is why it
#: gets a much larger one: the visible lift in these gestures is joint3's.
HW_PER_JOINT_EXCURSION_DEG = {"joint2": 15.0, "joint3": 25.0}
HW_SLOW_SPEED_DEG_S = 15.0       # "do it slow": under half the 34.4 deg/s cap
HW_GRIPPER_GENTLE_PCT_S = 12.0   # far below the sim cap; the jaws move slowly
HW_GAIN_SCALE = 0.5              # yam.arm gain_scale: softer than the SDK default


def hw_excursion_limit(joint_name: str) -> float:
    return HW_PER_JOINT_EXCURSION_DEG.get(joint_name, HW_MAX_EXCURSION_DEG)

# Gripper calibration, from yam.arm (measured by Boris on this robot with these
# jaws). The closed stop is a true hard stop and is the datum; the open stop is
# compliant. The 0-100 percent channel used in trajectories maps onto this.
GRIPPER_CLOSED_RAD = -5.158
GRIPPER_OPEN_RAD = 0.056

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
    """Targets for `yam.arm.YamArm.command_positions`, built for a 7-joint arm.

    The gripper is included because the hardware backend constructs YamArm with
    ARM_JOINTS + GRIPPER_JOINT: Boris's sync calibrated the jaws, so the gripper
    is now a commandable joint rather than an excluded one. Its channel converts
    from percent-open to the motor angle through the measured stops.
    """
    return [math.radians(v) for v in positions[:6]] + [gripper_percent_to_rad(positions[6])]


def gripper_percent_to_rad(percent: float) -> float:
    span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    return GRIPPER_CLOSED_RAD + (max(0.0, min(100.0, percent)) / 100.0) * span


def gripper_rad_to_percent(position: float) -> float:
    """Inverse of the above, via yam.arm.gripper_opening_fraction's definition."""
    span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    return float(min(max((position - GRIPPER_CLOSED_RAD) / span, 0.0), 1.0)) * 100.0


#: yam.arm: "Motors can sit marginally outside these limits at rest -- j2/j3 read
#: 0.0115 deg below their 0.0 lower bound". Measured again on this arm during
#: read-only bring-up: joint2 and joint3 both rest at -0.01 deg. A limit check
#: that refuses the pose the arm is physically resting in is a broken check, so
#: the bound carries this tolerance.
REST_LIMIT_TOLERANCE_DEG = 0.05

#: Interpolating a joint back to its starting value does not land exactly on it:
#: 19.95 - 20.0 is -0.050000000000000710 in binary floating point, which read as
#: a limit violation against a bound of exactly -0.05. A millionth of a degree is
#: many orders of magnitude below anything the encoder or the arm can express.
LIMIT_EPSILON_DEG = 1e-6


def check_limits(positions, base=None) -> list[str]:
    """Human-readable soft-limit violations (empty == clean).

    `base` is the pose a relative motion starts from. Where the arm already
    rests outside a bound, that much excursion is accepted — the rule is that a
    motion may not push a joint FURTHER out than it already is, not that the
    arm's resting pose is illegal. yam.arm's own clamp_position() pulls such a
    joint back inside on the first command either way.
    """
    if len(positions) != N_JOINTS:
        return [f"expected {N_JOINTS} channels {JOINT_NAMES}, got {len(positions)}"]
    out = []
    for i, (name, value) in enumerate(zip(JOINT_NAMES, positions)):
        lo, hi = LIMITS_DEG[name]
        allowance = REST_LIMIT_TOLERANCE_DEG
        if base is not None:
            resting = base[i]
            allowance = max(allowance, lo - resting, resting - hi)
        allowance += LIMIT_EPSILON_DEG
        if not (lo - allowance) <= value <= (hi + allowance):
            out.append(f"{name}={value:.2f} outside soft limit [{lo:.2f}, {hi:.2f}]")
    return out
