"""Approve a deliberate touch, or refuse it with a reason.

Planning and touching want opposite things from the same map. A plan is safe
when the arm never reaches the surface; a touch is only useful when it does. So
a contact task cannot borrow the planner's guarantee, and this module states
the one it can offer instead:

* the approach is an ordinary collision-free plan to a standoff pose,
* the probe is a straight line along one direction with nothing but the target
  surface in its corridor,
* and its length is bounded by how uncertain that surface's position is, rather
  than by a number someone chose.

That last point is the whole reason the bound is defensible. The surface might
be anywhere within the map's registration uncertainty plus the scatter of the
patch it was fitted to, so the probe must be allowed to travel at least that
far to touch at all -- and no further, because past that it is pressing rather
than touching.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from yam.environment import ArmSafetyChecker
from yam.hardware_calibration import CalibrationError, HardwareSafetyCalibration, file_sha256
from yam.kinematics import YamKinematics, numerical_jacobian, solve_ik_collision_free
from yam.planner import PlannerConfig, PlanningError, RRTConnectPlanner, resample
from yam.safe_planning import PlanningPolicy, SceneInterlock, validate_map_for_hardware, verify_tracking_envelope
from yam.safety_contract import ApprovedContact, SafetyCode, SafetyDecision
from yam.surface import SurfaceUnknown, estimate_normal, occupied_points
from yam.voxel_map import VoxelMap


@dataclass(frozen=True)
class ContactOutcome:
    decision: SafetyDecision
    approved: Optional[ApprovedContact] = None
    preview: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict:
        payload = self.decision.to_dict()
        if self.preview is not None:
            payload["preview"] = dict(self.preview)
        return payload


def travel_bound(voxel_map: VoxelMap, residual_m: float) -> float:
    """How far the probe must be allowed to travel to reach an uncertain surface.

    Two independent uncertainties place the surface: where the whole scan sits
    in the robot frame, and how thick the fitted patch was. Neither is chosen
    here; both are measured and reported by the map and the fit.
    """
    return float(voxel_map.uncertainty + 2.0 * residual_m + voxel_map.resolution)


def straight_probe(
    kinematics: YamKinematics,
    start_pose: np.ndarray,
    direction: np.ndarray,
    distance: float,
    step_m: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Joint poses that carry the jaw tip along `direction` and nowhere else.

    Each step is the minimum-norm joint change achieving that tip motion, so the
    arm keeps its configuration instead of drifting through the null space that
    position-only IK leaves free.
    """
    poses = [np.asarray(start_pose, dtype=float)]
    steps = max(1, int(round(distance / step_m)))
    increment = direction * (distance / steps)

    for _ in range(steps):
        jacobian = numerical_jacobian(kinematics, poses[-1])
        delta, *_ = np.linalg.lstsq(jacobian, increment, rcond=None)
        poses.append(np.clip(poses[-1] + delta, lower, upper))
    return np.asarray(poses)


def corridor_obstructions(
    kinematics: YamKinematics,
    voxel_map: VoxelMap,
    poses: np.ndarray,
    body_links: Sequence[str],
) -> list:
    """Where the probe would drive the arm's *body* into the scan.

    A contact check cannot ask whether the arm touches anything -- it is meant
    to. It asks which part does. The gripper is the tool and is expected to
    reach the surface; every other link is not, and a probe that buries a
    forearm in a shelf while the jaws approach a tabletop is not a touch.

    Masking a region around the target was the obvious alternative and does not
    work: a tabletop or a floor extends past any radius, so the surface being
    approached keeps reporting itself as an obstruction.
    """
    field_map = VoxelMap(origin=voxel_map.origin.copy(), resolution=voxel_map.resolution,
                         occupancy=voxel_map.occupancy.copy())
    if voxel_map.synthetic_occupancy is not None:
        field_map.occupancy |= voxel_map.synthetic_occupancy
    field_map.compute_distance_field()

    for index, pose in enumerate(poses):
        centres, radii = kinematics.collision_spheres(pose, body_links)
        gaps = field_map.measured_distance_at(centres) - radii
        worst = int(np.argmin(gaps))
        if gaps[worst] <= 0.0:
            return [{
                "probe_index": index,
                "sphere_m": centres[worst].tolist(),
                "penetration_m": float(-gaps[worst]),
            }]
    return []


def approve_contact(
    start_joint_positions: Sequence[float],
    target_point: Sequence[float],
    map_path: str,
    arm_xml_path: str,
    policy: PlanningPolicy,
    hardware_requested: bool,
    calibration_path: Optional[str] = None,
    scene_interlock: Optional[SceneInterlock] = None,
    probe_step_m: float = 0.001,
    now: Optional[float] = None,
) -> ContactOutcome:
    """Plan a touch of `target_point`, or refuse with a specific reason."""
    try:
        policy.validate(hardware_requested)
    except ValueError as error:
        return _refuse(SafetyCode.INVALID_REQUEST, str(error))

    try:
        voxel_map = VoxelMap.load(map_path)
    except Exception as error:
        return _refuse(SafetyCode.MAP_MISSING_PROVENANCE, f"the workcell map cannot be loaded ({error})")

    calibration = None
    if hardware_requested:
        decision = validate_map_for_hardware(voxel_map, policy.max_map_age_seconds, now=now)
        if not decision.allowed:
            return ContactOutcome(decision)
        if scene_interlock is None or scene_interlock.remaining_seconds(now) <= 0.0:
            return _refuse(SafetyCode.SCENE_NOT_INTERLOCKED,
                           "no fresh observation of the scene backs this touch")
        if not calibration_path:
            return _refuse(SafetyCode.CALIBRATION_MISSING,
                           "touching requires a hardware calibration; the guard is what stops the probe")
        try:
            calibration = HardwareSafetyCalibration.load(
                calibration_path, max_age_seconds=policy.max_calibration_age_seconds, now=now)
        except CalibrationError as error:
            return _refuse(SafetyCode.CALIBRATION_INVALID, str(error))

    target = np.asarray(target_point, dtype=float).reshape(3)
    cloud = occupied_points(voxel_map)
    try:
        surface = estimate_normal(voxel_map, target, cloud=cloud)
    except SurfaceUnknown as error:
        return _refuse(SafetyCode.CONTACT_SURFACE_UNKNOWN, str(error))

    max_travel = travel_bound(voxel_map, surface.residual_m)
    kinematics = YamKinematics()
    try:
        checker = ArmSafetyChecker(kinematics, voxel_map, arm_xml_path,
                                   margin=policy.requested_clearance_m,
                                   self_collision_margin=policy.self_collision_margin_m)
    except Exception as error:
        return _refuse(SafetyCode.INVALID_REQUEST, f"the collision checker cannot start ({error})")

    from yam.arm import ARM_JOINTS
    from yam.environment import GRIPPER_LINKS
    lower = np.array([joint.lower_limit for joint in ARM_JOINTS])
    upper = np.array([joint.upper_limit for joint in ARM_JOINTS])
    start = np.asarray(start_joint_positions, dtype=float)

    body_links = tuple(link for link in checker.links if link not in set(GRIPPER_LINKS))

    if not checker.is_free(start):
        return _refuse(SafetyCode.START_UNSAFE,
                       f"the arm does not start clear: {'; '.join(checker.explain(start))}")

    # The standoff has to clear the surface by the planner's own requirement,
    # but the jaw tip is not the closest part of the arm to what it is reaching
    # for -- the gripper body extends past it, by an amount no surface property
    # predicts. So the smallest standoff that actually admits a clear pose is
    # searched for rather than derived, and the one that is found is reported.
    base_standoff = checker.measured_margin + voxel_map.resolution
    attempts = []
    for direction in surface.approach_directions():
        approach_pose, standoff_distance = None, None
        for scale in (1.0, 1.5, 2.0, 3.0, 4.0):
            candidate = base_standoff * scale
            pose = solve_ik_collision_free(
                kinematics, target - direction * candidate, checker, lower, upper, seed=start)
            if pose is not None:
                approach_pose, standoff_distance = pose, candidate
                break
        attempts.append({
            "direction": direction.tolist(),
            "reachable": approach_pose is not None,
            "standoff_m": standoff_distance,
        })
        if approach_pose is None:
            continue

        probe = straight_probe(kinematics, approach_pose, direction,
                               standoff_distance + max_travel, probe_step_m, lower, upper)
        obstructions = corridor_obstructions(kinematics, voxel_map, probe, body_links)
        if obstructions:
            attempts[-1]["obstructed_at"] = obstructions[0]
            continue

        planner = RRTConnectPlanner(checker, lower, upper, PlannerConfig(seed=policy.planner_seeds[0]))
        try:
            approach = resample(planner.plan(start, approach_pose), policy.path_step_rad)
        except PlanningError as error:
            attempts[-1]["unplannable"] = str(error)
            continue

        return _finish(approach, probe, direction, surface, max_travel, voxel_map, checker,
                       calibration, policy, map_path, calibration_path, hardware_requested,
                       lower, upper, attempts, now)

    return _refuse(SafetyCode.CONTACT_UNREACHABLE,
                   f"no side of {np.round(target, 3).tolist()} admits a clear approach",
                   {"attempts": attempts, "surface": surface.to_dict()})


def _finish(approach, probe, direction, surface, max_travel, voxel_map, checker, calibration,
            policy, map_path, calibration_path, hardware_requested, lower, upper, attempts, now):
    report = {
        "surface": surface.to_dict(),
        "approach_poses": int(len(approach)),
        "probe_poses": int(len(probe)),
        "max_travel_m": max_travel,
        "registration_uncertainty_m": float(voxel_map.uncertainty),
        "attempts": attempts,
    }

    if not hardware_requested:
        return ContactOutcome(
            SafetyDecision.preview("this touch passed model checks but has no hardware calibration "
                                   "or live-scene interlock"),
            preview=report)

    envelope = verify_tracking_envelope(checker, approach, calibration.max_tracking_error_rad, lower, upper)
    if not envelope.get("ok", False):
        report["tracking_envelope"] = envelope
        return ContactOutcome(SafetyDecision.refuse(
            SafetyCode.TRACKING_ENVELOPE_UNSAFE,
            "the approach collides inside the tracking-error envelope measured on this arm",
            report))

    issued = time.time() if now is None else float(now)
    approved = ApprovedContact(
        approach=approach,
        probe=probe,
        approach_sha256=_hash(approach),
        probe_sha256=_hash(probe),
        map_sha256=file_sha256(map_path),
        calibration_sha256=calibration.sha256,
        issued_at_unix=issued,
        valid_for_seconds=float(policy.approval_valid_seconds),
        start_tolerance_rad=tuple(np.minimum(calibration.max_tracking_error_rad, 0.05).tolist()),
        approach_direction=direction,
        max_travel_m=max_travel,
        surface=surface.to_dict(),
        report=report,
    )
    return ContactOutcome(SafetyDecision.approve(report), approved=approved)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def _refuse(code: SafetyCode, reason: str, details: Optional[Mapping[str, Any]] = None) -> ContactOutcome:
    return ContactOutcome(SafetyDecision.refuse(code, reason, details or {}))
