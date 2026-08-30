"""Fail-closed, on-demand motion planning against the measured workcell.

The language/action layer may propose goals. It never approves motion. This
module resolves those goals, plans from a measured start state, validates the
entire path and the calibrated tracking-error envelope, then issues the only
object the guarded executor accepts.
"""

import hashlib
import itertools
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from yam.arm import ARM_JOINTS
from yam.environment import ArmSafetyChecker
from yam.hardware_calibration import CalibrationError, HardwareSafetyCalibration, file_sha256
from yam.kinematics import YamKinematics, solve_ik_collision_free
from yam.dm_motor import MOTOR_SPECS
from yam.planner import PlannerConfig, PlanningError, RRTConnectPlanner, path_length, resample
from yam.safety_contract import ApprovedPlan, SafetyCode, SafetyDecision
from yam.voxel_map import VoxelMap


FEEDBACK_POSITION_RESOLUTION_RAD = max(
    2.0 * specification.position_max / ((1 << 16) - 1)
    for specification in MOTOR_SPECS.values()
)


@dataclass(frozen=True)
class MotionGoal:
    kind: str
    values: tuple[float, ...]
    label: str = ""

    @classmethod
    def tip(cls, position_m: Sequence[float], label: str = "") -> "MotionGoal":
        values = tuple(float(value) for value in position_m)
        if len(values) != 3:
            raise ValueError("tip goal must contain x, y, z in metres")
        return cls("tip", values, label)

    @classmethod
    def joints_radians(cls, positions: Sequence[float], label: str = "") -> "MotionGoal":
        values = tuple(float(value) for value in positions)
        if len(values) != 6:
            raise ValueError("joint goal must contain six angles")
        return cls("joints", values, label)

    @classmethod
    def joints_degrees(cls, positions: Sequence[float], label: str = "") -> "MotionGoal":
        values = tuple(math.radians(float(value)) for value in positions)
        return cls.joints_radians(values, label)


@dataclass(frozen=True)
class SceneInterlock:
    """A fresh trusted observation supplied by the hardware layer, never the LLM."""

    source: str
    observed_at_unix: float
    valid_for_seconds: float

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        current = time.time() if now is None else float(now)
        return self.observed_at_unix + self.valid_for_seconds - current


@dataclass(frozen=True)
class PlanningPolicy:
    requested_clearance_m: float = 0.03
    self_collision_margin_m: float = 0.003
    path_step_rad: float = 0.02
    verification_step_rad: float = 0.01
    planner_seeds: tuple[int, ...] = (1, 7, 23)
    max_map_age_seconds: Optional[float] = None
    max_calibration_age_seconds: Optional[float] = None
    approval_valid_seconds: Optional[float] = None

    def validate(self, hardware_requested: bool) -> None:
        positive = {
            "requested_clearance_m": self.requested_clearance_m,
            "self_collision_margin_m": self.self_collision_margin_m,
            "path_step_rad": self.path_step_rad,
            "verification_step_rad": self.verification_step_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.planner_seeds:
            raise ValueError("at least one deterministic planner seed is required")
        if hardware_requested:
            required = {
                "max_map_age_seconds": self.max_map_age_seconds,
                "max_calibration_age_seconds": self.max_calibration_age_seconds,
                "approval_valid_seconds": self.approval_valid_seconds,
            }
            for name, value in required.items():
                if value is None or not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"hardware planning requires a finite positive {name}")


@dataclass
class PlanningOutcome:
    decision: SafetyDecision
    approved_plan: Optional[ApprovedPlan] = None
    preview_path: Optional[np.ndarray] = None
    goal_indices: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = self.decision.to_dict()
        result["goal_indices"] = list(self.goal_indices)
        if self.approved_plan is not None:
            result["plan"] = dict(self.approved_plan.report)
        elif self.preview_path is not None:
            result["plan"] = {"poses": len(self.preview_path), "hardware_eligible": False}
        return result


def plan_on_demand(
    start: Sequence[float],
    goals: Sequence[MotionGoal],
    map_path: str | os.PathLike[str],
    arm_xml_path: str | os.PathLike[str],
    policy: Optional[PlanningPolicy] = None,
    *,
    hardware_requested: bool = False,
    calibration_path: Optional[str | os.PathLike[str]] = None,
    scene_interlock: Optional[SceneInterlock] = None,
    now: Optional[float] = None,
) -> PlanningOutcome:
    """Generate a plan or a specific refusal; never return an unapproved hardware path."""
    current_time = time.time() if now is None else float(now)
    policy = policy or PlanningPolicy()
    try:
        policy.validate(hardware_requested)
        start_array = _configuration(start, "start")
        goals = list(goals)
        if not goals:
            raise ValueError("at least one motion goal is required")
    except ValueError as error:
        return _refusal(SafetyCode.INVALID_REQUEST, str(error))

    try:
        voxel_map = VoxelMap.load(str(map_path))
    except (OSError, ValueError) as error:
        return _refusal(SafetyCode.MAP_MISSING_PROVENANCE, f"the workcell map cannot be loaded ({error})")

    map_digest = file_sha256(map_path)
    calibration = None
    if hardware_requested:
        map_decision = validate_map_for_hardware(
            voxel_map,
            policy.max_map_age_seconds,
            current_time,
        )
        if not map_decision.allowed:
            return PlanningOutcome(map_decision)
        if calibration_path is None:
            return _refusal(
                SafetyCode.CALIBRATION_MISSING,
                "there is no hardware calibration trace for the tracking and contact limits",
            )
        try:
            calibration = HardwareSafetyCalibration.load(
                calibration_path,
                max_age_seconds=policy.max_calibration_age_seconds,
                now=current_time,
            )
        except CalibrationError as error:
            code = SafetyCode.CALIBRATION_STALE if "old; policy permits" in str(error) else SafetyCode.CALIBRATION_INVALID
            return _refusal(code, str(error))
        if scene_interlock is None or scene_interlock.remaining_seconds(current_time) <= 0.0:
            return _refusal(
                SafetyCode.SCENE_NOT_INTERLOCKED,
                "there is no current trusted observation that the workspace stayed clear after the scan",
            )

    try:
        kinematics = YamKinematics()
        checker = ArmSafetyChecker(
            kinematics,
            voxel_map,
            str(arm_xml_path),
            margin=policy.requested_clearance_m,
            self_collision_margin=policy.self_collision_margin_m,
        )
    except Exception as error:
        return _refusal(SafetyCode.INVALID_REQUEST, f"the collision checker cannot start ({error})")

    lower = np.array([joint.lower_limit for joint in ARM_JOINTS], dtype=float)
    upper = np.array([joint.upper_limit for joint in ARM_JOINTS], dtype=float)
    limit_problem = _limit_problem(
        start_array,
        lower,
        upper,
        tolerance=FEEDBACK_POSITION_RESOLUTION_RAD,
    )
    if limit_problem:
        return _refusal(SafetyCode.START_UNSAFE, limit_problem)
    if not checker.is_free(start_array):
        return _refusal(
            SafetyCode.START_UNSAFE,
            "the measured start pose is already in collision",
            {"collision": checker.explain(start_array)},
        )

    path_parts = [start_array[None, :]]
    goal_indices = []
    current = start_array

    for goal_number, goal in enumerate(goals):
        try:
            target = _resolve_goal(goal, current, kinematics, checker, lower, upper)
        except GoalRefused as error:
            details = {"goal": goal_number, "label": goal.label}
            details.update(error.details)
            return _refusal(error.code, str(error), details)

        segment = None
        last_error = None
        for seed in policy.planner_seeds:
            planner = RRTConnectPlanner(
                checker,
                lower,
                upper,
                PlannerConfig(
                    seed=seed,
                    collision_resolution=policy.verification_step_rad,
                ),
            )
            try:
                candidate = planner.plan(current, target)
                if np.allclose(candidate[0], current) and np.allclose(candidate[-1], target):
                    segment = np.asarray(resample(candidate, policy.path_step_rad))
                    break
                last_error = PlanningError("planner returned reversed or disconnected endpoints")
            except PlanningError as error:
                last_error = error
        if segment is None:
            return _refusal(
                SafetyCode.PATH_NOT_FOUND,
                f"no collision-free path was found for {goal.label or f'goal {goal_number + 1}'}",
                {"planner_error": str(last_error) if last_error else "unknown"},
            )

        path_parts.append(segment[1:])
        current = target
        goal_indices.append({
            "index": sum(len(part) for part in path_parts) - 1,
            "label": goal.label,
            "kind": goal.kind,
        })

    path = np.concatenate(path_parts, axis=0)
    verification = verify_nominal_path(
        checker,
        path,
        start_array,
        current,
        policy.path_step_rad,
        policy.verification_step_rad,
    )
    if not verification["ok"]:
        return _refusal(
            SafetyCode(verification["code"]),
            verification["reason"],
            verification,
        )

    report: dict[str, Any] = {
        "poses": len(path),
        "joint_travel_rad": path_length(path),
        "max_step_rad": float(np.linalg.norm(np.diff(path, axis=0), axis=1).max()) if len(path) > 1 else 0.0,
        "minimum_nominal_slack_m": verification["minimum_slack_m"],
        "requested_clearance_m": policy.requested_clearance_m,
        "registration_uncertainty_m": voxel_map.uncertainty,
        "hardware_eligible": hardware_requested,
    }

    if not hardware_requested:
        reason = "this path passed model checks but has no hardware calibration or live-scene interlock"
        return PlanningOutcome(
            SafetyDecision.preview(reason, report),
            preview_path=path,
            goal_indices=goal_indices,
        )

    tracking_report = verify_tracking_envelope(
        checker,
        path,
        calibration.max_tracking_error_rad,
        lower,
        upper,
    )
    report["tracking_envelope"] = tracking_report
    if not tracking_report["ok"]:
        return _refusal(
            SafetyCode.TRACKING_ENVELOPE_UNSAFE,
            "the path collides inside the tracking-error envelope measured on this arm",
            report,
        )

    issued_at = time.time()
    valid_for = min(policy.approval_valid_seconds, scene_interlock.remaining_seconds(issued_at))
    if valid_for <= 0.0:
        return _refusal(
            SafetyCode.SCENE_NOT_INTERLOCKED,
            "the trusted workspace observation expired while the path was being checked",
        )
    path_hash = hashlib.sha256(np.asarray(path, dtype="<f8").tobytes()).hexdigest()
    report.update({
        "path_sha256": path_hash,
        "map_sha256": map_digest,
        "calibration_sha256": calibration.sha256,
        "scene_interlock_source": scene_interlock.source,
        "approval_valid_seconds": valid_for,
    })
    approved = ApprovedPlan(
        path=path,
        path_sha256=path_hash,
        map_sha256=map_digest,
        calibration_sha256=calibration.sha256,
        issued_at_unix=issued_at,
        valid_for_seconds=valid_for,
        start_tolerance_rad=calibration.max_tracking_error_rad,
        report=report,
    )
    return PlanningOutcome(SafetyDecision.approve(report), approved, goal_indices=goal_indices)


def validate_map_for_hardware(
    voxel_map: VoxelMap,
    max_age_seconds: Optional[float],
    now: Optional[float] = None,
) -> SafetyDecision:
    provenance = voxel_map.provenance
    if provenance.get("schema_version") != 1:
        return SafetyDecision.refuse(
            SafetyCode.MAP_MISSING_PROVENANCE,
            "the map does not identify the scan and registration that produced it",
        )
    if max_age_seconds is None or not math.isfinite(max_age_seconds) or max_age_seconds <= 0.0:
        return SafetyDecision.refuse(
            SafetyCode.INVALID_REQUEST,
            "hardware policy does not specify how old a scene observation may be",
        )

    scan = provenance.get("scan") or {}
    registration = provenance.get("registration") or {}
    captured_at = scan.get("captured_at_unix")
    try:
        captured_at = float(captured_at)
    except (TypeError, ValueError):
        return SafetyDecision.refuse(
            SafetyCode.MAP_MISSING_PROVENANCE,
            "the map has no measured scan capture time",
        )
    current = time.time() if now is None else float(now)
    age = current - captured_at
    if age < 0.0:
        return SafetyDecision.refuse(SafetyCode.MAP_MISSING_PROVENANCE, "the scan timestamp is in the future")
    if age > max_age_seconds:
        return SafetyDecision.refuse(
            SafetyCode.MAP_STALE,
            f"the scene scan is {age:.0f}s old; policy permits {max_age_seconds:.0f}s",
            {"map_age_seconds": age, "max_map_age_seconds": max_age_seconds},
        )

    for name, source in (("scan", scan), ("registration", registration)):
        source_path = source.get("path")
        expected_hash = source.get("sha256")
        if not source_path or not expected_hash or not Path(source_path).is_file():
            return SafetyDecision.refuse(
                SafetyCode.MAP_MISSING_PROVENANCE,
                f"the map's {name} source cannot be revalidated",
            )
        if file_sha256(source_path) != expected_hash:
            return SafetyDecision.refuse(
                SafetyCode.MAP_SOURCE_CHANGED,
                f"the {name} source changed after this map was built",
            )
    return SafetyDecision.approve({"map_age_seconds": age})


def verify_nominal_path(
    checker,
    path: np.ndarray,
    expected_start: np.ndarray,
    expected_goal: np.ndarray,
    max_step_rad: float,
    segment_resolution_rad: float,
) -> dict:
    if not np.allclose(path[0], expected_start, atol=1e-9) or not np.allclose(path[-1], expected_goal, atol=1e-9):
        return {
            "ok": False,
            "code": SafetyCode.PATH_DISCONTINUOUS.value,
            "reason": "the path does not connect the measured start to the requested goal",
        }
    if len(path) > 1:
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        worst_step = float(steps.max())
        if worst_step > max_step_rad * (1.0 + 1e-9):
            return {
                "ok": False,
                "code": SafetyCode.PATH_DISCONTINUOUS.value,
                "reason": f"the path contains a {worst_step:.4f}rad jump",
                "max_step_rad": worst_step,
            }

    minimum_slack = float("inf")
    for index, configuration in enumerate(path):
        slack = checker.clearance(configuration)
        minimum_slack = min(minimum_slack, slack)
        if not checker.is_free(configuration):
            return {
                "ok": False,
                "code": SafetyCode.PATH_UNSAFE.value,
                "reason": f"path pose {index} is in collision",
                "collision": checker.explain(configuration),
            }
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        if not checker.segment_is_free(start, end, segment_resolution_rad):
            return {
                "ok": False,
                "code": SafetyCode.PATH_UNSAFE.value,
                "reason": f"path segment {index} enters collision between saved poses",
            }
    return {"ok": True, "minimum_slack_m": minimum_slack}


def verify_tracking_envelope(
    checker,
    path: Sequence[Sequence[float]],
    tracking_bounds: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict:
    """Deterministically stress every path pose at axes and all error-box corners.

    The bound itself must come from the hardware trace. This is stronger and
    repeatable compared with the former random perturbation check, though it is
    still a model-based verification rather than a mathematical proof of all
    configurations inside the six-dimensional box.
    """
    bounds = np.asarray(tracking_bounds, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    offsets = [np.zeros(6)]
    for joint in range(6):
        for sign in (-1.0, 1.0):
            offset = np.zeros(6)
            offset[joint] = sign * bounds[joint]
            offsets.append(offset)
    offsets.extend(np.asarray(signs) * bounds for signs in itertools.product((-1.0, 1.0), repeat=6))

    checked = 0
    worst_clearance = float("inf")
    for path_index, waypoint in enumerate(path):
        waypoint = np.asarray(waypoint, dtype=float)
        for offset_index, offset in enumerate(offsets):
            perturbed = np.clip(waypoint + offset, lower_array, upper_array)
            clearance = checker.clearance(perturbed)
            worst_clearance = min(worst_clearance, clearance)
            checked += 1
            if not checker.is_free(perturbed):
                return {
                    "ok": False,
                    "checked": checked,
                    "samples_per_pose": len(offsets),
                    "first_failure": {
                        "path_index": path_index,
                        "offset_index": offset_index,
                        "offset_deg": np.degrees(perturbed - waypoint).round(2).tolist(),
                        "collision": checker.explain(perturbed),
                    },
                    "worst_clearance_m": worst_clearance,
                    "tracking_bounds_deg": np.degrees(bounds).round(3).tolist(),
                }
    return {
        "ok": True,
        "checked": checked,
        "samples_per_pose": len(offsets),
        "first_failure": None,
        "worst_clearance_m": worst_clearance,
        "tracking_bounds_deg": np.degrees(bounds).round(3).tolist(),
    }


class GoalRefused(ValueError):
    def __init__(self, code: SafetyCode, message: str, details: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _resolve_goal(goal, current, kinematics, checker, lower, upper) -> np.ndarray:
    values = np.asarray(goal.values, dtype=float)
    if not np.isfinite(values).all():
        raise GoalRefused(SafetyCode.INVALID_REQUEST, "goal contains non-finite values")
    if goal.kind == "joints":
        if values.shape != (6,):
            raise GoalRefused(SafetyCode.INVALID_REQUEST, "joint goal must contain six angles")
        problem = _limit_problem(values, lower, upper)
        if problem:
            raise GoalRefused(SafetyCode.GOAL_UNSAFE, problem)
        target = values
    elif goal.kind == "tip":
        if values.shape != (3,):
            raise GoalRefused(SafetyCode.INVALID_REQUEST, "tip goal must contain x, y, z")
        target = solve_ik_collision_free(
            kinematics,
            values,
            checker,
            lower,
            upper,
            seed=current,
        )
        if target is None:
            raise GoalRefused(
                SafetyCode.GOAL_UNREACHABLE,
                "no collision-free inverse-kinematics solution reaches the requested point",
                {"tip_m": values.tolist()},
            )
        reached = kinematics.probe_position(target)
        if np.linalg.norm(reached - values) >= 0.005:
            raise GoalRefused(
                SafetyCode.GOAL_UNREACHABLE,
                "inverse kinematics did not reach the requested point within 5mm",
                {"tip_error_m": float(np.linalg.norm(reached - values))},
            )
    else:
        raise GoalRefused(SafetyCode.INVALID_REQUEST, f"unknown goal kind {goal.kind!r}")

    if not checker.is_free(target):
        raise GoalRefused(
            SafetyCode.GOAL_UNSAFE,
            "the requested goal pose is in collision",
            {"collision": checker.explain(target)},
        )
    return np.asarray(target, dtype=float)


def _configuration(values: Sequence[float], name: str) -> np.ndarray:
    configuration = np.asarray(values, dtype=float)
    if configuration.shape != (6,) or not np.isfinite(configuration).all():
        raise ValueError(f"{name} must contain six finite joint angles in radians")
    return configuration


def _limit_problem(
    configuration: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float = 0.0,
) -> Optional[str]:
    outside = np.flatnonzero(
        (configuration < lower - tolerance) | (configuration > upper + tolerance)
    )
    if not len(outside):
        return None
    details = ", ".join(
        f"joint{index + 1}={math.degrees(configuration[index]):.1f}deg outside "
        f"[{math.degrees(lower[index]):.1f}, {math.degrees(upper[index]):.1f}]"
        for index in outside
    )
    return f"joint limits would be exceeded ({details})"


def _refusal(code: SafetyCode, reason: str, details: Optional[Mapping[str, Any]] = None) -> PlanningOutcome:
    return PlanningOutcome(SafetyDecision.refuse(code, reason, details))
