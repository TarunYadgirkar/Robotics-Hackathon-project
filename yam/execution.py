"""Guarded execution of a planned path.

The residual-torque guard is an implementation awaiting hardware validation.
No raw calibration or deliberate-contact trace is present in this repository,
so its provisional thresholds cannot authorize motion.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from yam.arm import ArmState, YamArm
from yam.hardware_calibration import HardwareSafetyCalibration
from yam.safety_contract import ApprovedPlan, normalized_tracking_bounds


class ExecutionAborted(RuntimeError):
    def __init__(self, reason: str, state: Optional[ArmState] = None):
        super().__init__(reason)
        self.reason = reason
        self.state = state


@dataclass
class GuardLimits:
    #: Provisional values retained for offline guard tests only.
    max_torque_residual: Sequence[float] = (0.5, 1.5, 4.5, 2.0, 0.8, 0.8)
    absolute_torque: float = 13.0
    max_tracking_error: Sequence[float] | float = 0.10
    max_temperature: float = 65.0
    #: Seconds of exponential history the torque baseline averages over.
    baseline_seconds: float = 0.35
    #: Seconds before the residual guard arms, so the baseline can settle.
    warmup_seconds: float = 0.5
    require_free: bool = True
    hardware_validated: bool = False
    calibration_sha256: Optional[str] = None
    calibrated_rate_hz: Optional[float] = None
    calibrated_gain_scale: Optional[float] = None

    @classmethod
    def from_calibration(cls, calibration: HardwareSafetyCalibration) -> "GuardLimits":
        return cls(
            max_torque_residual=calibration.max_torque_residual_nm,
            absolute_torque=calibration.absolute_torque_nm,
            max_tracking_error=calibration.max_tracking_error_rad,
            max_temperature=calibration.max_temperature_c,
            baseline_seconds=calibration.baseline_seconds,
            warmup_seconds=calibration.warmup_seconds,
            hardware_validated=True,
            calibration_sha256=calibration.sha256,
            calibrated_rate_hz=calibration.rate_hz,
            calibrated_gain_scale=calibration.gain_scale,
        )


@dataclass
class ExecutionReport:
    completed: bool = False
    waypoints_sent: int = 0
    peak_tracking_error: float = 0.0
    peak_torque: float = 0.0
    peak_torque_residual: float = 0.0
    abort_reason: Optional[str] = None
    samples: List[dict] = field(default_factory=list)


class GuardedExecutor:
    def __init__(
        self,
        arm: YamArm,
        checker=None,
        limits: Optional[GuardLimits] = None,
        map_sha256: Optional[str] = None,
    ):
        self.arm = arm
        self.checker = checker
        self.limits = limits or GuardLimits()
        self.map_sha256 = map_sha256
        self._baseline: Optional[np.ndarray] = None
        self._elapsed = 0.0

    def _residual_allowance(self, count: int) -> np.ndarray:
        allowance = np.atleast_1d(np.asarray(self.limits.max_torque_residual, dtype=float))
        if allowance.size == 1:
            return np.full(count, float(allowance[0]))
        if allowance.size < count:
            # A gripper appended to the arm joints inherits the tightest allowance.
            return np.concatenate([allowance, np.full(count - allowance.size, allowance.min())])
        return allowance[:count]

    def _update_baseline(self, torque: np.ndarray, period: float) -> np.ndarray:
        """Slow exponential average of torque: what gravity is asking for right now."""
        if self._baseline is None:
            self._baseline = torque.copy()
            return self._baseline
        alpha = min(1.0, period / max(self.limits.baseline_seconds, 1e-3))
        self._baseline += alpha * (torque - self._baseline)
        return self._baseline

    def _check(self, target: np.ndarray, state: ArmState, report: ExecutionReport, period: float) -> None:
        measured = np.asarray(state.positions)
        error = np.abs(target - measured)
        torque = np.asarray(state.torques)

        report.peak_tracking_error = max(report.peak_tracking_error, float(error.max()))
        report.peak_torque = max(report.peak_torque, float(np.abs(torque).max()))

        # Compare against the baseline BEFORE folding this sample into it, so a
        # step change is measured against history rather than against itself.
        if self._baseline is not None and self._elapsed >= self.limits.warmup_seconds:
            residual = np.abs(torque - self._baseline)
            allowance = self._residual_allowance(len(torque))
            # Rank by how far each joint is through its own allowance, so the
            # joint in most trouble is reported rather than the loudest one.
            worst = int(np.argmax(residual / allowance))
            report.peak_torque_residual = max(report.peak_torque_residual, float(residual.max()))
            if residual[worst] > allowance[worst]:
                raise ExecutionAborted(
                    f"{self.arm.joints[worst].name} torque jumped {residual[worst]:.2f}Nm above its "
                    f"gravity baseline ({torque[worst]:+.2f}Nm vs {self._baseline[worst]:+.2f}Nm expected, "
                    f"allowance {allowance[worst]:.2f}Nm) -- treating that step as contact",
                    state,
                )

        self._update_baseline(torque, period)
        self._elapsed += period

        worst = int(np.argmax(np.abs(torque)))
        if abs(torque[worst]) > self.limits.absolute_torque:
            raise ExecutionAborted(
                f"{self.arm.joints[worst].name} is pulling {torque[worst]:+.2f}Nm, past the "
                f"{self.limits.absolute_torque:.1f}Nm backstop",
                state,
            )

        tracking_allowance = normalized_tracking_bounds(self.limits.max_tracking_error, len(error))
        worst = int(np.argmax(error / tracking_allowance))
        if error[worst] > tracking_allowance[worst]:
            raise ExecutionAborted(
                f"{self.arm.joints[worst].name} is {np.degrees(error[worst]):.1f}deg behind its command "
                f"(limit {np.degrees(tracking_allowance[worst]):.1f}deg) -- something is resisting the arm",
                state,
            )

        for joint, feedback in zip(self.arm.joints, state.feedback):
            if not feedback.is_healthy:
                raise ExecutionAborted(f"{joint.name} reports {feedback.error_message}", state)
            hottest = max(feedback.temperature_mos, feedback.temperature_rotor)
            if hottest > self.limits.max_temperature:
                raise ExecutionAborted(f"{joint.name} at {hottest:.0f}C", state)

    def run(
        self,
        path: ApprovedPlan,
        rate_hz: float = 100.0,
        gain_scale: float = 0.5,
        on_sample: Optional[Callable[[int, ArmState], None]] = None,
    ) -> ExecutionReport:
        if not self.limits.hardware_validated:
            raise ExecutionAborted(
                "contact guard is not hardware-validated; no calibration trace "
                "in this repository supports its provisional thresholds"
            )

        if not isinstance(path, ApprovedPlan):
            raise ExecutionAborted(
                "motion has no safety approval certificate; raw trajectories cannot command the arm"
            )
        if self.limits.calibration_sha256 != path.calibration_sha256:
            raise ExecutionAborted("the plan was approved under a different hardware calibration")
        if self.map_sha256 is None or self.map_sha256 != path.map_sha256:
            raise ExecutionAborted("the plan was approved against a different or unidentified workcell map")
        if time.time() > path.issued_at_unix + path.valid_for_seconds:
            raise ExecutionAborted("the path safety approval expired; observe the scene and plan again")
        actual_path_hash = hashlib.sha256(np.asarray(path.path, dtype="<f8").tobytes()).hexdigest()
        if actual_path_hash != path.path_sha256:
            raise ExecutionAborted("the path changed after it was safety-approved")
        if self.limits.calibrated_rate_hz is None or not np.isclose(
            rate_hz, self.limits.calibrated_rate_hz, rtol=0.0, atol=1e-9
        ):
            raise ExecutionAborted(
                f"execution rate {rate_hz:g}Hz does not match the calibrated rate "
                f"{self.limits.calibrated_rate_hz!r}Hz"
            )
        if self.limits.calibrated_gain_scale is None or not np.isclose(
            gain_scale, self.limits.calibrated_gain_scale, rtol=0.0, atol=1e-9
        ):
            raise ExecutionAborted(
                f"gain scale {gain_scale:g} does not match the calibrated gain "
                f"{self.limits.calibrated_gain_scale!r}"
            )

        start_state = self.arm.read_state()
        start_error = np.abs(np.asarray(start_state.positions) - path.path[0])
        start_tolerance = np.asarray(path.start_tolerance_rad)
        if np.any(start_error > start_tolerance):
            worst = int(np.argmax(start_error / start_tolerance))
            raise ExecutionAborted(
                f"live start pose differs from the approved path on {self.arm.joints[worst].name} "
                f"by {np.degrees(start_error[worst]):.1f}deg; re-plan from the live pose",
                start_state,
            )

        report = ExecutionReport()
        period = 1.0 / rate_hz

        for index, waypoint in enumerate(path.path):
            target = np.asarray(waypoint, dtype=float)

            if self.limits.require_free and self.checker is not None and not self.checker.is_free(target):
                # Never let the abort path itself fail: a checker without explain()
                # would raise AttributeError here instead of reporting the problem.
                describe = getattr(self.checker, "explain", None)
                detail = "; ".join(describe(target)) if callable(describe) else "no detail available"
                raise ExecutionAborted(f"waypoint {index} is not collision-free: {detail}")

            state = self.arm.command_positions(target, gain_scale=gain_scale)
            report.waypoints_sent += 1

            self._check(target, state, report, period)
            report.samples.append({
                "index": index,
                "target": target.tolist(),
                "measured": list(state.positions),
                "torque": list(state.torques),
            })
            if on_sample is not None:
                on_sample(index, state)
            time.sleep(period)

        report.completed = True
        return report
