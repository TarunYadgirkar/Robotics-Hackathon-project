"""Guarded execution of a planned path.

A verified plan only protects against obstacles that are *in the model*. It says
nothing about model error -- bad kinematics, an obstacle enrolled a few
centimetres from where it really is, or something that moved after the scan. The
guard is what covers that, by watching the arm rather than the model.

WHAT DOES NOT WORK, measured on this arm rather than assumed:

* Tracking error and measured torque are NOT two independent signals. In MIT
  mode the motor's torque is kp * tracking_error (plus damping), so the two are
  proportional and one trips on the other's slack.
* A flat torque ceiling is useless here. Joint 3 draws ~9.5Nm lifting the arm's
  own weight from the folded pose, and up to 11Nm during a tuck -- the whole of
  the MJCF's 10Nm actuatorfrcrange. Any ceiling below that aborts every ordinary
  move; any ceiling above it no longer detects contact.
* Gravity feedforward from MuJoCo's `qfrc_bias` is not a fix, because the shipped
  inertial model does not match this hardware. It over-predicts in some poses
  (6.91Nm modelled against 0.02Nm measured, consistent with a shoulder
  counterbalance the model omits) and under-predicts in others (3.83 against
  7.66). The collision geometry is trustworthy; the mass properties are not.

WHAT THIS USES INSTEAD. Gravity load is large but changes *slowly and smoothly*
as the pose changes; a collision is a *step*. So the guard compares each joint's
torque against a slow exponential baseline of its own recent torque, and treats
the residual as the contact signal. That needs no mass model and no ceiling
tuned per trajectory. A hard absolute backstop and a generous tracking-error
limit sit behind it for the cases a step never appears in -- a gradual squeeze.

Every gravity number above was measured by a second session on this arm; see the
commit history for the raw walk-up.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from yam.arm import ArmState, YamArm


class ExecutionAborted(RuntimeError):
    def __init__(self, reason: str, state: Optional[ArmState] = None):
        super().__init__(reason)
        self.reason = reason
        self.state = state


@dataclass
class GuardLimits:
    #: Nm a joint may deviate from its own recent torque before we call it contact.
    max_torque_residual: float = 2.5
    #: Hard backstop for a gradual squeeze that never produces a step. Above the
    #: ~11Nm this arm draws under its own weight, so it only catches real trouble.
    absolute_torque: float = 13.0
    #: Generous, because gravity sag alone reaches ~0.22 rad on this arm. This is
    #: a backstop, not the primary signal.
    max_tracking_error: float = 0.35
    max_temperature: float = 65.0
    #: Seconds of exponential history the torque baseline averages over.
    baseline_seconds: float = 0.35
    #: Seconds before the residual guard arms, so the baseline can settle.
    warmup_seconds: float = 0.5
    require_free: bool = True


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
    def __init__(self, arm: YamArm, checker=None, limits: Optional[GuardLimits] = None):
        self.arm = arm
        self.checker = checker
        self.limits = limits or GuardLimits()
        self._baseline: Optional[np.ndarray] = None
        self._elapsed = 0.0

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
            worst = int(np.argmax(residual))
            report.peak_torque_residual = max(report.peak_torque_residual, float(residual.max()))
            if residual[worst] > self.limits.max_torque_residual:
                raise ExecutionAborted(
                    f"{self.arm.joints[worst].name} torque jumped {residual[worst]:.2f}Nm above its "
                    f"gravity baseline ({torque[worst]:+.2f}Nm vs {self._baseline[worst]:+.2f}Nm expected) "
                    f"-- treating that step as contact",
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

        worst = int(np.argmax(error))
        if error[worst] > self.limits.max_tracking_error:
            raise ExecutionAborted(
                f"{self.arm.joints[worst].name} is {np.degrees(error[worst]):.1f}deg behind its command "
                f"-- something is resisting the arm",
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
        path: Sequence[Sequence[float]],
        rate_hz: float = 100.0,
        gain_scale: float = 0.5,
        on_sample: Optional[Callable[[int, ArmState], None]] = None,
    ) -> ExecutionReport:
        report = ExecutionReport()
        period = 1.0 / rate_hz

        for index, waypoint in enumerate(path):
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
