"""A YamArm stand-in that models the 0xD latch, so keepalive can be tested off the bus.

The real arm is owned by whoever is doing bring-up; nothing in hwsupport/ opens
a CAN bus. This mirrors yam.arm.YamArm's control-path signatures and reuses the
driver's own JointConfig, SafetyLimits, ArmState, MotorFeedback and error codes,
so a signature drift in yam/ breaks these tests rather than the demo.

What it actually models, from yam/arm.py's measured note: an ENABLED motor that
stops receiving frames for longer than `latch_gap_s` latches 0xD, the latch
survives disable/enable, and clear_errors() is what clears it. Reads and
commands both count as frames, because both are a real exchange on the bus.
"""

import threading
import time
from typing import Dict, List, Optional, Sequence

from yam.arm import (
    ARM_JOINTS,
    GRIPPER_JOINT,
    ArmState,
    JointConfig,
    MotorCommunicationError,
    MotorFaultError,
    SafetyLimits,
)
from yam.dm_motor import COMMUNICATION_LOST, ERROR_MESSAGES, NORMAL_ERROR_CODE
from yam.dm_motor import MotorFeedback

#: yam/arm.py measured 0.3-0.4s silent gaps latching the gripper at 20Hz. The
#: bottom of that range is the pessimistic choice for a test.
DEFAULT_LATCH_GAP_S = 0.35

DISABLED_ERROR_CODE = 0x0
OVERLOAD_ERROR_CODE = 0xE


class Exchange:
    """One recorded bus exchange: what was sent, by which thread, when."""

    __slots__ = ("t", "kind", "thread", "targets", "gain_scale")

    def __init__(self, t: float, kind: str, thread: str, targets, gain_scale):
        self.t = t
        self.kind = kind
        self.thread = thread
        self.targets = targets
        self.gain_scale = gain_scale

    def __repr__(self) -> str:
        return f"Exchange({self.kind}, {self.thread}, t={self.t:.3f})"


class MockYamArm:
    """Duck-compatible with yam.arm.YamArm for everything Keepalive touches."""

    def __init__(
        self,
        joints: Optional[Sequence[JointConfig]] = None,
        safety: Optional[SafetyLimits] = None,
        latch_gap_s: float = DEFAULT_LATCH_GAP_S,
        resting: Optional[Sequence[float]] = None,
    ):
        self.joints: List[JointConfig] = (
            list(joints) if joints is not None else list(ARM_JOINTS) + [GRIPPER_JOINT]
        )
        self.safety = safety or SafetyLimits()
        self.latch_gap_s = latch_gap_s

        start = list(resting) if resting is not None else [j.clamp_position(0.0) for j in self.joints]
        self._position: Dict[int, float] = {j.motor_id: p for j, p in zip(self.joints, start)}
        self._error: Dict[int, int] = {j.motor_id: DISABLED_ERROR_CODE for j in self.joints}

        self._lock = threading.Lock()
        self._last_exchange_at: Optional[float] = None
        self._inject_comms = 0
        self._inject_fault: Optional[int] = None

        # Observability for the tests.
        self.exchanges: List[Exchange] = []
        self.max_gap_s = 0.0
        self.latched: List[str] = []
        self.enable_calls = 0
        self.clear_error_calls = 0
        self.recover_calls = 0
        self.read_state_calls = 0
        self.disable_calls = 0
        self.close_calls = 0

    # -- fault injection ---------------------------------------------------

    def inject_comms_fault(self, count: int = 1) -> None:
        """The next `count` exchanges raise MotorCommunicationError (motor silent)."""
        self._inject_comms = count

    def inject_motor_fault(self, error_code: int = OVERLOAD_ERROR_CODE) -> None:
        """Latch a physical error word on every motor: a fault that must NOT be cleared."""
        self._inject_fault = error_code

    # -- the modelled latch ------------------------------------------------

    def _touch(self, kind: str, targets=None, gain_scale=None) -> None:
        """Record an exchange and latch 0xD if the bus was silent for too long."""
        now = time.monotonic()
        if self._last_exchange_at is not None:
            gap = now - self._last_exchange_at
            self.max_gap_s = max(self.max_gap_s, gap)
            if gap > self.latch_gap_s:
                self._latch_comms_lost()
        self._last_exchange_at = now
        self.exchanges.append(
            Exchange(now, kind, threading.current_thread().name, targets, gain_scale)
        )

    def _latch_comms_lost(self) -> None:
        for joint in self.joints:
            if self._error[joint.motor_id] == NORMAL_ERROR_CODE:
                self._error[joint.motor_id] = COMMUNICATION_LOST
                if joint.name not in self.latched:
                    self.latched.append(joint.name)

    def _maybe_raise_injected(self, joint_name: str) -> None:
        if self._inject_comms > 0:
            self._inject_comms -= 1
            raise MotorCommunicationError(
                f"no response from {joint_name} (injected); expected 0x1{1:02X}, saw nothing"
            )

    def _feedback(self, joint: JointConfig) -> MotorFeedback:
        code = self._inject_fault if self._inject_fault is not None else self._error[joint.motor_id]
        return MotorFeedback(
            motor_id=joint.motor_id,
            error_code=code,
            position=self._position[joint.motor_id],
            velocity=0.0,
            torque=0.0,
            temperature_mos=32.0,
            temperature_rotor=30.0,
            raw_position=0,
        )

    def _to_state(self, feedback: Sequence[MotorFeedback]) -> ArmState:
        state = ArmState(
            positions=[fb.position for fb in feedback],
            velocities=[fb.velocity for fb in feedback],
            torques=[fb.torque for fb in feedback],
            feedback=list(feedback),
        )
        for joint, fb in zip(self.joints, state.feedback):
            if not fb.is_healthy:
                raise MotorFaultError(f"{joint.name}: {fb.error_message}")
            hottest = max(fb.temperature_mos, fb.temperature_rotor)
            if hottest > self.safety.max_temperature:
                raise MotorFaultError(f"{joint.name}: {hottest:.0f}C exceeds limit")
        return state

    # -- YamArm surface ----------------------------------------------------

    def read_state(self) -> ArmState:
        with self._lock:
            self.read_state_calls += 1
            self._touch("read_state")
            self._maybe_raise_injected(self.joints[0].name)
            return self._to_state([self._feedback(j) for j in self.joints])

    def command_positions(self, targets: Sequence[float], gain_scale: Optional[float] = None) -> ArmState:
        if len(targets) != len(self.joints):
            raise ValueError(f"expected {len(self.joints)} targets, got {len(targets)}")
        with self._lock:
            self._touch("command", tuple(targets), gain_scale)
            self._maybe_raise_injected(self.joints[0].name)
            for joint, target in zip(self.joints, targets):
                self._position[joint.motor_id] = joint.clamp_position(target)
            return self._to_state([self._feedback(j) for j in self.joints])

    def enable(self) -> ArmState:
        with self._lock:
            self.enable_calls += 1
            self._touch("enable")
            self._maybe_raise_injected(self.joints[0].name)
            for joint in self.joints:
                # A latched error word survives enable; only clear_errors() clears it.
                if self._error[joint.motor_id] in (DISABLED_ERROR_CODE, NORMAL_ERROR_CODE):
                    self._error[joint.motor_id] = NORMAL_ERROR_CODE
            return self._to_state([self._feedback(j) for j in self.joints])

    def clear_errors(self) -> None:
        with self._lock:
            self.clear_error_calls += 1
            self._touch("clear_errors")
            for joint in self.joints:
                if self._error[joint.motor_id] == COMMUNICATION_LOST:
                    self._error[joint.motor_id] = DISABLED_ERROR_CODE

    def recover_stale_motors(self) -> List[str]:
        with self._lock:
            self.recover_calls += 1
            self._touch("recover")
            stale = [j.name for j in self.joints if self._error[j.motor_id] == COMMUNICATION_LOST]
        if stale:
            self.clear_errors()
        return stale

    def disable(self) -> None:
        with self._lock:
            self.disable_calls += 1
            for joint in self.joints:
                self._error[joint.motor_id] = DISABLED_ERROR_CODE

    def close(self) -> None:
        self.close_calls += 1

    def reconnect(self) -> None:
        with self._lock:
            self._last_exchange_at = None

    # -- test helpers ------------------------------------------------------

    def error_messages(self) -> Dict[str, str]:
        return {
            j.name: ERROR_MESSAGES.get(self._error[j.motor_id], "unknown") for j in self.joints
        }

    def exchanges_between(self, t0: float, t1: float) -> List[Exchange]:
        return [e for e in self.exchanges if t0 <= e.t <= t1]
