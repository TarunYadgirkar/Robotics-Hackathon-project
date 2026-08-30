"""Control layer for the i2rt YAM 6-DoF arm over a CANable (gs_usb) adapter.

Joint limits and motor assignments come from the i2rt SDK's yam_pro_v1 config and
MJCF model. Every command is clamped to a joint limit, a torque ceiling and a
per-tick slew rate before it reaches the bus.
"""

import math
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import can

from yam import can_compat  # noqa: F401  (patches gs_usb for macOS)
from yam.dm_motor import (
    CLEAR_ERROR,
    COMMUNICATION_LOST,
    NORMAL_ERROR_CODE,
    DISABLE,
    ENABLE,
    FEEDBACK_ID_OFFSET,
    MOTOR_SPECS,
    MotorFeedback,
    MotorSpec,
    decode_feedback,
    encode_mit_command,
)

CAN_BITRATE = 1_000_000


@dataclass(frozen=True)
class JointConfig:
    name: str
    motor_id: int
    motor_type: str
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float
    max_torque: float

    @property
    def spec(self) -> MotorSpec:
        return MOTOR_SPECS[self.motor_type]

    def clamp_position(self, position: float) -> float:
        return min(max(position, self.lower_limit), self.upper_limit)


# Limits from i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml; gains from
# i2rt/robots/config/yam_pro_v1.yml. max_torque follows the MJCF actuatorfrcrange
# (+/-10 Nm), which is well under what the DM4340s could physically deliver.
ARM_JOINTS: List[JointConfig] = [
    JointConfig("joint1", 0x01, "DM4340", -2.61799, 3.14159, kp=80.0, kd=5.0, max_torque=10.0),
    JointConfig("joint2", 0x02, "DM4340", 0.0, 3.66519, kp=80.0, kd=5.0, max_torque=10.0),
    JointConfig("joint3", 0x03, "DM4340", 0.0, 3.14159, kp=80.0, kd=5.0, max_torque=10.0),
    JointConfig("joint4", 0x04, "DM4310", -1.69297, 1.5708, kp=40.0, kd=1.5, max_torque=10.0),
    JointConfig("joint5", 0x05, "DM4310", -1.5708, 1.5708, kp=10.0, kd=1.5, max_torque=10.0),
    JointConfig("joint6", 0x06, "DM4310", -2.0944, 2.0944, kp=10.0, kd=1.5, max_torque=10.0),
]

# The gripper is a linear stage whose travel depends on the jaws fitted, so it has no
# fixed limit the way the arm joints do -- i2rt calibrates it per robot. Until that
# calibration exists here, these bounds only cover the observed range and the gripper is
# excluded from motion commands; a guessed limit would move it on the first clamp.
# Known hazards, measured on this arm rather than inferred:
#
# * joint2 is the "raise the arm" trap. From a folded home pose it moves the tip
#   DOWN (0.165 -> 0.012 m) and self-collides against the base past roughly
#   +105 deg. joint3 is the joint that actually lifts, and is self-collision-free
#   across its whole range from home.
# * Motors can sit marginally outside these limits at rest -- j2/j3 read 0.0115 deg
#   below their 0.0 lower bound -- so the first clamped command nudges them.
# * Being inside the joint limits does NOT mean a pose is safe: about 10% of
#   random in-limit poses self-collide. Nothing in this module checks that; use
#   yam.environment.ArmSafetyChecker.
# * An enabled motor that stops receiving commands latches error 0xD
#   ("communication lost"). Measured: 20Hz polling with 0.3-0.4s silent gaps
#   latched the gripper; 100Hz with zero-torque frames through the gaps did not,
#   across five cycles. Anything that enables motors and then waits -- on a human,
#   on a file, on a network -- must keep streaming frames while it waits. The
#   error survives disable/enable and needs clear_errors().

#: Jaw travel per radian of motor, from linear_4310.yml (motor_stroke 6.57 rad
#: over gripper_stroke 0.096 m). Torque in Nm is grip force in newtons times this.
GRIPPER_METRES_PER_RAD = 0.096 / 6.57

#: Measured on this arm with the jaws currently fitted, 5 cycles at 0.2/0.35/0.5Nm.
#: The closed stop is a true hard stop and repeats to 0.1mm, so it is the datum.
#: The open stop is compliant -- it drifts 1.3mm as torque rises -- so it is not.
GRIPPER_CLOSED_POSITION = -5.158
GRIPPER_OPEN_POSITION = 0.056

#: Limits backed off 50mrad from each measured stop. kp is the SDK's 20.0, not the
#: 5.0 guessed here earlier. max_torque 0.6Nm is about 41N of grip; the 3.0Nm that
#: stood here allowed roughly 205N, which is a lot of force for a jaw to close with.
GRIPPER_JOINT = JointConfig(
    "gripper", 0x07, "DM4310",
    GRIPPER_CLOSED_POSITION + 0.05, GRIPPER_OPEN_POSITION - 0.05,
    kp=20.0, kd=0.5, max_torque=0.6,
)


def gripper_force_to_torque(newtons: float) -> float:
    return newtons * GRIPPER_METRES_PER_RAD


def gripper_opening_fraction(position: float) -> float:
    """0.0 fully closed, 1.0 fully open, from the motor angle."""
    span = GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION
    return float(min(max((position - GRIPPER_CLOSED_POSITION) / span, 0.0), 1.0))


class MotorCommunicationError(RuntimeError):
    pass


class MotorFaultError(RuntimeError):
    pass


@dataclass
class SafetyLimits:
    """Caps applied on top of the per-joint limits, as a single place to turn things down."""

    torque_scale: float = 1.0
    gain_scale: float = 1.0
    max_step_per_tick: float = 0.02  # rad; at 100 Hz this is ~2 rad/s
    max_temperature: float = 70.0


@dataclass
class ArmState:
    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    torques: List[float] = field(default_factory=list)
    feedback: List[MotorFeedback] = field(default_factory=list)

    def describe(self) -> str:
        return "\n".join(
            f"  {fb.motor_id}  pos={fb.position:+7.4f} rad ({math.degrees(fb.position):+8.2f}deg)  "
            f"vel={fb.velocity:+6.3f}  tau={fb.torque:+6.2f} Nm  "
            f"T={fb.temperature_mos:.0f}/{fb.temperature_rotor:.0f}C  [{fb.error_message}]"
            for fb in self.feedback
        )


class YamArm:
    def __init__(
        self,
        joints: Optional[Sequence[JointConfig]] = None,
        channel: int = 0,
        safety: Optional[SafetyLimits] = None,
        response_timeout: float = 0.05,
    ):
        self.joints = list(joints) if joints is not None else list(ARM_JOINTS)
        self.safety = safety or SafetyLimits()
        self.response_timeout = response_timeout
        self.bus = can.interface.Bus(interface="gs_usb", channel=channel, bitrate=CAN_BITRATE)
        self._enabled_ids: List[int] = []
        self._last_command: Dict[int, float] = {}

    # -- transport ---------------------------------------------------------

    def _exchange(self, joint: JointConfig, data: Sequence[int], retries: int = 5) -> MotorFeedback:
        """Send one frame to a motor and return its feedback frame."""
        expected_id = joint.motor_id + FEEDBACK_ID_OFFSET
        message = can.Message(arbitration_id=joint.motor_id, data=bytearray(data), is_extended_id=False)

        for _ in range(retries):
            self.bus.send(message)
            deadline = time.time() + self.response_timeout
            while time.time() < deadline:
                reply = self.bus.recv(timeout=self.response_timeout)
                if reply is None:
                    break
                # The adapter echoes our own transmissions; only real bus traffic counts.
                if reply.is_rx and reply.arbitration_id == expected_id:
                    return decode_feedback(reply.arbitration_id, reply.data, joint.spec)
            time.sleep(0.002)

        raise MotorCommunicationError(
            f"no response from {joint.name} (motor id 0x{joint.motor_id:02X}) after {retries} attempts"
        )

    def _drain(self) -> None:
        while self.bus.recv(timeout=0.005) is not None:
            pass

    # -- lifecycle ---------------------------------------------------------

    def recover_stale_motors(self) -> List[str]:
        """Clear the benign comms-timeout latch, and only that one.

        0xD means a motor sat enabled without a command stream -- a timeout, not
        damage, and it clears on request. Every other code (over-temperature,
        overcurrent, overload) reports a physical condition, so those are left
        latched rather than cleared out from under the operator.
        """
        stale = []
        for joint in self.joints:
            try:
                feedback = self._exchange(joint, encode_mit_command(joint.spec, position=0.0), retries=2)
            except MotorCommunicationError:
                continue
            if feedback.error_code == COMMUNICATION_LOST:
                stale.append(joint.name)

        if stale:
            self.clear_errors()
        return stale

    def enable(self) -> ArmState:
        """Enable every motor, clearing a comms-timeout latch per motor as it appears.

        Clearing all the motors up front does not hold: enabling is sequential,
        and a motor cleared at the start of the sweep can time out again while
        the motors ahead of it are still being enabled. So each motor is cleared
        and retried at the moment it reports the latch.
        """
        self._drain()
        feedback = []
        for joint in self.joints:
            fb = self._exchange(joint, ENABLE)
            # Retry while the motor reports anything but "enabled". A latched
            # 0xD needs clearing first; a 0x0 usually means we read a frame the
            # motor queued before the enable landed, so draining is enough.
            for _ in range(4):
                if fb.error_code == NORMAL_ERROR_CODE:
                    break
                if fb.error_code == COMMUNICATION_LOST:
                    self._clear_joint(joint)
                self._drain()
                time.sleep(0.01)
                fb = self._exchange(joint, ENABLE)
            feedback.append(fb)
            self._enabled_ids.append(joint.motor_id)
            self._last_command[joint.motor_id] = fb.position
        return self._to_state(feedback)

    def _clear_joint(self, joint: JointConfig) -> None:
        for _ in range(3):
            self.bus.send(can.Message(
                arbitration_id=joint.motor_id, data=bytearray(CLEAR_ERROR), is_extended_id=False
            ))
            time.sleep(0.002)

    def clear_errors(self) -> None:
        """Clear latched motor error words.

        Sent blind and repeated: a motor in a fault state may not answer, and the
        point of this call is to reach one that has stopped talking. Follow with
        enable() -- clearing the error does not re-enable the motor.
        """
        for joint in self.joints:
            self._clear_joint(joint)
        self._drain()

    def disable(self) -> None:
        for joint in self.joints:
            try:
                self._exchange(joint, DISABLE, retries=2)
            except MotorCommunicationError:
                # Best effort: a motor that has already dropped off cannot be told to stop,
                # and raising here would skip disabling the motors after it.
                pass
        self._enabled_ids.clear()

    def close(self) -> None:
        self.bus.shutdown()

    # -- state -------------------------------------------------------------

    def read_state(self) -> ArmState:
        """Read every joint with zero gains, so reading never applies torque."""
        feedback = [
            self._exchange(joint, encode_mit_command(joint.spec, position=0.0, kp=0.0, kd=0.0, torque=0.0))
            for joint in self.joints
        ]
        return self._to_state(feedback)

    def _to_state(self, feedback: Sequence[MotorFeedback]) -> ArmState:
        state = ArmState(
            positions=[fb.position for fb in feedback],
            velocities=[fb.velocity for fb in feedback],
            torques=[fb.torque for fb in feedback],
            feedback=list(feedback),
        )
        self._check_faults(state)
        return state

    def _check_faults(self, state: ArmState) -> None:
        for joint, fb in zip(self.joints, state.feedback):
            if not fb.is_healthy:
                raise MotorFaultError(f"{joint.name}: {fb.error_message}")
            hottest = max(fb.temperature_mos, fb.temperature_rotor)
            if hottest > self.safety.max_temperature:
                raise MotorFaultError(f"{joint.name}: {hottest:.0f}C exceeds {self.safety.max_temperature:.0f}C limit")

    # -- control -----------------------------------------------------------

    def command_positions(self, targets: Sequence[float], gain_scale: Optional[float] = None) -> ArmState:
        """One control tick: clamp to limits and slew rate, then send MIT commands."""
        if len(targets) != len(self.joints):
            raise ValueError(f"expected {len(self.joints)} targets, got {len(targets)}")

        scale = self.safety.gain_scale if gain_scale is None else gain_scale
        feedback = []
        for joint, target in zip(self.joints, targets):
            previous = self._last_command[joint.motor_id]
            step = min(max(target - previous, -self.safety.max_step_per_tick), self.safety.max_step_per_tick)
            commanded = joint.clamp_position(previous + step)

            frame = encode_mit_command(
                joint.spec,
                position=commanded,
                velocity=0.0,
                kp=joint.kp * scale,
                kd=joint.kd * scale,
                torque=0.0,
            )
            feedback.append(self._exchange(joint, frame))
            self._last_command[joint.motor_id] = commanded

        return self._to_state(feedback)

    def hold(self, targets: Optional[Sequence[float]] = None, duration: float = 5.0, rate_hz: float = 100.0) -> ArmState:
        """Hold a pose, ramping gains in from zero so nothing lurches on the first tick."""
        if targets is None:
            targets = self.read_state().positions

        period = 1.0 / rate_hz
        ramp_ticks = max(int(0.5 * rate_hz), 1)
        total_ticks = max(int(duration * rate_hz), 1)

        state = None
        for tick in range(total_ticks):
            gain_scale = self.safety.gain_scale * min(1.0, (tick + 1) / ramp_ticks)
            state = self.command_positions(targets, gain_scale=gain_scale)
            time.sleep(period)
        return state

    def move_to(self, targets: Sequence[float], duration: float = 3.0, rate_hz: float = 100.0) -> ArmState:
        """Interpolate from the current pose to `targets`, then hold briefly."""
        start = self.read_state().positions
        goal = [joint.clamp_position(t) for joint, t in zip(self.joints, targets)]

        ticks = max(int(duration * rate_hz), 1)
        period = 1.0 / rate_hz

        state = None
        for tick in range(ticks):
            alpha = (tick + 1) / ticks
            waypoint = [s + alpha * (g - s) for s, g in zip(start, goal)]
            state = self.command_positions(waypoint)
            time.sleep(period)
        return state


@contextmanager
def connected_arm(**kwargs):
    """Open the arm, guaranteeing motors are disabled on any exit path, including Ctrl-C."""
    arm = YamArm(**kwargs)
    previous_handler = signal.getsignal(signal.SIGINT)

    def stop_on_interrupt(signum, frame):
        arm.disable()
        signal.signal(signal.SIGINT, previous_handler)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_on_interrupt)
    try:
        yield arm
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        try:
            arm.disable()
        finally:
            arm.close()
