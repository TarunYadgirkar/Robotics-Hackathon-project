"""Damiao (DM) servo protocol as used by the i2rt YAM arm.

Frame layout and scaling verified against i2rt/motor_drivers/dm_driver.py.
Commands go to the motor's CAN id; feedback comes back on id + 0x10.
"""

from dataclasses import dataclass
from typing import Dict

FEEDBACK_ID_OFFSET = 0x10

ENABLE = [0xFF] * 7 + [0xFC]
DISABLE = [0xFF] * 7 + [0xFD]
SET_ZERO = [0xFF] * 7 + [0xFE]
#: A DM motor latches its error word: disable/enable does NOT clear it. Confirmed
#: on hardware -- a gripper stuck in 0xD survived two disable/enable cycles and
#: cleared instantly on this. Without it the only recovery is a power cycle.
CLEAR_ERROR = [0xFF] * 7 + [0xFB]

KP_MAX = 500.0
KD_MAX = 5.0

ERROR_MESSAGES = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "overvoltage",
    0x9: "undervoltage",
    0xA: "overcurrent",
    0xB: "MOS over-temperature",
    0xC: "coil over-temperature",
    0xD: "communication lost",
    0xE: "overload",
}
NORMAL_ERROR_CODE = 0x1
#: Sat enabled without a command stream. A timeout, not damage.
COMMUNICATION_LOST = 0xD


@dataclass(frozen=True)
class MotorSpec:
    name: str
    position_max: float
    velocity_max: float
    torque_max: float


MOTOR_SPECS: Dict[str, MotorSpec] = {
    "DM4310": MotorSpec("DM4310", position_max=12.5, velocity_max=30.0, torque_max=10.0),
    "DM4340": MotorSpec("DM4340", position_max=12.5, velocity_max=10.0, torque_max=28.0),
}


def float_to_uint(value: float, low: float, high: float, bits: int) -> int:
    span = high - low
    clamped = min(max(value, low), high)
    return int((clamped - low) * ((1 << bits) - 1) / span)


def uint_to_float(value: int, low: float, high: float, bits: int) -> float:
    span = high - low
    return value * span / ((1 << bits) - 1) + low


@dataclass(frozen=True)
class MotorFeedback:
    motor_id: int
    error_code: int
    position: float
    velocity: float
    torque: float
    temperature_mos: float
    temperature_rotor: float
    raw_position: int

    @property
    def is_healthy(self) -> bool:
        return self.error_code == NORMAL_ERROR_CODE

    @property
    def error_message(self) -> str:
        return ERROR_MESSAGES.get(self.error_code, f"unknown (0x{self.error_code:X})")


def encode_mit_command(
    spec: MotorSpec,
    position: float,
    velocity: float = 0.0,
    kp: float = 0.0,
    kd: float = 0.0,
    torque: float = 0.0,
) -> bytearray:
    position_raw = float_to_uint(position, -spec.position_max, spec.position_max, 16)
    velocity_raw = float_to_uint(velocity, -spec.velocity_max, spec.velocity_max, 12)
    kp_raw = float_to_uint(kp, 0.0, KP_MAX, 12)
    kd_raw = float_to_uint(kd, 0.0, KD_MAX, 12)
    torque_raw = float_to_uint(torque, -spec.torque_max, spec.torque_max, 12)

    return bytearray([
        (position_raw >> 8) & 0xFF,
        position_raw & 0xFF,
        (velocity_raw >> 4) & 0xFF,
        ((velocity_raw & 0xF) << 4) | (kp_raw >> 8),
        kp_raw & 0xFF,
        (kd_raw >> 4) & 0xFF,
        ((kd_raw & 0xF) << 4) | (torque_raw >> 8),
        torque_raw & 0xFF,
    ])


def decode_feedback(arbitration_id: int, data: bytes, spec: MotorSpec) -> MotorFeedback:
    position_raw = (data[1] << 8) | data[2]
    velocity_raw = (data[3] << 4) | (data[4] >> 4)
    torque_raw = ((data[4] & 0xF) << 8) | data[5]

    return MotorFeedback(
        motor_id=arbitration_id - FEEDBACK_ID_OFFSET,
        error_code=(data[0] & 0xF0) >> 4,
        position=uint_to_float(position_raw, -spec.position_max, spec.position_max, 16),
        velocity=uint_to_float(velocity_raw, -spec.velocity_max, spec.velocity_max, 12),
        torque=uint_to_float(torque_raw, -spec.torque_max, spec.torque_max, 12),
        temperature_mos=float(data[6]),
        temperature_rotor=float(data[7]),
        raw_position=position_raw,
    )
