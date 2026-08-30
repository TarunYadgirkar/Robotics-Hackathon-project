"""Load only hardware safety limits that are tied to an immutable raw trace."""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


CALIBRATION_SCHEMA_VERSION = 1


class CalibrationError(ValueError):
    pass


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HardwareSafetyCalibration:
    path: Path
    sha256: str
    robot_id: str
    measured_at_unix: float
    trace_path: Path
    trace_sha256: str
    max_tracking_error_rad: tuple[float, ...]
    max_torque_residual_nm: tuple[float, ...]
    absolute_torque_nm: float
    max_temperature_c: float
    baseline_seconds: float
    warmup_seconds: float
    rate_hz: float
    gain_scale: float

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        max_age_seconds: Optional[float] = None,
        now: Optional[float] = None,
    ) -> "HardwareSafetyCalibration":
        calibration_path = Path(path).resolve()
        if not calibration_path.is_file():
            raise CalibrationError(f"hardware safety calibration is missing: {calibration_path}")

        try:
            data = json.loads(calibration_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"cannot read hardware safety calibration: {error}") from error

        if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError(
                f"calibration schema must be {CALIBRATION_SCHEMA_VERSION}, "
                f"got {data.get('schema_version')!r}"
            )
        if data.get("hardware_validated") is not True:
            raise CalibrationError("calibration is not marked hardware_validated")

        robot_id = str(data.get("robot_id", "")).strip()
        if not robot_id:
            raise CalibrationError("calibration has no robot_id")

        measured_at = _finite_positive(data.get("measured_at_unix"), "measured_at_unix")
        if max_age_seconds is not None:
            max_age_seconds = _finite_positive(max_age_seconds, "max_age_seconds")
            age = (time.time() if now is None else float(now)) - measured_at
            if age < 0.0:
                raise CalibrationError("calibration timestamp is in the future")
            if age > max_age_seconds:
                raise CalibrationError(
                    f"hardware calibration is {age:.0f}s old; policy permits {max_age_seconds:.0f}s"
                )

        raw_trace = str(data.get("raw_trace_path", "")).strip()
        expected_trace_hash = str(data.get("raw_trace_sha256", "")).strip().lower()
        if not raw_trace or len(expected_trace_hash) != 64:
            raise CalibrationError("calibration must name a raw trace and its SHA-256")
        trace_path = Path(raw_trace)
        if not trace_path.is_absolute():
            trace_path = calibration_path.parent / trace_path
        trace_path = trace_path.resolve()
        if not trace_path.is_file():
            raise CalibrationError(f"raw hardware trace is missing: {trace_path}")
        actual_trace_hash = file_sha256(trace_path)
        if actual_trace_hash != expected_trace_hash:
            raise CalibrationError("raw hardware trace changed after calibration")

        tracking = _joint_values(data.get("max_tracking_error_rad"), "max_tracking_error_rad")
        residual = _joint_values(data.get("max_torque_residual_nm"), "max_torque_residual_nm")

        absolute_torque = _finite_positive(data.get("absolute_torque_nm"), "absolute_torque_nm")
        max_temperature = _finite_positive(data.get("max_temperature_c"), "max_temperature_c")
        baseline_seconds = _finite_positive(data.get("baseline_seconds"), "baseline_seconds")
        warmup_seconds = _finite_positive(data.get("warmup_seconds"), "warmup_seconds")
        _validate_raw_trace(
            trace_path,
            tracking,
            residual,
            absolute_torque,
            max_temperature,
            baseline_seconds,
            warmup_seconds,
        )

        return cls(
            path=calibration_path,
            sha256=file_sha256(calibration_path),
            robot_id=robot_id,
            measured_at_unix=measured_at,
            trace_path=trace_path,
            trace_sha256=actual_trace_hash,
            max_tracking_error_rad=tracking,
            max_torque_residual_nm=residual,
            absolute_torque_nm=absolute_torque,
            max_temperature_c=max_temperature,
            baseline_seconds=baseline_seconds,
            warmup_seconds=warmup_seconds,
            rate_hz=_finite_positive(data.get("rate_hz"), "rate_hz"),
            gain_scale=_bounded(data.get("gain_scale"), "gain_scale", 0.0, 1.0),
        )


def _finite_positive(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} must be a number") from error
    if not math.isfinite(number) or number <= 0.0:
        raise CalibrationError(f"{name} must be finite and positive")
    return number


def _bounded(value, name: str, low: float, high: float) -> float:
    number = _finite_positive(value, name)
    if not low < number <= high:
        raise CalibrationError(f"{name} must be in ({low}, {high}]")
    return number


def _joint_values(value, name: str) -> tuple[float, ...]:
    values = np.asarray(value, dtype=float)
    if values.shape != (6,):
        raise CalibrationError(f"{name} must contain six values")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise CalibrationError(f"{name} must contain finite positive values")
    return tuple(float(item) for item in values)


def _validate_raw_trace(
    trace_path: Path,
    tracking_limit: tuple[float, ...],
    residual_limit: tuple[float, ...],
    absolute_torque_limit: float,
    temperature_limit: float,
    baseline_seconds: float,
    warmup_seconds: float,
) -> None:
    """Recompute the observable claims instead of trusting the JSON summary."""
    try:
        with np.load(trace_path, allow_pickle=False) as trace:
            required = {
                "timestamps_unix",
                "commanded_positions_rad",
                "measured_positions_rad",
                "torques_nm",
                "temperatures_c",
                "deliberate_contact",
            }
            missing = sorted(required - set(trace.files))
            if missing:
                raise CalibrationError(f"raw trace is missing fields: {', '.join(missing)}")
            timestamps = np.asarray(trace["timestamps_unix"], dtype=float)
            commanded = np.asarray(trace["commanded_positions_rad"], dtype=float)
            measured = np.asarray(trace["measured_positions_rad"], dtype=float)
            torques = np.asarray(trace["torques_nm"], dtype=float)
            temperatures = np.asarray(trace["temperatures_c"], dtype=float)
            deliberate_contact = np.asarray(trace["deliberate_contact"], dtype=bool)
    except CalibrationError:
        raise
    except Exception as error:
        raise CalibrationError(f"cannot read raw hardware trace: {error}") from error

    sample_count = len(timestamps)
    expected_shape = (sample_count, 6)
    if sample_count < 2 or any(
        values.shape != expected_shape for values in (commanded, measured, torques, temperatures)
    ):
        raise CalibrationError("raw trace arrays must contain matching N x 6 samples")
    if deliberate_contact.shape != (sample_count,):
        raise CalibrationError("deliberate_contact must contain one flag per sample")
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0.0):
        raise CalibrationError("raw trace timestamps must be finite and strictly increasing")
    if not all(np.isfinite(values).all() for values in (commanded, measured, torques, temperatures)):
        raise CalibrationError("raw trace contains non-finite measurements")
    if not deliberate_contact.any() or deliberate_contact.all():
        raise CalibrationError("raw trace must include both free motion and deliberate contact")

    observed_tracking = np.abs(commanded - measured).max(axis=0)
    tracking_limit_array = np.asarray(tracking_limit)
    if np.any(tracking_limit_array < observed_tracking):
        joints = np.flatnonzero(tracking_limit_array < observed_tracking) + 1
        raise CalibrationError(
            f"tracking limit is below the observed trace on joints {joints.tolist()}"
        )

    baseline = torques[0].copy()
    residuals = np.zeros_like(torques)
    elapsed = 0.0
    for index in range(1, sample_count):
        period = timestamps[index] - timestamps[index - 1]
        elapsed += period
        residuals[index] = np.abs(torques[index] - baseline)
        alpha = min(1.0, period / baseline_seconds)
        baseline += alpha * (torques[index] - baseline)
        if elapsed < warmup_seconds:
            residuals[index] = 0.0

    free = ~deliberate_contact
    residual_limit_array = np.asarray(residual_limit)
    if np.any(residuals[free].max(axis=0) >= residual_limit_array):
        joints = np.flatnonzero(residuals[free].max(axis=0) >= residual_limit_array) + 1
        raise CalibrationError(
            f"contact residual threshold false-trips free-motion trace on joints {joints.tolist()}"
        )
    if not np.any(residuals[deliberate_contact] > residual_limit_array):
        raise CalibrationError("contact residual threshold detects none of the deliberate contacts")
    if np.abs(torques[free]).max() >= absolute_torque_limit:
        raise CalibrationError("absolute torque threshold false-trips the free-motion trace")
    if temperatures[free].max() >= temperature_limit:
        raise CalibrationError("temperature threshold is below a free-motion trace sample")
