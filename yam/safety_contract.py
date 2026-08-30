"""Data contracts shared by planning, execution, and the conversational layer.

Task recognition and motion safety answer different questions. A task can be
understood perfectly and still be unsafe in the current scene. The classes in
this module keep those outcomes separate and make the safe default explicit:
only an :class:`ApprovedPlan` may reach the guarded executor.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class SafetyCode(str, Enum):
    APPROVED = "approved"
    PREVIEW_ONLY = "preview_only"
    INVALID_REQUEST = "invalid_request"
    MAP_MISSING_PROVENANCE = "map_missing_provenance"
    MAP_STALE = "map_stale"
    MAP_SOURCE_CHANGED = "map_source_changed"
    CALIBRATION_MISSING = "calibration_missing"
    CALIBRATION_INVALID = "calibration_invalid"
    CALIBRATION_STALE = "calibration_stale"
    START_UNSAFE = "start_unsafe"
    GOAL_UNREACHABLE = "goal_unreachable"
    GOAL_UNSAFE = "goal_unsafe"
    PATH_NOT_FOUND = "path_not_found"
    PATH_DISCONTINUOUS = "path_discontinuous"
    PATH_UNSAFE = "path_unsafe"
    TRACKING_ENVELOPE_UNSAFE = "tracking_envelope_unsafe"
    SCENE_NOT_INTERLOCKED = "scene_not_interlocked"


def spoken_safety_refusal(reason: str) -> str:
    """Fixed wording suitable for TTS; an LLM never decides or edits this line."""
    detail = reason.strip().rstrip(".")
    return f"I understand the task, but I can't do it safely: {detail}. I won't move the arm."


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    code: SafetyCode
    reason: str
    spoken_response: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def approve(cls, details: Optional[Mapping[str, Any]] = None) -> "SafetyDecision":
        return cls(True, SafetyCode.APPROVED, "all required safety checks passed", "", details or {})

    @classmethod
    def preview(cls, reason: str, details: Optional[Mapping[str, Any]] = None) -> "SafetyDecision":
        return cls(False, SafetyCode.PREVIEW_ONLY, reason, spoken_safety_refusal(reason), details or {})

    @classmethod
    def refuse(
        cls,
        code: SafetyCode,
        reason: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> "SafetyDecision":
        if code in (SafetyCode.APPROVED, SafetyCode.PREVIEW_ONLY):
            raise ValueError(f"{code.value} is not a refusal code")
        return cls(False, code, reason, spoken_safety_refusal(reason), details or {})

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "code": self.code.value,
            "reason": self.reason,
            "spoken_response": self.spoken_response,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ApprovedPlan:
    """A path plus the evidence that authorized it.

    This is deliberately not a plain array. The executor hashes ``path`` again
    before use and checks the map/calibration identities, so a caller cannot
    plan one path and silently substitute another.
    """

    path: np.ndarray
    path_sha256: str
    map_sha256: str
    calibration_sha256: str
    issued_at_unix: float
    valid_for_seconds: float
    start_tolerance_rad: tuple[float, ...]
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = np.asarray(self.path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != 6:
            raise ValueError(f"approved path must have shape (N, 6), got {path.shape}")
        if not np.isfinite(path).all():
            raise ValueError("approved path contains non-finite values")
        if len(self.start_tolerance_rad) != 6:
            raise ValueError("start_tolerance_rad must contain six joint tolerances")
        object.__setattr__(self, "path", path.copy())


def normalized_tracking_bounds(values: Sequence[float] | float, joint_count: int = 6) -> np.ndarray:
    bounds = np.asarray(values, dtype=float)
    if bounds.ndim == 0:
        bounds = np.full(joint_count, float(bounds))
    if bounds.shape != (joint_count,):
        raise ValueError(f"tracking bounds must contain {joint_count} values")
    if not np.isfinite(bounds).all() or np.any(bounds <= 0.0):
        raise ValueError("tracking bounds must be finite and positive")
    return bounds
