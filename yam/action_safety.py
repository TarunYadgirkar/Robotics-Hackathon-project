"""Combine task understanding with motion safety without letting either blur into the other."""

from copy import deepcopy
from typing import Mapping

from yam.safe_planning import PlanningOutcome


def apply_motion_safety(task_decision: Mapping, planning: PlanningOutcome) -> dict:
    """Return the decision shape consumed by an action/voice orchestrator.

    ``task_decision`` may say the task is understood. A failed planning verdict
    replaces only the motion authorization and spoken response; it does not
    rewrite the recognition evidence or pretend the task was ambiguous.
    """
    result = deepcopy(dict(task_decision))
    result["motion_safety"] = planning.decision.to_dict()
    result["motion_allowed"] = bool(planning.decision.allowed)

    if planning.decision.allowed:
        result["motion_status"] = "approved"
        return result

    result["motion_status"] = "refused"
    result["response_kind"] = "safety_refusal"
    result["spoken_response"] = planning.decision.spoken_response
    return result
