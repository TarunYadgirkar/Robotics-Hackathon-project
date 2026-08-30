"""Reader for coordination/FACTS.md — the single source of environment truth.

No constant in arm/ may shadow a FACTS key; everything hardware-conditional routes
through here so the flag cannot drift from what P0 measured.
"""

from pathlib import Path
import os
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = REPO_ROOT / "coordination" / "FACTS.md"

_KEY_LINE = re.compile(r"^([A-Z][A-Z0-9_]*):\s*(.*)$")


class FactsError(RuntimeError):
    pass


def _load() -> dict[str, str]:
    if not FACTS_PATH.exists():
        raise FactsError(
            f"{FACTS_PATH} not found. arm/ refuses to guess the environment; "
            "run Agent P0 first (protocol rule 1)."
        )
    facts: dict[str, str] = {}
    for line in FACTS_PATH.read_text().splitlines():
        m = _KEY_LINE.match(line.strip())
        if m and m.group(1) not in facts:
            facts[m.group(1)] = m.group(2).strip()
    return facts


FACTS = _load()


def get(key: str, default: str | None = None) -> str:
    value = FACTS.get(key, default)
    if value is None:
        raise FactsError(f"{key} missing from {FACTS_PATH}")
    return value


FORCE_SIM_ENV = "ARM_FORCE_SIM"


def force_sim() -> bool:
    """Operator override: run the simulator even with an arm plugged in.

    For F's `--rehearse`, so a full demo run can be walked through without
    energising the robot. It can only ever move in the safe direction — there is
    deliberately no env var that forces hardware ON, because that would let a
    stale shell variable outrank FACTS.md and command a bus nobody verified.
    """
    return os.environ.get(FORCE_SIM_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def hardware_present() -> bool:
    """True only for an explicit 'yes', and never when ARM_FORCE_SIM is set.

    'uncertain' resolves to False on purpose: commanding motion over a link P0
    could not confirm is the failure mode this whole flag exists to prevent.
    """
    if force_sim():
        return False
    return get("HARDWARE_PRESENT").split()[0].lower() == "yes"


def hardware_flag_raw() -> str:
    return get("HARDWARE_PRESENT")


def device_path() -> str | None:
    raw = get("HARDWARE_DEVICE_PATH", "none")
    return None if raw.lower().startswith("none") else raw.split()[0]
