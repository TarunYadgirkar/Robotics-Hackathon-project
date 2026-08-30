"""Fault triage for the YAM arm demo -- static lookup, never touches the bus.

  .venv/bin/python hwsupport/triage.py --list
  .venv/bin/python hwsupport/triage.py --symptom "0xD comms lost"
  .venv/bin/python hwsupport/triage.py --validate
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "failure_modes.json"

REQUIRED_KEYS = {"symptom", "cause", "remedy", "source", "severity"}
VALID_SEVERITIES = {"recoverable", "stop_the_demo"}


def load_entries() -> list:
    with open(DATA_PATH, "r") as handle:
        return json.load(handle)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def score_entry(query: str, entry: dict) -> float:
    haystack = " ".join([entry["symptom"], entry.get("cause", "")])
    query_tokens = _tokenize(query)
    hay_tokens = _tokenize(haystack)
    if not query_tokens or not hay_tokens:
        return 0.0
    overlap = len(query_tokens & hay_tokens) / len(query_tokens)
    ratio = difflib.SequenceMatcher(None, query.lower(), entry["symptom"].lower()).ratio()
    return 0.7 * overlap + 0.3 * ratio


def find_matches(query: str, entries: list, top_n: int = 3) -> list:
    scored = [(score_entry(query, entry), entry) for entry in entries]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [(score, entry) for score, entry in scored[:top_n] if score > 0]


def cmd_symptom(query: str) -> int:
    entries = load_entries()
    matches = find_matches(query, entries)
    if not matches:
        print(f"No match for: {query!r}")
        print("Try --list to see all known symptoms.")
        return 1

    best_score, best = matches[0]
    print(f"MATCH ({best_score:.0%}): {best['symptom']}")
    print(f"  severity: {best['severity']}")
    print(f"  cause:    {best['cause']}")
    print(f"  remedy:   {best['remedy']}")
    print(f"  source:   {best['source']}")

    rest = matches[1:]
    if rest:
        print("\nOther possible matches:")
        for score, entry in rest:
            print(f"  ({score:.0%}) {entry['symptom'][:90]}")
    return 0


def cmd_list() -> int:
    entries = load_entries()
    width = max(len(e["symptom"]) for e in entries)
    width = min(width, 70)
    for entry in entries:
        symptom = entry["symptom"]
        if len(symptom) > width:
            symptom = symptom[: width - 1] + "…"
        print(f"[{entry['severity']:>13}]  {symptom:<{width}}  {entry['source'].split(';')[0]}")
    print(f"\n{len(entries)} entries.")
    return 0


def cmd_validate() -> int:
    entries = load_entries()
    errors = []

    if not isinstance(entries, list) or not entries:
        print("FAIL: failure_modes.json must be a non-empty JSON array")
        return 1

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index}: not an object")
            continue

        missing = REQUIRED_KEYS - entry.keys()
        if missing:
            errors.append(f"entry {index}: missing keys {sorted(missing)}")
            continue

        for key in REQUIRED_KEYS:
            if not isinstance(entry[key], str) or not entry[key].strip():
                errors.append(f"entry {index} ({entry.get('symptom', '?')[:40]!r}): '{key}' must be a non-empty string")

        if entry["severity"] not in VALID_SEVERITIES:
            errors.append(
                f"entry {index} ({entry['symptom'][:40]!r}): severity {entry['severity']!r} "
                f"not in {sorted(VALID_SEVERITIES)}"
            )

        paths = re.findall(r"([A-Za-z0-9_./-]+\.py)", entry.get("source", ""))
        if not paths:
            errors.append(f"entry {index} ({entry['symptom'][:40]!r}): source cites no .py file")
        for rel_path in paths:
            if not (REPO_ROOT / rel_path).exists():
                errors.append(f"entry {index} ({entry['symptom'][:40]!r}): cited file does not exist: {rel_path}")

    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"OK: {len(entries)} entries, schema valid, all cited files exist.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symptom", type=str, help="fuzzy-match a symptom description and print its remedy")
    group.add_argument("--list", action="store_true", help="print a compact table of all known failure modes")
    group.add_argument("--validate", action="store_true", help="validate the JSON schema and cited file paths")
    args = parser.parse_args()

    if args.validate:
        return cmd_validate()
    if args.list:
        return cmd_list()
    return cmd_symptom(args.symptom)


if __name__ == "__main__":
    sys.exit(main())
