"""voice/smoke.py — smoke test for voice/speak.py. Exit 0 on pass.

Runs speak.py --no-audio over the fixture decision JSONs in voice/fixtures/
(conforming to the frozen decision-JSON schema) and asserts the rendered
text contains no unfilled "{" template placeholder. --no-audio is used so
this runs headlessly/deterministically; TTS playback itself (ElevenLabs
dry-run + macOS say) is verified separately and recorded in
coordination/status/D.json, not by this script.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEAK = HERE / "speak.py"
PYTHON = HERE.parent / ".venv" / "bin" / "python"
FIXTURES = sorted((HERE / "fixtures").glob("*.json"))


def main():
    assert FIXTURES, "no fixture decision JSONs found in voice/fixtures/"
    for fixture in FIXTURES:
        result = subprocess.run(
            [str(PYTHON), str(SPEAK), str(fixture), "--no-audio"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{fixture.name}: exited {result.returncode}: {result.stderr}"
        text = result.stdout.rstrip("\n")
        assert "{" not in text, f"{fixture.name}: unfilled template placeholder in {text!r}"
        print(f"[smoke] {fixture.name}: {text!r}")

    print(f"[smoke] voice/speak.py: PASS ({len(FIXTURES)} fixtures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
