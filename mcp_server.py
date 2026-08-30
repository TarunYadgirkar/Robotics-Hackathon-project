"""MCP server for the "I Don't Know How To Do That Yet" demo.

Exposes the demo's three faculties as tools any MCP client can drive:

  voice  robot_speak / robot_listen      (voice/speak.py, listen/transcribe.py)
  brain  robot_decide                    (brain/decide.py — computed, no LLM)
  arm    arm_status / arm_home / arm_gesture / arm_disconnect

The arm runs IN-PROCESS on purpose: exactly one process may own the CAN bus,
and holding the connection here keeps hwsupport's keepalive streaming between
tool calls — which is what lets can_pickup end holding a can and can_fling
play from that carrying pose minutes later. Corollary: while this server has
the arm connected, demo/run_demo.py must not run (call arm_disconnect first).

Voice and brain shell out to the same CLIs the stage demo uses, so anything
spoken or decided here is byte-identical to the show.

Run:      .venv/bin/python mcp_server.py           (stdio transport)
Register: claude mcp add yam-robot -- <repo>/.venv/bin/python <repo>/mcp_server.py
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
PY = str(REPO_ROOT / ".venv" / "bin" / "python")

from mcp.server.mcpserver import MCPServer  # noqa: E402

FLIP_FLOW = (
    "The flip-demo sequence, composed from these tools: "
    "1) arm_gesture('can_pickup') — really grips the can and ends holding it; "
    "2) robot_listen() for 'flip'; 3) robot_decide('flip') then robot_speak the "
    "honest abstain; 4) robot_listen() for 'hold it at the top'; "
    "5) arm_gesture('can_fling') — releases on the way up; the can not landing "
    "is expected. To back out while holding: arm_gesture('can_release')."
)

server = MCPServer(
    "yam-robot",
    instructions=(
        "Voice, decision brain and REAL robot-arm control for the Berkeley "
        "hackathon demo. Arm motions move a physical YAM arm — keep the "
        "workspace clear and never run demo/run_demo.py while the arm is "
        "connected here. " + FLIP_FLOW
    ),
)


def _run(cmd, timeout=120):
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                         timeout=timeout)
    return out.returncode, out.stdout.strip(), out.stderr.strip()


@server.tool()
def robot_speak(text: str, no_audio: bool = False) -> str:
    """Speak a line out loud in the demo's voice (ElevenLabs 'Eric', falls back
    to macOS `say`). Returns the text actually spoken. Set no_audio=True to
    render without playing (e.g. when testing over SSH)."""
    cmd = [PY, str(REPO_ROOT / "voice" / "speak.py"), "--say", text]
    if no_audio:
        cmd.append("--no-audio")
    rc, out, err = _run(cmd)
    if rc != 0:
        return f"speak failed (exit {rc}): {err or 'no stderr'}"
    return out or text


@server.tool()
def robot_listen(seconds: int = 6) -> str:
    """Capture `seconds` of microphone audio and transcribe it (mlx-whisper).
    Returns the transcript, or an error string. Note: macOS mic permission
    belongs to the process hosting this server; if capture returns nothing,
    grant Microphone access to that app or run the server from a Terminal that
    has it."""
    rc, out, err = _run(
        [PY, str(REPO_ROOT / "listen" / "transcribe.py"), "--seconds", str(seconds)],
        timeout=seconds + 60,
    )
    if rc != 0 or not out:
        return f"no transcript (exit {rc}): {err or 'empty capture — check mic permission'}"
    return out


@server.tool()
def robot_decide(query: str) -> dict:
    """Run the computed act/ask/abstain decision for a task query against the
    424-clip corpus (content-word coverage, no LLM). Returns the full decision
    JSON: tier, match_score, matched_task_id, evidence, utterance_slots."""
    rc, out, err = _run([PY, str(REPO_ROOT / "brain" / "decide.py"), query])
    if rc != 0:
        return {"error": f"decide.py exit {rc}", "stderr": err}
    return json.loads(out)


def _arm():
    from arm import arm_io
    return arm_io


@server.tool()
def arm_status() -> dict:
    """Backend name, whether the arm is simulated, available gesture names, and
    (hardware only) a passive joint read. Safe: enables no motors."""
    arm_io = _arm()
    info = {
        "backend": arm_io.backend_name(),
        "simulated": arm_io.is_simulated(),
        "gestures": list(arm_io.GESTURE_NAMES),
        "gesture_dir": str(arm_io.GESTURE_DIR),
    }
    if not arm_io.is_simulated():
        try:
            import math
            readings = arm_io.probe_passive()
            info["joints_deg"] = {
                name: round(math.degrees(fb.position), 2)
                for name, fb in readings[:6]
            }
        except BaseException as exc:  # noqa: BLE001 — surface, don't die
            info["arm_error"] = (
                f"{type(exc).__name__}: {exc} — if the CANable is missing, "
                "reseat its USB and run arm/precheck.py"
            )
    return info


@server.tool()
def arm_home() -> str:
    """Connect to the arm (if needed) and hold the current resting pose. Run
    this once before the first gesture. MOVES NOTHING but enables motors."""
    try:
        _arm().home()
        return "home complete — motors enabled, holding current pose"
    except BaseException as exc:  # noqa: BLE001
        return (f"arm unavailable: {type(exc).__name__}: {exc}. Reseat the "
                f"CANable, check arm power, then run .venv/bin/python arm/precheck.py")


@server.tool()
def arm_gesture(name: str, speed: float = 1.0) -> str:
    """Play a named gesture ON THE REAL ARM (mujoco-collision-checked, velocity
    capped, joint1 locked). Names: wake, attention, decline, point_screen,
    attempt, task_demo, approach_can, can_grip_top, can_grip_bottom,
    can_pickup (really grips the can and ENDS HOLDING IT — keepalive maintains
    the grip until the next gesture), can_fling (play only from can_pickup's
    end pose: releases the can on the way up), can_release (gentle put-down
    from the carrying pose). speed in (0,1] slows it down."""
    arm_io = _arm()
    path = Path(arm_io.GESTURE_DIR) / f"{name}.json"
    if not path.exists():
        return (f"unknown gesture {name!r}; available: "
                + ", ".join(sorted(p.stem for p in Path(arm_io.GESTURE_DIR).glob('*.json'))))
    try:
        arm_io.replay(path, speed=speed)
        return f"{name} complete"
    except BaseException as exc:  # noqa: BLE001
        return (f"{name} failed: {type(exc).__name__}: {exc}. If this is a "
                f"SoftLimitError the arm is not at the pose this gesture plays "
                f"from (can_fling/can_release need can_pickup's carrying pose); "
                f"if CAN-related, reseat the CANable and run arm/precheck.py")


@server.tool()
def arm_disconnect() -> str:
    """Release the CAN bus (stops keepalive; a held can will stay gripped only
    until motors disable). Required before running demo/run_demo.py in another
    process — only one process may own the bus."""
    try:
        _arm().shutdown()
        return "shutdown complete — the CAN bus is free"
    except BaseException as exc:  # noqa: BLE001
        return f"shutdown failed: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    server.run("stdio")
