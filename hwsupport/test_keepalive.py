"""Verify hwsupport/keepalive.py against the mock. Never touches a CAN bus.

    .venv/bin/python hwsupport/test_keepalive.py

Exits 0 when every check passes, 1 on the first failure. Real threads and real
wall-clock timing, because the thing under test is a cadence.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwsupport.keepalive import Keepalive, KeepaliveFaulted  # noqa: E402
from hwsupport.mock_arm import MockYamArm, OVERLOAD_ERROR_CODE  # noqa: E402
from yam.arm import MotorCommunicationError, MotorFaultError, SafetyLimits  # noqa: E402

LATCH_GAP_S = 0.35
RATE_HZ = 100.0

_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  PASS  {label}")
        return
    print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))
    raise SystemExit(1)


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def new_arm(**kwargs) -> MockYamArm:
    arm = MockYamArm(safety=SafetyLimits(gain_scale=0.5), latch_gap_s=LATCH_GAP_S, **kwargs)
    arm.enable()
    return arm


def kinds(alive) -> list:
    return [e.kind for e in alive.events()]


# -- [1] the mock's latch has teeth ------------------------------------------
def test_mock_latches_without_a_keepalive() -> None:
    print("[1] the 0xD latch the keepalive exists to prevent")
    arm = new_arm()
    time.sleep(LATCH_GAP_S + 0.1)
    try:
        arm.command_positions([0.0] * len(arm.joints))
    except MotorFaultError as exc:
        check("a silent gap past 0.35s latches every motor", "communication lost" in str(exc), str(exc))
    else:
        check("a silent gap past 0.35s latches every motor", False, "no fault raised")
    check("the latch names the motors it caught", len(arm.latched) == len(arm.joints), str(arm.latched))
    check("recover_stale_motors + enable clears it",
          arm.recover_stale_motors() and arm.enable().feedback[0].is_healthy)


# -- [2] latch prevention cadence --------------------------------------------
def test_keepalive_prevents_the_latch() -> None:
    print("[2] latch prevention cadence across a 1.2s silent beat gap")
    arm = new_arm()
    alive = Keepalive(arm, rate_hz=RATE_HZ).start()
    time.sleep(1.2)
    running = alive.is_running
    alive.stop()

    check("the thread survived the gap", running and alive.fault is None)
    check("no motor latched 0xD", arm.latched == [], str(arm.error_messages()))
    check(f"largest silence {arm.max_gap_s * 1000:.0f}ms stayed well under the 350ms latch gap",
          arm.max_gap_s < LATCH_GAP_S / 3, f"max_gap_s={arm.max_gap_s:.3f}")
    check(f"streamed {alive.ticks} ticks at ~{RATE_HZ:.0f} Hz", alive.ticks > 60, f"ticks={alive.ticks}")
    commands = [e for e in arm.exchanges if e.kind == "command"]
    check("held the resting pose, no commanded motion",
          all(abs(p) < 1e-9 for p in commands[-1].targets), str(commands[-1].targets))
    gains = [e.gain_scale for e in commands]
    check("gains ramped from near zero to the arm's gain_scale",
          gains[0] < gains[-1] and abs(gains[-1] - arm.safety.gain_scale) < 1e-9,
          f"first={gains[0]} last={gains[-1]}")


# -- [3] pause / resume around a scripted motion -----------------------------
def test_pause_and_resume_around_motion() -> None:
    print("[3] pause before a scripted motion, resume after")
    arm = new_arm()
    alive = Keepalive(arm, rate_hz=RATE_HZ).start()
    time.sleep(0.3)

    with alive.paused():
        # pause() has already waited for the in-flight tick, so the window opens
        # with the bus quiet. The motion owns it and keeps it fed itself, exactly
        # as arm/hw_backend.py's streamer does during a beat.
        window_start = time.monotonic()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            arm.command_positions([0.0] * len(arm.joints), gain_scale=arm.safety.gain_scale)
            time.sleep(1.0 / RATE_HZ)
        window_end = time.monotonic()

    time.sleep(0.3)
    alive.stop()

    main_name = threading.main_thread().name
    during = arm.exchanges_between(window_start, window_end)
    check("keepalive issued nothing while the motion held the bus",
          during and all(e.thread == main_name for e in during),
          str(sorted({e.thread for e in during})))
    after = [e for e in arm.exchanges if e.t > window_end and e.thread != main_name]
    check("keepalive resumed streaming after the motion", len(after) > 15, f"{len(after)} exchanges")
    check("resume re-read the pose the motion left behind", arm.read_state_calls >= 2,
          f"read_state_calls={arm.read_state_calls}")
    check("no motor latched across the handoff", arm.latched == [], str(arm.latched))
    check("event log records the handoff",
          "pause" in kinds(alive) and "resume" in kinds(alive), str(kinds(alive)))


# -- [4] one recovery pass, then surface -------------------------------------
def test_single_pass_recovery_then_surface() -> None:
    print("[4] fault injection -> one recovery -> surfaced error")
    arm = new_arm()
    alive = Keepalive(arm, rate_hz=RATE_HZ).start()
    time.sleep(0.15)

    arm.inject_comms_fault(1)
    check("recovered from the first comms error",
          wait_for(lambda: "recovered" in kinds(alive)), str(kinds(alive)))
    check("recovery was one recover_stale_motors + clear_errors + enable",
          arm.recover_calls == 1 and arm.clear_error_calls >= 1 and arm.enable_calls >= 2,
          f"recover={arm.recover_calls} clear={arm.clear_error_calls} enable={arm.enable_calls}")
    check("keepalive kept running after the recovery", alive.is_running and alive.fault is None)

    arm.inject_comms_fault(1)
    check("the second comms error stops the thread instead of retrying",
          wait_for(lambda: not alive.is_running), "thread still alive")
    check("no second recovery attempt was made", arm.recover_calls == 1, f"recover={arm.recover_calls}")
    check("the error is surfaced, not swallowed", isinstance(alive.fault, MotorCommunicationError),
          repr(alive.fault))
    try:
        alive.raise_if_faulted()
        check("raise_if_faulted() raises KeepaliveFaulted", False, "did not raise")
    except KeepaliveFaulted as exc:
        check("raise_if_faulted() raises KeepaliveFaulted", "Not retrying into it" in str(exc), str(exc))
    try:
        alive.pause()
        check("pause() refuses to hand over a faulted bus", False, "did not raise")
    except KeepaliveFaulted:
        check("pause() refuses to hand over a faulted bus", True)
    alive.stop()


def test_0xd_latch_gets_the_recovery_pass() -> None:
    print("[5] a latched 0xD surfaced as MotorFaultError is recoverable")
    arm = new_arm()
    alive = Keepalive(arm, rate_hz=RATE_HZ).start()
    time.sleep(0.1)
    with alive.paused():
        arm._latch_comms_lost()  # exactly what a silent gap does to the motors
    check("the latch is cleared in one pass", wait_for(lambda: "recovered" in kinds(alive)),
          str(kinds(alive)))
    check("only the comms latch was cleared", arm.recover_calls == 1)
    check("motors are healthy again",
          all(m == "enabled" for m in arm.error_messages().values()), str(arm.error_messages()))
    alive.stop()


def test_physical_fault_is_never_cleared() -> None:
    print("[6] a physical fault is surfaced immediately, never cleared")
    arm = new_arm()
    alive = Keepalive(arm, rate_hz=RATE_HZ).start()
    time.sleep(0.1)
    arm.inject_motor_fault(OVERLOAD_ERROR_CODE)

    check("the thread stops on the overload", wait_for(lambda: not alive.is_running), "still alive")
    check("no recovery was attempted", arm.recover_calls == 0 and arm.clear_error_calls == 0,
          f"recover={arm.recover_calls} clear={arm.clear_error_calls}")
    check("the overload is surfaced as MotorFaultError",
          isinstance(alive.fault, MotorFaultError) and "overload" in str(alive.fault),
          repr(alive.fault))
    alive.stop()


# -- [7] clean stop ----------------------------------------------------------
def test_clean_stop() -> None:
    print("[7] clean stop")
    arm = new_arm()
    faults = []
    alive = Keepalive(arm, rate_hz=RATE_HZ, on_fault=faults.append).start()
    time.sleep(0.3)
    alive.stop()

    check("the thread joined", not alive.is_running)
    check("no fault was raised", alive.fault is None and faults == [])
    check("stop() does NOT disable the motors or close the bus",
          arm.disable_calls == 0 and arm.close_calls == 0)
    check("the event log ends with the stop", kinds(alive)[-1] == "stop", str(kinds(alive)))
    logged = len(alive.events())
    alive.stop()
    check("stop() twice is a no-op", len(alive.events()) == logged)
    check("report() is printable for the demo", "start" in alive.report())
    try:
        alive.start()
        check("a stopped keepalive refuses to restart in place", False, "restarted")
    except RuntimeError:
        check("a stopped keepalive refuses to restart in place", True)


def main() -> None:
    started = time.monotonic()
    for test in (
        test_mock_latches_without_a_keepalive,
        test_keepalive_prevents_the_latch,
        test_pause_and_resume_around_motion,
        test_single_pass_recovery_then_surface,
        test_0xd_latch_gets_the_recovery_pass,
        test_physical_fault_is_never_cleared,
        test_clean_stop,
    ):
        test()
    print(f"=== all checks passed === ({_checks} checks, {time.monotonic() - started:.1f}s wall, exit 0)")


if __name__ == "__main__":
    main()
