"""Intermittent-connection hunt. Wiggle the harness; this records what drops.

Read-only: every frame commands zero gain and zero torque, so the arm stays limp
throughout and you can move it and the cabling by hand freely.

Errors are cleared ONCE at startup to establish a clean baseline, and never
again -- the state a motor is in when it returns is the evidence:

  * back as 'disabled' (0x0) -> the motor REBOOTED, because a reboot is the only
    thing that puts an enabled motor back into the disabled state. That means it
    lost POWER.
  * back as 'enabled' or 'communication lost' (0xD) -> it kept running the whole
    time and simply could not be reached, so the CAN SIGNAL was interrupted.

The continuous polling matters: a DM motor left enabled with no command stream
latches 0xD on its own watchdog within seconds, which is why an idle arm shows
0xD everywhere and why that code means nothing unless it appears right after a
dropout.

Usage:
  .venv/bin/python wiggle_test.py            # 120 s
  .venv/bin/python wiggle_test.py 300        # 5 minutes
Ctrl-C stops early and still prints the summary.
"""

import collections
import sys
import time

import can

from yam import can_compat
from yam.arm import ARM_JOINTS, GRIPPER_JOINT
from yam.dm_motor import (CLEAR_ERROR, ENABLE, FEEDBACK_ID_OFFSET,
                          decode_feedback, encode_mit_command)

JOINTS = ARM_JOINTS + [GRIPPER_JOINT]


def poll(bus, joint, timeout=0.03):
    """One zero-torque read. Returns feedback, or None if the motor did not answer."""
    frame = encode_mit_command(joint.spec, position=0.0, kp=0.0, kd=0.0, torque=0.0)
    bus.send(can.Message(arbitration_id=joint.motor_id, data=frame, is_extended_id=False))
    deadline = time.time() + timeout
    while time.time() < deadline:
        reply = bus.recv(timeout=timeout)
        if reply is None:
            break
        if reply.is_rx and reply.arbitration_id == joint.motor_id + FEEDBACK_ID_OFFSET:
            return decode_feedback(reply.arbitration_id, reply.data, joint.spec)
    return None


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    bus = can_compat.open_bus()

    # Clear once, then enable, so every motor starts from a known good state.
    for joint in JOINTS:
        for _ in range(3):
            bus.send(can.Message(arbitration_id=joint.motor_id,
                                 data=bytearray(CLEAR_ERROR), is_extended_id=False))
            time.sleep(0.004)
    time.sleep(0.2)
    while bus.recv(timeout=0.01) is not None:
        pass
    for joint in JOINTS:
        bus.send(can.Message(arbitration_id=joint.motor_id,
                             data=bytearray(ENABLE), is_extended_id=False))
        time.sleep(0.02)
        while bus.recv(timeout=0.01) is not None:
            pass

    online = {j.name: True for j in JOINTS}
    status = {j.name: "enabled" for j in JOINTS}
    dropouts = collections.Counter()
    recovery_codes = collections.Counter()
    events = []
    started = time.time()

    print(f"Watching {len(JOINTS)} motors for {duration:.0f}s. The arm is LIMP.\n")
    print("Wiggle the harness -- at the arm base, and anywhere it crosses a moving")
    print("joint. Also try lifting/rotating each link by hand. Ctrl-C to stop early.\n")
    print(f"{'time':>7}  event")
    print(f"{'':>7}  {'-' * 60}")

    try:
        while time.time() - started < duration:
            now = time.time() - started
            for joint in JOINTS:
                feedback = poll(bus, joint)
                was, is_now = online[joint.name], feedback is not None

                if was and not is_now:
                    dropouts[joint.name] += 1
                    events.append((now, joint.name, "LOST"))
                    print(f"{now:7.2f}  {joint.name:>8}  LOST", flush=True)
                elif not was and is_now:
                    recovery_codes[f"{joint.name}:{feedback.error_message}"] += 1
                    events.append((now, joint.name, f"back [{feedback.error_message}]"))
                    print(f"{now:7.2f}  {joint.name:>8}  BACK, reporting "
                          f"[{feedback.error_message}]", flush=True)
                elif is_now and feedback.error_message != status[joint.name]:
                    events.append((now, joint.name, f"-> {feedback.error_message}"))
                    print(f"{now:7.2f}  {joint.name:>8}  now [{feedback.error_message}]", flush=True)

                online[joint.name] = is_now
                if is_now:
                    status[joint.name] = feedback.error_message
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n(stopped early)")
    finally:
        bus.shutdown()

    elapsed = time.time() - started
    print(f"\n{'=' * 68}\nRan {elapsed:.0f}s, {len(events)} events.\n")

    if not dropouts:
        print("No dropouts. Either the fault needs more provocation, or it is")
        print("triggered by the arm driving under load rather than by cable movement.")
        return

    print("Dropouts per motor:")
    for name, count in dropouts.most_common():
        print(f"  {name:>8}  {count}")

    print("\nHow motors came back -- THIS IS THE ANSWER:")
    for label, count in recovery_codes.most_common():
        print(f"  {label}  x{count}")

    rebooted = any(":disabled" in k for k in recovery_codes)
    stayed_up = any((":enabled" in k or "communication lost" in k) for k in recovery_codes)
    print()
    if rebooted:
        print("VERDICT: motors came back DISABLED, which only a reboot does.")
        print("  -> they lost POWER. Check the supply connector at the arm, the")
        print("     PSU itself, and any part of the power path that flexes.")
    elif stayed_up:
        print("VERDICT: motors came back still enabled (or watchdog-latched), so")
        print("  they never stopped running -- they were just unreachable.")
        print("  -> the CAN SIGNAL line is being interrupted. Check the CAN")
        print("     connector and the twisted pair, not the power supply.")

    if len(dropouts) == 1:
        print(f"\nOnly {list(dropouts)[0]} ever dropped: suspect its own stub/connector")
        print("rather than the trunk of the harness.")
    elif len(dropouts) == len(JOINTS):
        print("\nEvery motor dropped together: the break is upstream of all of them")
        print("-- the trunk, the arm's main connector, or the adapter end.")


if __name__ == "__main__":
    main()
