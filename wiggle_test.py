"""Intermittent-connection hunt. Wiggle the harness; this records what drops.

Read-only: every frame commands zero gain and zero torque, so the arm stays limp
throughout and you can move it and the cabling by hand freely.

Deliberately does NOT clear latched errors. The error word a motor carries when
it comes back is the evidence we need:

  * back as 'communication lost' (0xD) -> it stayed POWERED and latched its own
    watchdog, so the CAN line was interrupted.
  * back clean as 'enabled' with nothing latched -> it was POWER-CYCLED.

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
from yam.dm_motor import FEEDBACK_ID_OFFSET, decode_feedback, encode_mit_command

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

    latched = any("communication lost" in k for k in recovery_codes)
    clean = any(k.endswith(":enabled") for k in recovery_codes)
    print()
    if latched and not clean:
        print("VERDICT: motors stayed powered and latched their watchdog.")
        print("  -> the CAN signal line is being interrupted. Check the CAN")
        print("     connector and the twisted pair, not the power supply.")
    elif clean and not latched:
        print("VERDICT: motors came back with no latched error at all.")
        print("  -> they lost POWER and rebooted. Check the supply connector,")
        print("     the PSU, and anything in the power path that flexes.")
    elif latched and clean:
        print("VERDICT: mixed -- some latched, some rebooted. Suggests a shared")
        print("  connector carrying both power and CAN, or two separate faults.")

    if len(dropouts) == 1:
        print(f"\nOnly {list(dropouts)[0]} ever dropped: suspect its own stub/connector")
        print("rather than the trunk of the harness.")
    elif len(dropouts) == len(JOINTS):
        print("\nEvery motor dropped together: the break is upstream of all of them")
        print("-- the trunk, the arm's main connector, or the adapter end.")


if __name__ == "__main__":
    main()
