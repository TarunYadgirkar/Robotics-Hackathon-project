"""Reproduce the dropout under LOAD, and capture what actually failed.

Manual wiggling at speed produced nothing, so the trigger is not cable flex.
Both real failures happened while the arm was driving. This drives it, and when
it drops, answers two questions that need different fixes:

  1. Did the ARM die, or did the ADAPTER go bus-off?
     After a loss we keep polling the SAME bus handle. If motors return on it,
     the adapter was fine and the arm went away. If the same handle stays deaf
     but a freshly opened one works, our CAN controller had gone bus-off --
     a host-side fault, fixable in software.

  2. If the arm died: power or signal?
     A motor that returns 'disabled' rebooted, so it lost POWER. One that
     returns 'enabled' or 'communication lost' kept running and was merely
     unreachable, so the CAN SIGNAL dropped.

Errors are cleared once at the start and never again.
"""
import math
import time

import can
import numpy as np

from yam import can_compat
from yam.arm import (ARM_JOINTS, MotorCommunicationError, MotorFaultError,
                     SafetyLimits, YamArm)
from yam.dm_motor import (CLEAR_ERROR, ENABLE, FEEDBACK_ID_OFFSET,
                          decode_feedback, encode_mit_command)

D = math.radians
RATE, GAIN = 100.0, 0.5


def raw_poll(bus, joint, timeout=0.03):
    frame = encode_mit_command(joint.spec, position=0.0, kp=0.0, kd=0.0, torque=0.0)
    try:
        bus.send(can.Message(arbitration_id=joint.motor_id, data=frame, is_extended_id=False))
    except Exception:
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        reply = bus.recv(timeout=timeout)
        if reply is None:
            break
        if reply.is_rx and reply.arbitration_id == joint.motor_id + FEEDBACK_ID_OFFSET:
            return decode_feedback(reply.arbitration_id, reply.data, joint.spec)
    return None


def investigate(arm):
    """Called right after the bus goes quiet. Does not clear anything."""
    print("\n--- dropout: investigating (no errors cleared) ---")

    print("  A. same bus handle, 15s:")
    found = {}
    started = time.time()
    while time.time() - started < 15.0:
        for joint in ARM_JOINTS:
            if joint.name in found:
                continue
            fb = raw_poll(arm.bus, joint)
            if fb is not None:
                found[joint.name] = fb.error_message
                print(f"     {time.time()-started:5.2f}s  {joint.name} back as "
                      f"[{fb.error_message}]", flush=True)
        if len(found) == len(ARM_JOINTS):
            break
        time.sleep(0.05)

    if found:
        print(f"  -> {len(found)}/{len(ARM_JOINTS)} returned on the SAME handle: "
              "the adapter was fine, the ARM went away.")
    else:
        print("  -> nothing on the same handle. Trying a fresh one.")
        try:
            arm.bus.shutdown()
        except Exception:
            pass
        time.sleep(2.0)
        fresh = can_compat.open_bus()
        print("  B. freshly opened bus, 10s:")
        started = time.time()
        while time.time() - started < 10.0:
            for joint in ARM_JOINTS:
                if joint.name in found:
                    continue
                fb = raw_poll(fresh, joint)
                if fb is not None:
                    found[joint.name] = fb.error_message
                    print(f"     {time.time()-started:5.2f}s  {joint.name} back as "
                          f"[{fb.error_message}]", flush=True)
            if len(found) == len(ARM_JOINTS):
                break
            time.sleep(0.05)
        if found:
            print("  -> only a FRESH handle works: our CAN controller had gone BUS-OFF.")
            print("     Host-side. Recoverable in software by reopening the bus.")
        else:
            print("  -> still nothing. The arm is genuinely unpowered.")
        try:
            fresh.shutdown()
        except Exception:
            pass

    if found:
        rebooted = [n for n, c in found.items() if c == "disabled"]
        alive = [n for n, c in found.items() if c in ("enabled", "communication lost")]
        print()
        if rebooted:
            print(f"  VERDICT: {len(rebooted)} motor(s) came back DISABLED -> they REBOOTED.")
            print("           That is a POWER loss. Check the supply and its connector.")
        elif alive:
            print(f"  VERDICT: {len(alive)} motor(s) still enabled/watchdog-latched -> they")
            print("           never stopped. That is a CAN SIGNAL interruption.")
    return found


def main():
    arm = YamArm(joints=ARM_JOINTS, safety=SafetyLimits(gain_scale=GAIN))
    try:
        for joint in ARM_JOINTS:
            for _ in range(3):
                arm.bus.send(can.Message(arbitration_id=joint.motor_id,
                                         data=bytearray(CLEAR_ERROR), is_extended_id=False))
                time.sleep(0.004)
        time.sleep(0.25)
        arm._drain()
        for joint in ARM_JOINTS:
            arm._exchange(joint, ENABLE, retries=10)
            time.sleep(0.02)

        home = np.array(arm.enable().positions)
        ready = home.copy()
        ready[1], ready[2], ready[3] = D(18), D(75), D(45)
        print(f"base pinned at {math.degrees(home[0]):+.2f}deg")
        print(f"{'pass':>5} {'deg/s':>6} {'peak tau':>9}  result")

        current = home
        try:
            for index, dps in enumerate([14, 18, 24, 30, 36, 42], start=1):
                for goal in (ready, home):
                    span = np.degrees(np.abs(goal - current)).max()
                    ticks = max(int((span / dps) * RATE), 1)
                    peak = 0.0
                    for t in range(ticks):
                        wp = current + (goal - current) * ((t + 1) / ticks)
                        state = arm.command_positions(wp, gain_scale=GAIN)
                        peak = max(peak, max(abs(x) for x in state.torques))
                        time.sleep(1 / RATE)
                    current = goal
                print(f"{index:5d} {dps:6d} {peak:9.2f}  ok", flush=True)
            print("\nNo dropout across the whole ramp.")
        except (MotorCommunicationError, MotorFaultError) as error:
            print(f"\nFAILED: {error}")
            investigate(arm)
    finally:
        try:
            arm.disable()
        except Exception:
            pass
        try:
            arm.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
