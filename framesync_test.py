"""Is the 'dropout' actually the transport losing frame sync?

_exchange sends to one motor and waits for that motor's feedback id, discarding
whatever else arrives. command_positions walks the six joints in order, so the
bus carries a strict request/reply alternation -- and nothing resynchronises it.
If one reply ever arrives later than response_timeout, the next read consumes
the PREVIOUS motor's frame, every subsequent match fails, and after 5 retries a
motor that is alive and well is declared missing.

This logs the arbitration ids actually received whenever a match fails, which
tells the two apart:
  * ids arriving that belong to OTHER motors -> desync, a software bug.
  * nothing arriving at all                  -> the motor really is silent.
"""
import collections
import math
import time

import can
import numpy as np

from yam import can_compat
from yam.arm import ARM_JOINTS, SafetyLimits, YamArm
from yam.dm_motor import (CLEAR_ERROR, ENABLE, FEEDBACK_ID_OFFSET,
                          decode_feedback, encode_mit_command)

D = math.radians
RATE, GAIN = 100.0, 0.5
mismatches = collections.Counter()
events = []


def exchange_logged(arm, joint, data, retries=5):
    expected = joint.motor_id + FEEDBACK_ID_OFFSET
    message = can.Message(arbitration_id=joint.motor_id, data=bytearray(data), is_extended_id=False)
    seen = []
    for attempt in range(retries):
        arm.bus.send(message)
        deadline = time.time() + arm.response_timeout
        while time.time() < deadline:
            reply = arm.bus.recv(timeout=arm.response_timeout)
            if reply is None:
                seen.append("none")
                break
            if reply.is_rx and reply.arbitration_id == expected:
                if attempt:
                    events.append((joint.name, attempt, list(seen)))
                return decode_feedback(reply.arbitration_id, reply.data, joint.spec)
            seen.append(f"{'rx' if reply.is_rx else 'echo'}:0x{reply.arbitration_id:02X}")
            mismatches[f"want 0x{expected:02X} got "
                       f"{'rx' if reply.is_rx else 'echo'}:0x{reply.arbitration_id:02X}"] += 1
        time.sleep(0.002)
    events.append((joint.name, "FAIL", list(seen)))
    raise RuntimeError(f"{joint.name}: no match; saw {seen}")


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
        ready = home.copy(); ready[1], ready[2], ready[3] = D(18), D(75), D(45)
        current, failed = home, False

        for dps in [14, 18, 24, 30]:
            for goal in (ready, home):
                span = np.degrees(np.abs(goal - current)).max()
                ticks = max(int((span / dps) * RATE), 1)
                for t in range(ticks):
                    wp = current + (goal - current) * ((t + 1) / ticks)
                    for index, joint in enumerate(arm.joints):
                        previous = arm._last_command[joint.motor_id]
                        step = np.clip(wp[index] - previous, -arm.safety.max_step_per_tick,
                                       arm.safety.max_step_per_tick)
                        commanded = joint.clamp_position(previous + step)
                        frame = encode_mit_command(joint.spec, position=commanded, velocity=0.0,
                                                   kp=joint.kp * GAIN, kd=joint.kd * GAIN, torque=0.0)
                        try:
                            exchange_logged(arm, joint, frame)
                        except RuntimeError as error:
                            print(f"\nFAILED at {dps} deg/s: {error}")
                            failed = True
                            break
                        arm._last_command[joint.motor_id] = commanded
                    if failed:
                        break
                    time.sleep(1 / RATE)
                if failed:
                    break
                current = goal
            print(f"  {dps} deg/s ok", flush=True)
            if failed:
                break

        print(f"\nframe mismatches while running: {sum(mismatches.values())}")
        for label, count in mismatches.most_common(10):
            print(f"  {count:6d}  {label}")
        print(f"\nrecovered-after-retry / failure events: {len(events)}")
        for name, attempt, seen in events[-6:]:
            print(f"  {name}  attempt={attempt}  saw={seen[:8]}")
    finally:
        try: arm.disable()
        except Exception: pass
        try: arm.close()
        except Exception: pass


if __name__ == "__main__":
    main()
