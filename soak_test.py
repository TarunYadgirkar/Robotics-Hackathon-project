"""Soak the lift with a resynchronising exchange, to see if the dropouts stop.

yam.arm._exchange never resynchronises. On a late reply it breaks out and
re-sends, which puts another echo AND another reply into the RX queue. From then
on every read is one frame stale, so matches keep failing, so it keeps
re-sending -- and 250ms later a motor that is perfectly healthy is declared
missing. That matches what we measured: every motor answered within 20ms of such
a 'dropout', none had rebooted, and none had lost power.

The change here is one line of intent: drop anything stale BEFORE sending, so a
late frame costs one tick instead of poisoning every tick after it.
"""
import math
import time

import can
import numpy as np

from yam.arm import ARM_JOINTS, SafetyLimits, YamArm
from yam.dm_motor import (CLEAR_ERROR, ENABLE, FEEDBACK_ID_OFFSET,
                          decode_feedback, encode_mit_command)

D = math.radians
RATE, GAIN = 100.0, 0.5
stats = {"resyncs": 0, "retries": 0, "ticks": 0}


def exchange_resync(arm, joint, data, retries=5):
    expected = joint.motor_id + FEEDBACK_ID_OFFSET
    message = can.Message(arbitration_id=joint.motor_id, data=bytearray(data), is_extended_id=False)

    dropped = 0
    while arm.bus.recv(timeout=0.0) is not None:      # discard anything stale first
        dropped += 1
    if dropped:
        stats["resyncs"] += 1

    for attempt in range(retries):
        arm.bus.send(message)
        deadline = time.time() + arm.response_timeout
        while time.time() < deadline:
            reply = arm.bus.recv(timeout=arm.response_timeout)
            if reply is None:
                break
            if reply.is_rx and reply.arbitration_id == expected:
                return decode_feedback(reply.arbitration_id, reply.data, joint.spec)
            # anything else is the echo or a straggler; keep reading, do not resend
        stats["retries"] += 1
        while arm.bus.recv(timeout=0.0) is not None:
            pass
    raise RuntimeError(f"{joint.name}: no reply after {retries} resynchronised attempts")


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
        current = home
        peak_overall = 0.0

        print(f"{'cycle':>6} {'deg/s':>6} {'peak tau':>9} {'resyncs':>8} {'retries':>8}")
        cycle = 0
        for dps in [14, 20, 26, 32, 38, 44]:
            for repeat in range(2):
                cycle += 1
                peak = 0.0
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
                            fb = exchange_resync(arm, joint, encode_mit_command(
                                joint.spec, position=commanded, velocity=0.0,
                                kp=joint.kp * GAIN, kd=joint.kd * GAIN, torque=0.0))
                            arm._last_command[joint.motor_id] = commanded
                            peak = max(peak, abs(fb.torque))
                        stats["ticks"] += 1
                        time.sleep(1 / RATE)
                    current = goal
                peak_overall = max(peak_overall, peak)
                print(f"{cycle:6d} {dps:6d} {peak:9.2f} {stats['resyncs']:8d} "
                      f"{stats['retries']:8d}", flush=True)

        print(f"\n{stats['ticks']} ticks, {stats['ticks']*6} exchanges, no dropouts.")
        print(f"peak torque seen: {peak_overall:.2f}Nm")
        print(f"resyncs: {stats['resyncs']}   retries: {stats['retries']}")
    except RuntimeError as error:
        print(f"\nSTILL FAILED: {error}")
        print(f"after {stats['ticks']} ticks, {stats['resyncs']} resyncs, {stats['retries']} retries")
    finally:
        try: arm.disable()
        except Exception: pass
        try: arm.close()
        except Exception: pass


if __name__ == "__main__":
    main()
