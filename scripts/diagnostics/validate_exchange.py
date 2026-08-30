"""Validate the resynchronising _exchange on hardware, and time it.

Two questions:
  1. Do the dropouts stop? This drives the loaded lift over a speed ramp using
     the real control path, and reports YamArm.resyncs / .failures.
  2. What does the fix cost? _drain() blocks for one recv timeout whenever the
     queue is already empty, and it now runs before every attempt -- six joints
     per tick, so it has to fit inside the 10ms budget of a 100Hz loop.
"""
import math
import time

import numpy as np

from yam.arm import ARM_JOINTS, SafetyLimits, connected_arm
from yam.dm_motor import CLEAR_ERROR, ENABLE

import can

D = math.radians
RATE, GAIN = 100.0, 0.5


def main():
    with connected_arm(joints=ARM_JOINTS, safety=SafetyLimits(gain_scale=GAIN)) as arm:
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

        # how long does one full 6-joint tick take, with no sleep at all?
        probe = []
        for _ in range(200):
            started = time.perf_counter()
            arm.command_positions(home, gain_scale=GAIN)
            probe.append(time.perf_counter() - started)
        probe = np.array(probe) * 1000
        print(f"one 6-joint tick: median {np.median(probe):.2f}ms  p95 {np.percentile(probe,95):.2f}ms  "
              f"max {probe.max():.2f}ms")
        print(f"-> max sustainable rate ~{1000/np.percentile(probe,95):.0f} Hz "
              f"(need 100Hz = 10.00ms)\n")

        print(f"{'cycle':>6} {'deg/s':>6} {'peak tau':>9} {'resyncs':>8} {'failures':>9} {'achieved Hz':>12}")
        current = home
        cycle = 0
        for dps in [14, 20, 26, 32, 38, 44]:
            cycle += 1
            peak = 0.0
            ticks_done = 0
            started = time.perf_counter()
            for goal in (ready, home):
                span = np.degrees(np.abs(goal - current)).max()
                ticks = max(int((span / dps) * RATE), 1)
                for t in range(ticks):
                    wp = current + (goal - current) * ((t + 1) / ticks)
                    state = arm.command_positions(wp, gain_scale=GAIN)
                    peak = max(peak, max(abs(x) for x in state.torques))
                    ticks_done += 1
                    time.sleep(1 / RATE)
                current = goal
            achieved = ticks_done / (time.perf_counter() - started)
            print(f"{cycle:6d} {dps:6d} {peak:9.2f} {arm.resyncs:8d} {arm.failures:9d} "
                  f"{achieved:12.1f}", flush=True)

        print(f"\ntotals: resyncs={arm.resyncs}  failures={arm.failures}")


if __name__ == "__main__":
    main()
