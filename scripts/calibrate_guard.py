"""Record the raw trace the motion guard's thresholds are derived from.

The guard distinguishes contact from gravity by watching each joint's torque
against a slow baseline of its own recent torque. Thresholds that separate the
two cannot be assumed; they are a property of this arm. This records both
classes on this hardware and fits thresholds to the recording:

  free motion    -- small motions in open air, nothing touching the arm
  contact        -- the same hold, while a person presses the arm by hand

Contact is made by hand rather than by driving into geometry, so the capture
never commands the arm toward an obstacle.

  python scripts/calibrate_guard.py --robot-id yam-pro-01
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yam.arm import ARM_JOINTS, connected_arm
from yam.hardware_calibration import HardwareSafetyCalibration, file_sha256

HOME = [0.0498, -0.0002, 0.0002, -0.0906, 0.0734, 1.1706]
BASELINE_SECONDS = 0.35
WARMUP_SECONDS = 0.5


def countdown(label: str, seconds: int, arm=None, hold_at=None, rate_hz: float = 100.0) -> None:
    """Count down while holding position.

    An enabled DM motor that stops receiving commands latches error 0xD on its
    own watchdog, so a silent pause here faults the arm rather than resting it.
    """
    period = 1.0 / rate_hz
    for remaining in range(seconds, 0, -1):
        print(f"  {label} in {remaining}...", flush=True)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if arm is not None and hold_at is not None:
                arm.command_positions(hold_at)
            time.sleep(period)


class Recorder:
    def __init__(self):
        self.timestamps = []
        self.commanded = []
        self.measured = []
        self.torques = []
        self.temperatures = []
        self.contact = []

    def add(self, target, state, in_contact):
        self.timestamps.append(time.time())
        self.commanded.append(list(target))
        self.measured.append(list(state.positions))
        self.torques.append(list(state.torques))
        self.temperatures.append([max(fb.temperature_mos, fb.temperature_rotor) for fb in state.feedback])
        self.contact.append(bool(in_contact))

    def arrays(self):
        return {
            "timestamps_unix": np.asarray(self.timestamps, dtype=float),
            "commanded_positions_rad": np.asarray(self.commanded, dtype=float),
            "measured_positions_rad": np.asarray(self.measured, dtype=float),
            "torques_nm": np.asarray(self.torques, dtype=float),
            "temperatures_c": np.asarray(self.temperatures, dtype=float),
            "deliberate_contact": np.asarray(self.contact, dtype=bool),
        }


def sweep_targets(home, seconds, rate_hz, amplitude_rad):
    """Lift the weight-bearing joints and lower them again.

    joint2 and joint3 rest against their lower limit of 0 rad at HOME, so a
    symmetric sweep spends half its cycle clamped and the arm barely moves.
    This raises them and returns, staying inside the limits the whole way and
    loading the joints against gravity where the torque signal lives.
    """
    ticks = max(2, int(seconds * rate_hz))
    for tick in range(ticks):
        phase = 2.0 * math.pi * tick / ticks
        lift = amplitude_rad * 0.5 * (1.0 - math.cos(phase))
        target = list(home)
        target[1] += lift
        target[2] += lift * 0.6
        target[3] -= lift * 0.4
        yield target


def verify_sweep_is_clear(poses, map_path, arm_xml_path) -> None:
    """Refuse to drive a calibration motion that the workcell map says collides."""
    from yam.environment import ArmSafetyChecker
    from yam.kinematics import YamKinematics
    from yam.voxel_map import VoxelMap

    checker = ArmSafetyChecker(YamKinematics(), VoxelMap.load(map_path), arm_xml_path, margin=0.0)
    clearances = [checker.clearance(pose) for pose in poses]
    worst = min(clearances)
    if worst <= 0.0:
        raise SystemExit(
            f"  the calibration sweep is not clear of the map (worst {worst * 1000:+.1f} mm).\n"
            f"  Reduce --amplitude-deg, or rebuild the map, before moving the arm."
        )
    print(f"  sweep verified against {map_path}: {len(poses)} poses, "
          f"min clearance {worst * 1000:+.1f} mm")


def run_phase(arm, targets, rate_hz, gain_scale, recorder, in_contact):
    period = 1.0 / rate_hz
    for target in targets:
        state = arm.command_positions(target, gain_scale=gain_scale)
        recorder.add(target, state, in_contact)
        time.sleep(period)


def residual_peaks(arrays, baseline_seconds, warmup_seconds):
    torques = arrays["torques_nm"]
    timestamps = arrays["timestamps_unix"]
    baseline = torques[0].copy()
    residuals = np.zeros_like(torques)
    elapsed = 0.0
    for index in range(1, len(timestamps)):
        period = timestamps[index] - timestamps[index - 1]
        elapsed += period
        residuals[index] = np.abs(torques[index] - baseline)
        alpha = min(1.0, period / baseline_seconds)
        baseline += alpha * (torques[index] - baseline)
        if elapsed < warmup_seconds:
            residuals[index] = 0.0
    return residuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-id", required=True, help="which physical arm this describes")
    parser.add_argument("--trace", default="guard_trace.npz")
    parser.add_argument("--output", default="guard_calibration.json")
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--gain-scale", type=float, default=0.5)
    parser.add_argument("--free-seconds", type=float, default=20.0)
    parser.add_argument("--contact-seconds", type=float, default=20.0)
    parser.add_argument("--amplitude-deg", type=float, default=20.0)
    parser.add_argument("--path", help="calibrate along a planned path instead of a sweep, so the "
                                       "torque baseline covers the gravity regime the guard will run in")
    parser.add_argument("--slowdown", type=float, default=3.0,
                        help="replay a --path this many times slower than planned")
    parser.add_argument("--map", default="workcell_map.npz")
    parser.add_argument("--arm-xml", default="../i2rt/i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml")
    args = parser.parse_args()

    amplitude = math.radians(args.amplitude_deg)
    recorder = Recorder()

    if args.path:
        planned = np.load(args.path)
        poses = [pose for pose in planned for _ in range(max(1, int(args.slowdown)))]
        print(f"  calibrating along {args.path}: {len(planned):,} poses "
              f"replayed {int(args.slowdown)}x slow ({len(poses) / args.rate:.0f}s per phase)")
    else:
        poses = list(sweep_targets(HOME, args.free_seconds, args.rate, amplitude))
    verify_sweep_is_clear(poses, args.map, args.arm_xml)

    print("The arm will make a small out-and-back motion around its resting pose.")
    print("Keep the workspace clear and keep a hand on the power.")
    countdown("starting free-motion phase", 5)

    with connected_arm(joints=ARM_JOINTS) as arm:
        arm.clear_errors()
        arm.enable()
        arm.move_to(HOME, duration=3.0, rate_hz=args.rate)

        print(f"  recording {len(poses) / args.rate:.0f}s of free motion -- do not touch the arm")
        arm.move_to(poses[0], duration=3.0, rate_hz=args.rate)
        run_phase(arm, poses, args.rate, args.gain_scale, recorder, in_contact=False)
        arm.move_to(poses[0], duration=3.0, rate_hz=args.rate)

        print()
        print(f"  NEXT: press firmly on the forearm several times over {args.contact_seconds:.0f}s.")
        print("  Push against the motion; let go between pushes.")
        countdown("contact phase begins", 8, arm=arm, hold_at=poses[0], rate_hz=args.rate)
        try:
            run_phase(arm, poses, args.rate, args.gain_scale, recorder, in_contact=True)
            arm.move_to(HOME, duration=3.0, rate_hz=args.rate)
        except Exception as error:
            np.savez(args.trace, **recorder.arrays())
            raise SystemExit(f"\n  fault during the contact phase: {error}\n"
                             f"  the {len(recorder.timestamps):,} samples recorded so far are saved "
                             f"in {args.trace}; re-run to collect a clean pair.")

    arrays = recorder.arrays()
    free = ~arrays["deliberate_contact"]
    contact = arrays["deliberate_contact"]

    residuals = residual_peaks(arrays, BASELINE_SECONDS, WARMUP_SECONDS)
    free_residual = residuals[free].max(axis=0)
    contact_residual = residuals[contact].max(axis=0)

    tracking_observed = np.abs(arrays["commanded_positions_rad"] - arrays["measured_positions_rad"]).max(axis=0)
    free_torque = np.abs(arrays["torques_nm"][free]).max()
    free_temperature = arrays["temperatures_c"][free].max()

    # A handful of transient spikes -- stiction breakaway, a command staircase --
    # sit far above the free-motion noise floor and are not what the guard must
    # tolerate continuously. Fitting to the raw peak lets one 70ms outlier set a
    # threshold no real contact ever reaches, so the floor is taken robustly and
    # the observed peak only widens it where contact still separates cleanly.
    free_floor = np.percentile(residuals[free], 99.0, axis=0)
    residual_limit = np.maximum(free_floor * 1.5, 0.05)
    room = contact_residual > residual_limit * 1.3
    residual_limit = np.where(room, np.minimum(residual_limit, contact_residual * 0.6), residual_limit)
    detected = contact_residual > residual_limit
    print()
    print("  per-joint residual (Nm):")
    for index in range(6):
        print(f"    joint{index + 1}  free_p99 {free_floor[index]:5.2f}  free_peak {free_residual[index]:5.2f}   "
              f"contact {contact_residual[index]:5.2f}   threshold {residual_limit[index]:5.2f}"
              f"   {'detects contact' if detected[index] else 'no contact seen'}")

    if not detected.any():
        raise SystemExit("\n  No joint saw a contact above its free-motion noise. Push harder and re-run;\n"
                         "  a threshold that detects none of the contacts is not a calibration.")

    np.savez(args.trace, **arrays)
    calibration = {
        "schema_version": 1,
        "hardware_validated": True,
        "robot_id": args.robot_id,
        "measured_at_unix": time.time(),
        "raw_trace_path": str(Path(args.trace).name),
        "raw_trace_sha256": file_sha256(args.trace),
        "max_tracking_error_rad": (tracking_observed * 1.25).tolist(),
        "max_torque_residual_nm": residual_limit.tolist(),
        "absolute_torque_nm": float(free_torque * 1.15),
        "max_temperature_c": float(free_temperature + 15.0),
        "baseline_seconds": BASELINE_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "rate_hz": args.rate,
        "gain_scale": args.gain_scale,
    }
    Path(args.output).write_text(json.dumps(calibration, indent=2))

    loaded = HardwareSafetyCalibration.load(args.output)
    print(f"\n  wrote {args.trace} ({len(arrays['timestamps_unix']):,} samples) and {args.output}")
    print(f"  validated against the trace: robot {loaded.robot_id}, "
          f"absolute torque {loaded.absolute_torque_nm:.2f}Nm")


if __name__ == "__main__":
    main()
