"""Hold the arm at a pose.

With no target, holds wherever the arm already is: position error starts at zero,
so commanded torque starts at zero and nothing lurches. Gains ramp in over the
first half second. Every joint is held -- leaving some limp while others stiffen
would let the limp ones drop.

  python scripts/hold.py                      # hold current pose, 10s, half gain
  python scripts/hold.py --duration 30
  python scripts/hold.py --gain-scale 1.0
  python scripts/hold.py --degrees 0 90 90 0 0 0

Ctrl-C disables every motor. The arm goes limp when it stops, so support it.
"""

import argparse
import math

from yam.arm import ARM_JOINTS, SafetyLimits, connected_arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to hold")
    parser.add_argument("--gain-scale", type=float, default=0.5, help="fraction of the SDK's stiffness (0-1)")
    parser.add_argument("--degrees", type=float, nargs=6, default=None, help="absolute target pose for the 6 arm joints")
    parser.add_argument("--nudge", type=float, nargs=2, default=None, metavar=("JOINT", "DEGREES"),
                        help="move one joint (1-6) by a relative amount, leaving the rest where they are")
    parser.add_argument("--move-duration", type=float, default=3.0, help="seconds to interpolate to --degrees")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safety = SafetyLimits(gain_scale=args.gain_scale)

    with connected_arm(joints=ARM_JOINTS, safety=safety) as arm:
        arm.enable()
        start = arm.read_state()
        print("current pose:")
        print(start.describe())

        if args.nudge is not None:
            index = int(args.nudge[0]) - 1
            if not 0 <= index < len(ARM_JOINTS):
                raise SystemExit(f"--nudge joint must be 1-{len(ARM_JOINTS)}")
            targets = list(start.positions)
            targets[index] += math.radians(args.nudge[1])
            clamped = [j.clamp_position(t) for j, t in zip(ARM_JOINTS, targets)]
            moved = math.degrees(clamped[index] - start.positions[index])
            print(f"\nnudging {ARM_JOINTS[index].name} by {moved:+.1f}deg over {args.move_duration:.0f}s...")
            arm.move_to(clamped, duration=args.move_duration)
            targets = clamped
        elif args.degrees is None:
            targets = start.positions
            print(f"\nholding this pose for {args.duration:.0f}s at {args.gain_scale:.0%} gain...")
        else:
            targets = [math.radians(d) for d in args.degrees]
            clamped = [j.clamp_position(t) for j, t in zip(ARM_JOINTS, targets)]
            for joint, wanted, actual in zip(ARM_JOINTS, targets, clamped):
                if abs(wanted - actual) > 1e-6:
                    print(f"  {joint.name}: {math.degrees(wanted):.1f}deg clamped to {math.degrees(actual):.1f}deg")
            print(f"\nmoving over {args.move_duration:.0f}s, then holding {args.duration:.0f}s...")
            arm.move_to(clamped, duration=args.move_duration)
            targets = clamped

        final = arm.hold(targets, duration=args.duration)

        print("\nfinal state:")
        print(final.describe())

        error = [math.degrees(t - p) for t, p in zip(targets, final.positions)]
        print("\nhold error (deg): " + "  ".join(f"{j.name}={e:+.2f}" for j, e in zip(ARM_JOINTS, error)))
        print("\nreleasing -- the arm goes limp now.")


if __name__ == "__main__":
    main()
