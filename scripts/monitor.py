"""Print live joint feedback. Commands zero gain and zero torque, so the arm stays limp.

  python scripts/monitor.py            # 10 seconds
  python scripts/monitor.py 30
"""

import argparse
import math
import time

from yam.arm import ARM_JOINTS, GRIPPER_JOINT, connected_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("duration", type=float, nargs="?", default=10.0, help="seconds to read for")
    parser.add_argument("--interval", type=float, default=0.25, help="seconds between samples")
    args = parser.parse_args()

    duration = args.duration
    joints = ARM_JOINTS + [GRIPPER_JOINT]

    with connected_arm(joints=joints) as arm:
        arm.enable()
        print(f"{len(joints)} motors online. Reading for {duration:.0f}s -- the arm is limp, move it by hand.\n")

        header = "  ".join(f"{j.name:>9}" for j in joints)
        print(f"{'time':>6}  {header}")

        deadline = time.time() + duration
        start = time.time()
        while time.time() < deadline:
            state = arm.read_state()
            degrees = "  ".join(f"{math.degrees(p):>9.2f}" for p in state.positions)
            print(f"{time.time() - start:>6.1f}  {degrees}", flush=True)
            time.sleep(args.interval)

        print("\nfinal state:")
        print(arm.read_state().describe())


if __name__ == "__main__":
    main()
