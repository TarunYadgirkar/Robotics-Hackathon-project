"""Staged bring-up of the real YAM arm through the arm_io contract.

Stages, smallest blast radius first:

  --read-only   adapter scan, limits-mirror check, passive joint read, preflight.
                Touches the bus but never enables a motor and never commands a
                gain or a torque. Safe to run at any time.
  (default)     the above, then enable + keep-alive, then a single small gripper
                pulse, then the three gestures, then the gripper-cycle task_demo.
  --gated       same, but every motion waits for the operator to press ENTER.

Gating is off by default because the operator asked for it: the demo is the
robot trying something, not a human driving it with keys. Ctrl-C is still the
e-stop and still freezes-and-holds rather than homing, the velocity caps still
bind, and the hardware rule (arm within a few degrees, jaws gentle) is checked
against every prepared setpoint stream before a frame goes out.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arm import arm_io, facts, hw_backend, model, motion, safety  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def read_only_stage() -> bool:
    """Everything that can be learned without energising a motor."""
    rule("STAGE 0 — READ-ONLY (no motor is enabled, no torque is commanded)")

    print(f"FACTS.md HARDWARE_PRESENT : {facts.hardware_flag_raw()}")
    print(f"FACTS.md device           : {facts.device_path()}")
    print(f"arm_io backend            : {arm_io.backend_name()} (simulated={arm_io.is_simulated()})")
    if arm_io.is_simulated():
        print("FACTS says no hardware; nothing to bring up. Stopping.")
        return False

    from gs_usb.gs_usb import GsUsb

    adapters = GsUsb.scan()
    print(f"gs_usb adapters found     : {len(adapters)}")
    if not adapters:
        print("No adapter. Check the USB cable to the CANable. Stopping.")
        return False
    print(f"CAN clock                 : {adapters[0].device_capability.fclk_can / 1e6:.0f} MHz")

    print("\n-- limits mirror (arm/model.py vs yam.arm) --")
    hw_backend.verify_against_yam()
    print("verify_against_yam()          PASS")
    hw_backend.verify_cap_against_driver(model.YAM_TICK_HZ)
    print(f"verify_cap_against_driver()   PASS  (cap "
          f"{model.velocity_cap_deg_s()['joint1']:.1f} deg/s at {model.YAM_TICK_HZ:.0f} Hz)")

    print("\n-- passive joint read (zero gains, zero torque, motors NOT enabled) --")
    readings = arm_io.probe_passive()
    import math

    for joint, fb in readings:
        extra = ""
        if joint.name == "gripper":
            extra = f"  ({model.gripper_rad_to_percent(fb.position):.1f}% open)"
        print(f"  {joint.name:>8} (0x{joint.motor_id:02X})  pos={fb.position:+8.4f} rad "
              f"({math.degrees(fb.position):+8.2f} deg)  T={fb.temperature_mos:.0f}/"
              f"{fb.temperature_rotor:.0f}C  [{fb.error_message}]{extra}")
    gripper_pct = model.gripper_rad_to_percent(readings[-1][1].position)
    print(f"\ngripper resting opening   : {gripper_pct:.1f}%")
    if gripper_pct < 40.0:
        print("  NOTE: the hardware gestures CLOSE the jaws by up to 35 points from rest. With "
              "them this nearly shut the motion resolves below zero and is refused — part them "
              "by hand before enabling.")
    stale = [j.name for j, fb in readings if fb.error_code == 0xD]
    if stale:
        print(f"\n0xD comms-lost latch on   : {', '.join(stale)}")
        print("  Benign: these motors sat enabled without a command stream. connect() clears it "
              "once via recover_stale_motors() and then keeps a 100 Hz stream up. If a motor is "
              "still unhealthy after that single pass, bring-up stops rather than clearing again.")
    hot = [j.name for j, fb in readings if max(fb.temperature_mos, fb.temperature_rotor) > 60]
    if hot:
        print(f"  WARNING: running warm: {', '.join(hot)}")

    print("\n-- preflight checklist --")
    for i, item in enumerate(hw_backend.preflight_checklist(), 1):
        print(f"  {i}. {item}")
    return True


def gate(description: str, gated: bool) -> bool:
    if not gated:
        print(f"\n>>> {description}")
        return True
    try:
        answer = input(f"\n>>> {description}\n    ENTER to run, 's' to skip, 'q' to quit: ").strip().lower()
    except EOFError:
        return False
    if answer == "q":
        raise SystemExit("operator quit")
    return answer != "s"


def motion_stage(gated: bool, speed_gesture: float, speed_task: float) -> None:
    rule("STAGE 1 — ENABLE + KEEP-ALIVE (motors energise here)")
    print("Ctrl-C at any point freezes and HOLDS. It does not home.")
    if not gate("enable motors and start the 100 Hz keep-alive stream", gated):
        return
    from yam.arm import MotorCommunicationError, MotorFaultError

    try:
        arm_io.connect()
    except (MotorFaultError, MotorCommunicationError) as exc:
        print(f"\nSTOPPED at enable: {type(exc).__name__}: {exc}")
        print("That is after Boris's single clear-and-retry pass. Not clearing into it again — "
              "check the arm's power and e-stop, then run scripts/diagnose.py --clear by hand.")
        return
    pose = arm_io.current_pose()
    print("pose after enable: " + ", ".join(
        f"{n}={v:.2f}" for n, v in zip(model.JOINT_NAMES, pose)))

    try:
        rule("STAGE 2 — home() (on hardware: settle and hold, no sweep)")
        if gate("arm_io.home()", gated):
            arm_io.home()

        rule("STAGE 3 — gestures, gripper-first, at %d%% speed" % int(speed_gesture * 100))
        for name in arm_io.GESTURE_NAMES:
            if gate(f"arm_io.gesture({name!r}, speed={speed_gesture})", gated):
                started = time.perf_counter()
                arm_io.gesture(name, speed=speed_gesture)
                print(f"    done in {time.perf_counter() - started:.1f}s")

        rule("STAGE 4 — task_demo gripper cycle at %d%% speed" % int(speed_task * 100))
        if gate(f"arm_io.replay(TASK_DEMO_PATH, speed={speed_task})", gated):
            started = time.perf_counter()
            arm_io.replay(arm_io.TASK_DEMO_PATH, speed=speed_task)
            print(f"    done in {time.perf_counter() - started:.1f}s")

    except safety.MotionAborted:
        print("\nFROZEN by interrupt. The arm is holding, motors still enabled.")
        print("arm_io.recover_home() resumes the hold after you confirm; arm_io.shutdown() "
              "disables and releases the bus.")
        return
    except hw_backend.ArmUnhealthy as exc:
        print(f"\nSTOPPED: {exc}")
        return
    finally:
        pass

    rule("DONE — disabling motors and closing the bus")
    arm_io.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--read-only", action="store_true", help="stage 0 only; never enables a motor")
    parser.add_argument("--gated", action="store_true", help="wait for ENTER before every motion")
    parser.add_argument("--gesture-speed", type=float, default=0.5)
    parser.add_argument("--task-speed", type=float, default=0.25)
    args = parser.parse_args()

    if not read_only_stage():
        return 1
    if args.read_only:
        print("\nread-only stage complete; no motor was enabled.")
        return 0

    try:
        motion_stage(args.gated, args.gesture_speed, args.task_speed)
    except KeyboardInterrupt:
        print("\ninterrupted outside a motion; disabling.")
        arm_io.shutdown()
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
