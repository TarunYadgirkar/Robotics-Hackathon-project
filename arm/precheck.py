"""60-second pre-stage ritual. Run this right before walking on.

Answers one question — "will the arm move when the demo asks it to?" — and
answers it the only way that counts, by actually moving the arm. Four stages,
each printing PASS or FAIL, and a single verdict line at the end.

  .venv/bin/python arm/precheck.py

  --no-motion   stages 1-3 only (USB, mirror, passive read); nothing is enabled.

A FAIL is not the end of the demo: arm_io raises one exception, ArmUnavailable,
and the demo layer continues without motion. This script tells you that BEFORE
you are on stage rather than during BEAT 1.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arm import arm_io, hw_backend, model  # noqa: E402

GREEN, RED, YEL, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def line(ok: bool, title: str, detail: str = "") -> bool:
    tag = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
    print(f"  [{tag}] {title}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-motion", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    print(f"\n{'=' * 64}\n  ARM PRE-STAGE CHECK\n{'=' * 64}")
    results = []

    # 1 — USB enumeration
    print("\n1. USB / adapter")
    try:
        from gs_usb.gs_usb import GsUsb

        adapters = GsUsb.scan()
        results.append(line(bool(adapters), "CANable enumerated",
                            f"{len(adapters)} adapter(s)" if adapters else
                            "none found — reseat the USB cable, then re-run"))
    except Exception as exc:
        results.append(line(False, "CANable enumerated", str(exc)))

    # 2 — the limits mirror still matches the driver
    print("\n2. Configuration")
    if arm_io.is_simulated():
        results.append(line(False, "hardware backend selected",
                            "FACTS says no hardware, or ARM_FORCE_SIM is set"))
    else:
        try:
            hw_backend.verify_against_yam()
            hw_backend.verify_cap_against_driver(model.YAM_TICK_HZ)
            results.append(line(True, "limits mirror matches yam.arm",
                                f"cap {model.velocity_cap_deg_s()['joint1']:.0f} deg/s, "
                                f"slow ceiling {model.HW_SLOW_SPEED_DEG_S:.0f} deg/s, "
                                f"joint1 locked at {model.HW_JOINT1_LOCKED_DEG:.0f} deg"))
        except Exception as exc:
            results.append(line(False, "limits mirror matches yam.arm", str(exc)))

    # 3 — passive read, no motors enabled
    print("\n3. Arm state (read-only, nothing enabled)")
    rest = None
    try:
        readings = arm_io.probe_passive()
        online = len(readings)
        rest_gripper = model.gripper_rad_to_percent(readings[-1][1].position)
        hot = [j.name for j, fb in readings if max(fb.temperature_mos, fb.temperature_rotor) > 60]
        results.append(line(online == 7, "all 7 motors answer", f"{online}/7"))
        results.append(line(not hot, "temperatures under 60C",
                            f"warm: {', '.join(hot)}" if hot else "all cool"))
        results.append(line(rest_gripper >= 30.0, "jaws have room to close",
                            f"resting {rest_gripper:.0f}% open"
                            + ("" if rest_gripper >= 30 else " — part them by hand")))
        rest = True
    except Exception as exc:
        results.append(line(False, "passive read", f"{type(exc).__name__}: {exc}"))

    # 4 — one real gesture
    if args.no_motion:
        print(f"\n{YEL}4. Motion skipped (--no-motion){OFF}")
    elif rest and all(results):
        print("\n4. Live motion — one 'attention' gesture (~12.5s). STAND CLEAR.")
        for i in (3, 2, 1):
            print(f"   starting in {i}...", flush=True)
            time.sleep(1.0)
        try:
            arm_io.connect()
            arm_io.gesture("attention")
            arm = arm_io._backend._arm
            results.append(line(arm.failures == 0, "gesture completed",
                                f"resyncs={arm.resyncs} failures={arm.failures}"))
        except hw_backend.ArmUnavailable as exc:
            results.append(line(False, "gesture completed", str(exc)[:120]))
        except Exception as exc:
            results.append(line(False, "gesture completed", f"{type(exc).__name__}: {exc}"))
        finally:
            arm_io.shutdown()
    else:
        print(f"\n{YEL}4. Motion skipped — an earlier stage failed{OFF}")

    ok = all(results)
    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 64}")
    if ok:
        print(f"  {GREEN}READY{OFF} — the arm moves. ({elapsed:.0f}s)")
    else:
        print(f"  {RED}NOT READY{OFF} — the arm will not move. ({elapsed:.0f}s)")
        print("  The demo still runs: arm_io raises ArmUnavailable and run_demo continues")
        print("  without motion. Fix if there is time; do not debug on stage.")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
