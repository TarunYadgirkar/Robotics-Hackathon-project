"""Enroll obstacles by touching them with the arm, with a live 3D viewer.

The arm is held limp at zero torque throughout -- it is a measuring instrument
here, not an actuator. Move the gripper onto a feature of the obstacle and press
Space (in the browser) or Enter (in this terminal) to record where the tip is.

  python scripts/enroll.py --object "monitor stand"
  python scripts/enroll.py --validate-fk        # touch ONE point repeatedly

The viewer opens at http://127.0.0.1:8420/.
"""

import argparse
import os
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager

import numpy as np

from yam.arm import ARM_JOINTS, connected_arm
from yam.enrollment import EnrollmentSession, touch_repeatability
from yam.kinematics import YamKinematics
from yam.mesh_export import export_arm_meshes
from yam.viz_server import VizServer

DEFAULT_OUTPUT = "enrollment.json"


@contextmanager
def open_arm(simulate: bool):
    """Yield a live arm, or None when simulating so the viewer can run with no hardware."""
    if simulate:
        yield None
        return
    with connected_arm(joints=ARM_JOINTS) as arm:
        yield arm

PROMPTS = [
    "Put the gripper tip on a corner of the obstacle, then capture.",
    "Now a corner diagonally opposite -- the further apart, the better the box.",
    "Touch it from a different side. The shell shows which directions you have.",
    "Reach over or under it if you can; height is the axis people forget.",
    "Fill the remaining shell patches, then finish.",
]


def read_terminal_lines(sink):
    for line in sys.stdin:
        sink(line.strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default="object_1", help="name of the first obstacle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--padding", type=float, default=0.02, help="metres of margin added around touched points")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="0.0.0.0",
                        help="0.0.0.0 lets the phone that does the LiDAR scan open the viewer; "
                             "use 127.0.0.1 to keep it on this machine only")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--validate-fk", action="store_true",
                        help="touch one fixed point repeatedly and report the spread")
    parser.add_argument("--simulate", action="store_true",
                        help="drive the viewer from a synthetic pose, with no arm connected")
    args = parser.parse_args()

    kinematics = YamKinematics()
    session = EnrollmentSession()
    session.begin_object(args.object, padding=args.padding)

    server = VizServer(port=args.port, host=args.host)
    server.static_payloads["meshes"] = export_arm_meshes(kinematics)
    url = server.start()
    addresses = server.urls()

    print(f"\n  Viewer:      {addresses['local']}")
    if addresses["lan"]:
        print(f"  From phone:  {addresses['lan']}   (same wifi; anyone on this network can reach it)")
    print("  The arm is limp -- support it. Space/Enter captures, U undoes, N next object, F finishes.")
    print("  Scanning with a phone? Keep this running: the arm's pose is logged throughout,")
    print("  so it can be subtracted from the sweep afterwards.\n")
    if not args.no_browser:
        webbrowser.open(addresses["local"])

    threading.Thread(target=read_terminal_lines, args=(server.commands.put,), daemon=True).start()

    message = ""
    message_kind = ""
    finished = False

    def simulated_pose(elapsed: float):
        return [
            0.5 * np.sin(elapsed * 0.35),
            1.1 + 0.32 * np.sin(elapsed * 0.24 + 1.0),
            0.8 + 0.35 * np.sin(elapsed * 0.19 + 2.0),
            0.3 * np.sin(elapsed * 0.4),
            0.4 * np.sin(elapsed * 0.31 + 0.5),
            0.6 * np.sin(elapsed * 0.27),
        ]

    try:
        with open_arm(args.simulate) as arm:
            if arm is not None:
                arm.enable()
            started = time.time()

            while not finished:
                if arm is None:
                    q = simulated_pose(time.time() - started)
                    state = None
                else:
                    state = arm.read_state()      # zero gains: reading never applies torque
                    q = list(state.positions)
                tip = kinematics.tip_position(q)
                obstacle = session.current

                command = server.next_command()
                if command in ("capture", ""):
                    if command == "capture":
                        session.capture(tip, q)
                        message, message_kind = f"captured point {len(obstacle.points)}", ""
                elif command == "undo":
                    message = "undid last point" if session.undo() else "nothing to undo"
                    message_kind = "" if session.current.points else "warn"
                elif command == "next":
                    session.begin_object(f"object_{len(session.objects) + 1}", padding=args.padding)
                    message, message_kind = f"started {session.current.name}", ""
                elif command == "scan_uploaded":
                    message, message_kind = f"scan received: {os.path.basename(server.uploads[-1])}", ""
                elif command in ("done", "finish", "quit"):
                    finished = True
                    continue

                session.log_pose(q)

                obstacle = session.current
                bounds = obstacle.bounds()
                transforms = kinematics.link_transforms(q)

                server.update({
                    "status": "enrolling",
                    "object_name": obstacle.name,
                    "prompt": PROMPTS[min(len(obstacle.points), len(PROMPTS) - 1)],
                    "joints": q,
                    "tip": tip.tolist(),
                    "link_transforms": {name: matrix.ravel().tolist() for name, matrix in transforms.items()},
                    "points": [p.position for p in obstacle.points],
                    "point_count": len(obstacle.points),
                    "patches": obstacle.patch_coverage().tolist(),
                    "progress": obstacle.progress,
                    "box_min": None if bounds is None else bounds[0].tolist(),
                    "box_max": None if bounds is None else bounds[1].tolist(),
                    "temperature": None if state is None else max(
                        max(f.temperature_mos, f.temperature_rotor) for f in state.feedback
                    ),
                    "pose_samples": len(session.pose_log),
                    "scans": [os.path.basename(path) for path in server.uploads],
                    "message": message,
                    "message_kind": message_kind,
                })
                message = ""
                # Keep the frame rate up even while waiting on a keypress: an enabled
                # motor with silent gaps latches error 0xD and needs clear_errors().
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n  interrupted")

    if args.validate_fk:
        first = session.objects[0]
        if len(first.points) >= 3:
            result = touch_repeatability(kinematics, [p.joint_angles for p in first.points])
            print("\n  FK validation -- same physical point, different arm poses:")
            print(f"    {result['count']} touches, mean spread {result['mean_error_mm']:.1f} mm, "
                  f"worst {result['max_error_mm']:.1f} mm")
            print("    That spread is the error budget every enrolled obstacle inherits.")
        else:
            print("\n  need at least 3 touches to judge repeatability")

    session.save(args.output)
    server.update({**server.snapshot(), "status": "finished"})

    print(f"\n  saved {args.output}  ({len(session.pose_log)} arm poses logged for scan subtraction)")
    for obstacle in session.objects:
        bounds = obstacle.bounds()
        if bounds is None:
            print(f"    {obstacle.name}: no points")
            continue
        size = (bounds[1] - bounds[0]) * 1000
        print(f"    {obstacle.name}: {len(obstacle.points)} points -> box {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm")

    time.sleep(0.5)
    server.stop()


if __name__ == "__main__":
    main()
