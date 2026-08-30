"""Enroll obstacles by touching them with the arm, with a live 3D viewer.

The arm is held limp at zero torque throughout -- it is a measuring instrument
here, not an actuator. Move the gripper onto a feature of the obstacle and press
Space (in the browser) or Enter (in this terminal) to record where the tip is.

  python scripts/enroll.py --object "monitor stand"
  python scripts/enroll.py --validate-fk        # touch ONE point repeatedly

The viewer opens at http://127.0.0.1:8420/.
"""

import argparse
import glob
import os
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager

import can
import numpy as np

from yam.arm import ARM_JOINTS, MotorCommunicationError, MotorFaultError, connected_arm
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
    parser.add_argument("--fresh-scan", action="store_true",
                        help="ignore any scan already on disk instead of adopting the newest")
    parser.add_argument("--linger", type=float, default=180.0,
                        help="seconds to keep the viewer up after finishing, so the phone sees the result")
    parser.add_argument("--tunnel", action="store_true",
                        help="expose the viewer through a Cloudflare quick tunnel, for networks "
                             "like eduroam that block phone-to-laptop traffic")
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
    server.kinematics = kinematics

    # Adopt the most recent scan on disk. A scan survives a server restart as a
    # file, but used to live only in the previous process's memory, so the phone
    # showed nothing and there was no way to erase from it or align it.
    existing = sorted(glob.glob("phone_scan_*.ply") + glob.glob("*.ply"), key=os.path.getmtime)
    if existing and not args.fresh_scan:
        try:
            from yam.lidar import load_point_cloud

            points = load_point_cloud(existing[-1])
            step = max(1, len(points) // 40000)
            server.scan_points = points[::step]
            server.uploads.append(os.path.abspath(existing[-1]))
            print(f"  adopted existing scan {existing[-1]} ({len(points):,} points)")
        except Exception as error:
            print(f"  could not read {existing[-1]}: {error}")
    url = server.start()
    addresses = server.urls()

    print(f"\n  Viewer:      {addresses['local']}")
    if args.tunnel:
        # In a thread: bringing the tunnel up takes ~20s, and motors left without
        # a command stream for that long latch a comms timeout before enrollment
        # even starts.
        def announce_tunnel():
            public = server.start_tunnel()
            if public:
                print(f"\n  From phone:  {public}/?k={server.token}")
                print("               Public URL -- the API is token-guarded, so use the whole link.\n", flush=True)
            else:
                print("\n  Tunnel failed to start (is cloudflared installed?).\n", flush=True)

        print("  Opening a public tunnel in the background...", flush=True)
        threading.Thread(target=announce_tunnel, daemon=True).start()
    elif addresses["lan"]:
        print(f"  From phone:  {addresses['lan']}   (same wifi; blocked on eduroam -- use --tunnel)")
    print("  The arm is limp -- support it. Space/Enter captures, U undoes, N next object, F finishes.")
    print("  Scanning with a phone? Keep this running: the arm's pose is logged throughout,")
    print("  so it can be subtracted from the sweep afterwards.\n")
    if not args.no_browser:
        webbrowser.open(addresses["local"])

    threading.Thread(target=read_terminal_lines, args=(server.commands.put,), daemon=True).start()

    message = ""
    message_kind = ""
    finished = False
    consecutive_faults = 0

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
                # Motors latch a comms-timeout whenever something left them
                # enabled without a command stream, and read_state() refuses to
                # run against a latched motor. Clear that one code, report the
                # rest, and carry on.
                stale = arm.recover_stale_motors()
                if stale:
                    print(f"  cleared comms-timeout latch on: {', '.join(stale)}")
                arm.enable()
            started = time.time()

            while not finished:
                if arm is None:
                    q = simulated_pose(time.time() - started)
                    state = None
                else:
                    try:
                        state = arm.read_state()  # zero gains: reading never applies torque
                    except (MotorCommunicationError, MotorFaultError, can.CanError, OSError) as fault:
                        # One dropped frame or one latched motor should not end a
                        # session someone is halfway through. Recover in place and
                        # say so, rather than losing the captured points.
                        consecutive_faults += 1
                        message, message_kind = f"recovering: {fault}", "warn"
                        print(f"  {fault} -- recovering", flush=True)
                        if consecutive_faults > 20:
                            raise
                        try:
                            if isinstance(fault, (can.CanError, OSError)):
                                # The adapter itself went away; motor-level
                                # recovery cannot work until the bus is back.
                                arm.reconnect()
                            arm.recover_stale_motors()
                            arm.enable()
                        except Exception:
                            pass
                        time.sleep(0.05)
                        continue
                    consecutive_faults = 0
                    q = list(state.positions)
                tip = kinematics.probe_position(q)   # the jaws, not the wrist frame
                obstacle = session.current

                command = server.next_command()
                if command in ("capture", "reference", ""):
                    if command == "capture":
                        session.capture(tip, q)
                        message, message_kind = f"captured point {len(obstacle.positions())}", ""
                    elif command == "reference":
                        session.capture(tip, q, label=EnrollmentSession.REFERENCE_LABEL)
                        message = f"reference {len(session.reference_points())} set -- now tap it in the scan"
                        message_kind = ""
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
                    "points": obstacle.positions().tolist(),
                    "point_count": len(obstacle.positions()),
                    "references": [p.position for p in session.reference_points()],
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

    summary = []
    print(f"\n  saved {args.output}  ({len(session.pose_log)} arm poses logged for scan subtraction)")
    for obstacle in session.objects:
        bounds = obstacle.bounds()
        if bounds is None:
            print(f"    {obstacle.name}: no points")
            summary.append({"name": obstacle.name, "points": 0})
            continue
        size = (bounds[1] - bounds[0]) * 1000
        print(f"    {obstacle.name}: {len(obstacle.positions())} points -> "
              f"box {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm")
        summary.append({
            "name": obstacle.name,
            "points": len(obstacle.positions()),
            "box_mm": [round(float(v)) for v in size],
        })

    # Keep serving after the arm is released. Finish used to stop the server
    # immediately, so the phone's next poll failed and the operator was left
    # guessing whether anything had been saved.
    server.update({
        "status": "finished",
        "saved_to": os.path.abspath(args.output),
        "objects": summary,
        "pose_samples": len(session.pose_log),
        "scans": [os.path.basename(path) for path in server.uploads],
        "reference_count": len(session.reference_points()),
    })

    print(f"\n  Motors released. The viewer stays up for {args.linger:.0f}s so the phone can")
    print("  show the result -- Ctrl-C to stop it sooner.")
    try:
        time.sleep(args.linger)
    except KeyboardInterrupt:
        pass
    server.stop()


if __name__ == "__main__":
    main()
