"""Check the CAN link to the arm, from USB adapter down to each motor.

Read-only: it enables each motor to read its status, then immediately disables it.
No torque is ever commanded.

Usage: python scripts/diagnose.py
"""

import argparse
import time

import can
from gs_usb.gs_usb import GsUsb

from yam import can_compat  # noqa: F401
from yam.arm import ARM_JOINTS, CAN_BITRATE, GRIPPER_JOINT
from yam.dm_motor import CLEAR_ERROR, DISABLE, ENABLE, FEEDBACK_ID_OFFSET, decode_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clear", action="store_true",
                        help="clear latched motor errors before reporting")
    args = parser.parse_args()

    adapters = GsUsb.scan()
    print(f"1. USB adapter: {'found' if adapters else 'NOT FOUND'}")
    if not adapters:
        print("   -> Check the USB cable to the CANable.")
        return
    print(f"   CAN clock {adapters[0].device_capability.fclk_can / 1e6:.0f} MHz")

    bus = can.interface.Bus(interface="gs_usb", channel=0, bitrate=CAN_BITRATE)
    print(f"2. CAN bus opened at {CAN_BITRATE // 1000} kbit/s")

    joints = ARM_JOINTS + [GRIPPER_JOINT]

    if args.clear:
        for joint in joints:
            for _ in range(3):
                bus.send(can.Message(arbitration_id=joint.motor_id,
                                     data=bytearray(CLEAR_ERROR), is_extended_id=False))
                time.sleep(0.002)
        while bus.recv(timeout=0.01):
            pass
        print("   cleared latched errors")

    acknowledged = 0
    responded = []

    print("3. Motors:")
    for joint in joints:
        message = can.Message(arbitration_id=joint.motor_id, data=bytearray(ENABLE), is_extended_id=False)
        bus.send(message)

        echoed = False
        feedback = None
        deadline = time.time() + 0.15
        while time.time() < deadline:
            reply = bus.recv(timeout=0.05)
            if reply is None:
                continue
            if not reply.is_rx:
                echoed = True
            elif reply.arbitration_id == joint.motor_id + FEEDBACK_ID_OFFSET:
                feedback = decode_feedback(reply.arbitration_id, reply.data, joint.spec)

        acknowledged += echoed
        if feedback:
            responded.append(joint.name)
            print(
                f"   {joint.name:>8} (0x{joint.motor_id:02X})  ONLINE   "
                f"pos={feedback.position:+7.4f} rad  T={feedback.temperature_mos:.0f}/"
                f"{feedback.temperature_rotor:.0f}C  [{feedback.error_message}]"
            )
        else:
            print(f"   {joint.name:>8} (0x{joint.motor_id:02X})  no reply  (frame acked on bus: {echoed})")

        bus.send(can.Message(arbitration_id=joint.motor_id, data=bytearray(DISABLE), is_extended_id=False))
        time.sleep(0.01)
        while bus.recv(timeout=0.005):
            pass

    bus.shutdown()

    print(f"\n{len(responded)}/{len(joints)} motors online.")
    if not acknowledged:
        print(
            "No frame was acknowledged by anything on the bus. A CAN frame is only acked if at least\n"
            "one other powered node is listening, so this points at the arm side, not the laptop:\n"
            "  - is the arm's power supply on, and any e-stop released?\n"
            "  - is the CAN cable seated at both the adapter and the arm?\n"
            "  - does the adapter need its 120 ohm termination jumper set?"
        )


if __name__ == "__main__":
    main()
