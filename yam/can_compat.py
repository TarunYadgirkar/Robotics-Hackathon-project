"""macOS compatibility for CANable (gs_usb) adapters.

The `gs_usb` package unconditionally calls `detach_kernel_driver` on any
non-Windows platform. macOS has no gs_usb kernel driver to detach, and libusb
raises `Access denied` there rather than treating it as a no-op, so opening the
bus fails intermittently. This module makes that one call a no-op on Darwin and
leaves every other platform untouched.
"""

import platform

import usb.core
from gs_usb.constants import (
    GS_CAN_MODE_HW_TIMESTAMP,
    GS_CAN_MODE_LISTEN_ONLY,
    GS_CAN_MODE_LOOP_BACK,
    GS_CAN_MODE_ONE_SHOT,
)
from gs_usb.gs_usb import GS_CAN_MODE_START, GsUsb
from gs_usb.gs_usb_structures import DeviceMode

_GS_USB_BREQ_MODE = 2

# Mirrors the mask in GsUsb.start: the mode bits this driver knows how to honour.
# Dropping HW_TIMESTAMP here would desync frame sizes between start() and read().
_DRIVER_SUPPORTED_MODES = GS_CAN_MODE_LISTEN_ONLY | GS_CAN_MODE_LOOP_BACK | GS_CAN_MODE_ONE_SHOT | GS_CAN_MODE_HW_TIMESTAMP

IS_MACOS = platform.system().lower() == "darwin"


def apply() -> bool:
    """Patch gs_usb for macOS. Returns True if a patch was applied."""
    if not IS_MACOS or getattr(GsUsb.start, "_macos_patched", False):
        return False

    default_flags = GsUsb.start.__defaults__[0]

    def start(self, flags: int = default_flags) -> None:
        self.gs_usb.reset()
        flags &= self.device_capability.feature & _DRIVER_SUPPORTED_MODES
        self.device_flags = flags
        self.gs_usb.ctrl_transfer(0x41, _GS_USB_BREQ_MODE, 0, 0, DeviceMode(GS_CAN_MODE_START, flags).pack())

    start._macos_patched = True
    GsUsb.start = start
    return True


def reset_adapter() -> None:
    """Force the adapter back to a known state after an unclean shutdown."""
    for device in GsUsb.scan():
        try:
            device.stop()
        except usb.core.USBError:
            pass
        try:
            device.gs_usb.reset()
        except usb.core.USBError:
            pass


def find_adapter() -> GsUsb:
    devices = GsUsb.scan()
    if not devices:
        raise RuntimeError(
            "No gs_usb/CANable adapter found. Check the USB cable; confirm it enumerates with:\n"
            "  ioreg -p IOUSB -w0 -l | grep -i canable"
        )
    return devices[0]


apply()
