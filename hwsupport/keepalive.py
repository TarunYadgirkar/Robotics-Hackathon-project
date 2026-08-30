"""Keep the YAM arm's motors talking between demo beats.

yam/arm.py records the failure this exists for, measured on this arm: "An
enabled motor that stops receiving commands latches error 0xD ('communication
lost'). Measured: 20Hz polling with 0.3-0.4s silent gaps latched the gripper;
100Hz with zero-torque frames through the gaps did not ... The error survives
disable/enable and needs clear_errors()." Demo beats are separated by up to a
minute of talking, which is a silent gap two orders of magnitude longer than the
one that latched the gripper.

The hold mirrors scripts/hold.py and YamArm.hold(): target the pose the arm is
already resting in, so position error -- and therefore commanded torque --
starts at zero, and ramp gains in over the first half second so nothing lurches
on the first tick. Between beats the arm should not move; it should only keep
answering.

Three things here are load-bearing:

1. **One writer at a time.** command_positions() is a six-frame exchange and
   yam.arm._exchange resynchronises on the shared receive queue, so two threads
   commanding at once desynchronises frames and hands somebody a joint angle
   from a different moment. pause() does not just set a flag: it waits for the
   in-flight tick to finish, and paused() holds the bus lock for the caller.

2. **One recovery pass, ever.** 0xD is a timeout and clears on request; every
   other code (over-temperature, overcurrent, overload) reports a physical
   condition. So a comms error -- or a fault whose message is exactly the 0xD
   latch -- buys a single recover_stale_motors()/clear_errors()/enable() pass.
   After that, or for any other fault, the error is surfaced and the thread
   stops. Retrying into a faulted bus is what this must never do.

3. **Stopping does not disable.** The keepalive does not own the arm's
   lifecycle. stop() stops streaming; disable() stays with whoever connected.
   Callers that stop the keepalive and then go quiet get the 0xD latch back.
"""

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from yam.arm import MotorCommunicationError, MotorFaultError
from yam.dm_motor import COMMUNICATION_LOST, ERROR_MESSAGES

#: 100Hz is the rate yam/arm.py measured as latch-free; hw_backend streams at the
#: same rate. Lower rates are what latched the gripper, so this is a floor, not a
#: preference.
DEFAULT_RATE_HZ = 100.0

#: Matches YamArm.hold()'s ramp: gains reach full scale half a second in.
RAMP_SECONDS = 0.5

#: The exact text a 0xD latch surfaces as through MotorFaultError. Taken from the
#: driver's own table so it cannot drift from what _check_faults raises.
COMMS_LOST_TEXT = ERROR_MESSAGES[COMMUNICATION_LOST]

MAX_EVENTS = 400


class KeepaliveFaulted(RuntimeError):
    """The keepalive stopped because the arm reported something it will not retry into."""


@dataclass(frozen=True)
class Event:
    t: float
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"+{self.t:7.3f}s  {self.kind:<10} {self.detail}"


class Keepalive:
    """Hold the arm's current pose on a background thread between scripted motions.

    `arm` is anything with yam.arm.YamArm's surface: .safety, .read_state(),
    .command_positions(targets, gain_scale=...), .recover_stale_motors(),
    .clear_errors(), .enable(). It must already be connected and enabled.
    """

    def __init__(
        self,
        arm,
        rate_hz: float = DEFAULT_RATE_HZ,
        on_fault: Optional[Callable[[BaseException], None]] = None,
        name: str = "hwsupport-keepalive",
    ):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.arm = arm
        self.rate_hz = rate_hz
        self.name = name
        self.ticks = 0

        self.lock = threading.RLock()
        self._on_fault = on_fault
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._events: deque = deque(maxlen=MAX_EVENTS)
        self._started_at = time.monotonic()
        self._fault: Optional[BaseException] = None
        self._recovery_used = False

        self._target: Optional[List[float]] = None
        self._retarget = True
        self._ramp_tick = 0
        self._ramp_ticks = max(int(RAMP_SECONDS * rate_hz), 1)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "Keepalive":
        if self._started:
            raise RuntimeError("keepalive already started; make a new one after stop()")
        self.raise_if_faulted()
        self._started = True
        self._started_at = time.monotonic()
        self._log("start", f"{self.rate_hz:.0f} Hz, gain ramp over {RAMP_SECONDS:.1f}s")
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop streaming. Does NOT disable the motors -- that is the caller's call."""
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        self._thread = None
        self._log("stop", f"{self.ticks} ticks" + (" (thread did not join!)" if alive else ""))

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._pause_evt.is_set()

    # -- handoff -----------------------------------------------------------

    def pause(self) -> None:
        """Stop commanding and wait for the in-flight tick to land.

        Call this before any scripted motion. Returning means the keepalive holds
        no frame in flight, so the caller now owns the bus.
        """
        self.raise_if_faulted()
        if self._pause_evt.is_set():
            return
        self._pause_evt.set()
        with self.lock:  # blocks until the tick in progress, if any, has finished
            pass
        self._log("pause", "bus released to the caller")

    def resume(self) -> None:
        """Resume holding, re-reading the pose the motion left the arm in."""
        if not self._pause_evt.is_set():
            return
        self._retarget = True
        self._pause_evt.clear()
        self._log("resume", "re-reading pose, gains ramp again")

    @contextmanager
    def paused(self):
        """Own the bus for the duration of a scripted motion.

        Holds the keepalive's lock as well as pausing it, so a caller that shares
        the lock cannot interleave with a tick even if the pause flag races.
        """
        self.pause()
        self.lock.acquire()
        try:
            yield self
        finally:
            self.lock.release()
            self.resume()

    # -- faults ------------------------------------------------------------

    @property
    def fault(self) -> Optional[BaseException]:
        return self._fault

    def raise_if_faulted(self) -> None:
        if self._fault is None:
            return
        raise KeepaliveFaulted(
            f"keepalive stopped on {type(self._fault).__name__}: {self._fault}. Not retrying into "
            "it -- power-cycle or run scripts/diagnose.py --clear and re-check before any motion."
        ) from self._fault

    # -- event log ---------------------------------------------------------

    def events(self) -> Tuple[Event, ...]:
        return tuple(self._events)

    def report(self) -> str:
        """One line per event, for the demo to print between beats."""
        return "\n".join(str(event) for event in self._events)

    def _log(self, kind: str, detail: str = "") -> None:
        self._events.append(Event(time.monotonic() - self._started_at, kind, detail))

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop_evt.is_set():
            started = time.monotonic()
            if not self._pause_evt.is_set():
                with self.lock:
                    # Re-checked under the lock: pause() may have won the race.
                    if not self._pause_evt.is_set() and not self._stop_evt.is_set():
                        if not self._tick():
                            return
            delay = period - (time.monotonic() - started)
            if delay > 0:
                self._stop_evt.wait(delay)

    def _tick(self) -> bool:
        """One hold tick. False means stop: the arm reported something unrecoverable."""
        try:
            if self._retarget or self._target is None:
                self._target = list(self.arm.read_state().positions)
                self._retarget = False
                self._ramp_tick = 0
            self._ramp_tick += 1
            scale = self.arm.safety.gain_scale * min(1.0, self._ramp_tick / self._ramp_ticks)
            self.arm.command_positions(self._target, gain_scale=scale)
            self.ticks += 1
            return True
        except (MotorCommunicationError, MotorFaultError) as exc:
            return self._on_error(exc)

    def _is_comms_latch(self, exc: BaseException) -> bool:
        """Only the benign 0xD timeout is worth a recovery pass.

        A MotorCommunicationError is the motor not answering at all. A
        MotorFaultError is how yam.arm._check_faults surfaces a latched error
        word, and 0xD is the one that clears on request -- over-temperature,
        overcurrent and overload are physical and are left latched.
        """
        if isinstance(exc, MotorCommunicationError):
            return True
        return isinstance(exc, MotorFaultError) and COMMS_LOST_TEXT in str(exc)

    def _on_error(self, exc: BaseException) -> bool:
        if not self._is_comms_latch(exc):
            self._fail(exc, "not a comms latch -- physical condition, left alone")
            return False
        if self._recovery_used:
            self._fail(exc, "recovery pass already spent")
            return False

        self._recovery_used = True
        self._log("recover", f"{type(exc).__name__}: {exc}")
        try:
            stale = self.arm.recover_stale_motors()
            self.arm.clear_errors()  # recover_stale_motors only clears if it saw 0xD
            state = self.arm.enable()  # clearing the error does not re-enable the motor
        except (MotorCommunicationError, MotorFaultError) as recovery_error:
            self._fail(recovery_error, "the single recovery pass itself failed")
            return False

        self._target = list(state.positions)
        self._retarget = False
        self._ramp_tick = 0
        self._log("recovered", f"stale={', '.join(stale) if stale else 'none reported'}")
        return True

    def _fail(self, exc: BaseException, why: str) -> None:
        self._fault = exc
        self._stop_evt.set()
        self._log("fault", f"{type(exc).__name__}: {exc} [{why}]")
        if self._on_fault is not None:
            self._on_fault(exc)


@contextmanager
def keepalive(arm, **kwargs):
    """Run a keepalive for the duration of a block, surfacing any fault on exit."""
    alive = Keepalive(arm, **kwargs).start()
    try:
        yield alive
    finally:
        alive.stop()
    alive.raise_if_faulted()
