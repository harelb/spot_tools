"""Operator teleoperation through the Spot executor.

The executor holds the Spot body lease, so operator driving (the web console's
Teleop panel) goes through it instead of a second lease holder (spot_twist_node
takes its own lease and fights the executor).  This module is ROS free; the
executor node wires ``~/cmd_vel`` / ``~/stand`` / ``~/sit`` to it.

Safety rules, all enforced here (the console has its own copies, but the robot
must not trust the radio):

* hard caps: every velocity is clamped to ``max_vx`` / ``max_vy`` / ``max_wz``;
  the caps themselves can only be configured *lower* than ``HARD_MAX``.
* deadman: an active teleop with no ``cmd_vel`` for ``deadman_s`` is stopped
  (zero velocity).  Each velocity command also carries a short robot-side end
  time, so even a dead executor stops the robot.
* plan interlock: teleop is accepted only when no plan is executing or the plan
  is paused, and only while the executor owns the lease.  A plan that arrives
  (or is resumed) while teleoping cancels teleop.
"""

from __future__ import annotations

import inspect
import math
import threading
import time

import numpy as np

# Absolute ceilings; the configured caps may only be lower.
HARD_MAX = {"vx": 0.6, "vy": 0.4, "wz": 0.8}
DEADMAN_S = 0.5
# Robot-side end time of one velocity command: Spot stops on its own this long
# after the last command, whatever happens to the executor.
COMMAND_DURATION_S = 0.5
ZERO_EPS = 1e-3


def clamp(value: float, cap: float) -> float:
    return max(-cap, min(cap, float(value)))


def send_body_velocity(spot, vx: float, vy: float, wz: float, duration_s: float):
    """Body-frame velocity via whichever interface the Spot exposes.

    Real Spot and SimSpot implement ``set_twist`` (the real one sends a synchro
    velocity command with a short end time); FakeSpot implements ``set_vel``.
    """
    set_twist = getattr(spot, "set_twist", None)
    if set_twist is not None:
        try:
            takes_duration = "duration_s" in inspect.signature(set_twist).parameters
        except (TypeError, ValueError):
            takes_duration = False
        if takes_duration:
            set_twist(vx, vy, wz, duration_s=duration_s)
        else:
            set_twist(vx, vy, wz)
        return
    set_vel = getattr(spot, "set_vel", None)
    if set_vel is not None:
        set_vel(np.array([vx, vy, 0.0]), np.array([0.0, 0.0, wz]))
        return
    raise RuntimeError("spot interface has neither set_twist nor set_vel")


def standing_from_state(robot_state):
    """RobotState -> True (standing/stepping), False (not ready, e.g. sitting) or
    None (unknown: older SDK, sim, transition)."""
    try:
        behavior = getattr(robot_state, "behavior_state", None)
        state = int(getattr(behavior, "state", 0) or 0) if behavior is not None else 0
    except Exception:  # noqa: BLE001 -- proto shape drift must not raise
        return None
    # bosdyn BehaviorState: 1 NOT_READY, 2 TRANSITION, 3 STANDING, 4 STEPPING
    if state in (3, 4):
        return True
    if state == 1:
        return False
    return None


def _owner_ours(owner) -> bool:
    return str(owner or "").startswith("understanding")


class TeleopController:
    """Velocity / stand / sit for the operator, guarded by caps, a deadman and
    the plan interlock.  Thread safe: ROS callbacks and the deadman timer run on
    different threads."""

    def __init__(
        self,
        spot,
        executor,
        max_vx: float = HARD_MAX["vx"],
        max_vy: float = HARD_MAX["vy"],
        max_wz: float = HARD_MAX["wz"],
        deadman_s: float = DEADMAN_S,
        command_duration_s: float = COMMAND_DURATION_S,
        clock=time.monotonic,
        log=None,
    ):
        self.spot = spot
        self.executor = executor
        self.caps = {
            "vx": min(abs(float(max_vx)), HARD_MAX["vx"]),
            "vy": min(abs(float(max_vy)), HARD_MAX["vy"]),
            "wz": min(abs(float(max_wz)), HARD_MAX["wz"]),
        }
        self.deadman_s = float(deadman_s)
        self.command_duration_s = float(command_duration_s)
        self.clock = clock
        self.log = log or (lambda level, msg: None)
        self._lock = threading.RLock()
        self.active = False
        self.busy = ""  # "stand" / "sit" while that blocking call runs
        self.last_cmd_t = None
        self.commanded = (0.0, 0.0, 0.0)
        self.standing = None  # last stand/sit outcome; None = unknown
        self.last_reject = ""
        self.last_event = ""
        self.commands = 0
        self.rejected = 0

    # ------------------------------------------------------------ interlock
    def plan_executing(self) -> bool:
        ex = self.executor
        return bool(getattr(ex, "processing_action_sequence", False)) and not bool(
            getattr(ex, "paused", False)
        )

    def interlock(self) -> str:
        """Why teleop is not allowed right now ("" = allowed)."""
        if self.plan_executing():
            return "plan executing: pause it to teleop"
        lm = getattr(self.executor, "lease_manager", None)
        if lm is not None:
            if getattr(lm, "taking_back_lease", False):
                return "lease manager is taking the lease back"
            owner = getattr(lm, "owner_name", None)
            if owner is not None and not _owner_ours(owner):
                return f"lease held by {owner or 'nobody'}: take the lease first"
        return ""

    # ------------------------------------------------------------- velocity
    def cmd_vel(self, vx: float, vy: float, wz: float) -> tuple[bool, str]:
        """One operator velocity command (body frame).  ``(accepted, reason)``."""
        try:
            raw = [float(vx), float(vy), float(wz)]
        except (TypeError, ValueError):
            return self._reject("non-numeric velocity")
        if not all(math.isfinite(v) for v in raw):
            return self._reject("non-finite velocity")
        cmd = (
            clamp(raw[0], self.caps["vx"]),
            clamp(raw[1], self.caps["vy"]),
            clamp(raw[2], self.caps["wz"]),
        )
        zero = all(abs(v) < ZERO_EPS for v in cmd)
        with self._lock:
            if zero and not self.active:
                # an armed but idle console streams zeros: nothing to do, and
                # not worth a rejection while a plan runs
                return True, "idle"
            reason = self.busy and f"{self.busy} in progress" or self.interlock()
            if reason:
                if self.active:
                    self._halt(f"teleop cancelled: {reason}")
                return self._reject(reason)
            try:
                send_body_velocity(self.spot, *cmd, self.command_duration_s)
            except Exception as e:  # noqa: BLE001 -- e.g. LeaseUseError
                self.active = False
                return self._reject(f"velocity command failed: {type(e).__name__}: {e}")
            if not self.active:
                self.last_event = "teleop started"
                self.log("INFO", "Teleop started")
            self.active = True
            self.last_cmd_t = self.clock()
            self.commanded = cmd
            self.commands += 1
            self.last_reject = ""
            return True, "ok"

    def tick(self) -> None:
        """Deadman + interlock watch; call at >= 10 Hz."""
        with self._lock:
            if not self.active:
                return
            age = self.clock() - (self.last_cmd_t or 0.0)
            if age > self.deadman_s:
                if all(abs(v) < ZERO_EPS for v in self.commanded):
                    # the operator stopped (last command zero), then went quiet
                    self._halt(f"teleop idle: no cmd_vel for {age:.2f} s")
                else:
                    self._halt(f"deadman: no cmd_vel for {age:.2f} s")
                return
            reason = self.interlock()
            if reason:
                self._halt(f"teleop cancelled: {reason}")

    def cancel(self, reason: str, force_stop: bool = False) -> None:
        """End teleop (a new or resumed plan, an operator stop).  ``force_stop``
        commands a stop even when teleop was not active (operator Stop)."""
        with self._lock:
            if self.active:
                self._halt(reason)
            elif force_stop:
                self._stop_motion()
                self.last_event = reason

    def _halt(self, reason: str) -> None:
        self.active = False
        self.commanded = (0.0, 0.0, 0.0)
        self.last_event = reason
        self.log("INFO", f"Teleop stopped ({reason})")
        # A plan that took over owns the motion now: do not fight it.
        if not self.plan_executing():
            self._stop_motion()

    def _stop_motion(self) -> None:
        lm = getattr(self.executor, "lease_manager", None)
        owner = getattr(lm, "owner_name", None) if lm is not None else None
        if owner is not None and not _owner_ours(owner):
            return  # someone else (the tablet) is in control; leave them be
        try:
            # Spot.stop() halts in place and never stands a sitting robot (a
            # velocity command, even zero, would); SimSpot only has set_twist.
            if hasattr(self.spot, "stop"):
                self.spot.stop()
            else:
                send_body_velocity(self.spot, 0.0, 0.0, 0.0, self.command_duration_s)
        except Exception as e:  # noqa: BLE001
            self.log("WARNING", f"Teleop: stop failed: {e}")

    def _reject(self, reason: str) -> tuple[bool, str]:
        with self._lock:
            self.rejected += 1
            if reason != self.last_reject:
                self.log("WARNING", f"Teleop command rejected: {reason}")
            self.last_reject = reason
        return False, reason

    # ----------------------------------------------------------- stand / sit
    def stand(self) -> tuple[bool, str]:
        return self._posture("stand", True)

    def sit(self) -> tuple[bool, str]:
        return self._posture("sit", False)

    def _posture(self, what: str, standing: bool) -> tuple[bool, str]:
        with self._lock:
            if self.busy:
                return self._reject(f"{self.busy} in progress")
            reason = self.interlock()
            if reason:
                return self._reject(reason)
            if self.active:
                self._halt(f"{what} requested")
            self.busy = what
        try:
            getattr(self.spot, what)()
        except Exception as e:  # noqa: BLE001 -- AssertionError, lease errors ...
            msg = f"{what} failed: {type(e).__name__}: {e}"
            with self._lock:
                self.busy = ""
                self.last_event = msg
            return False, msg
        with self._lock:
            self.busy = ""
            self.standing = standing
            self.last_event = f"{what} done"
        return True, f"{what} done"

    # ---------------------------------------------------------------- state
    def robot_standing(self):
        """Standing from the robot state when it says, else the last stand/sit."""
        try:
            st = standing_from_state(self.spot.get_state())
        except Exception:  # noqa: BLE001
            st = None
        return self.standing if st is None else st

    def state(self, standing=None) -> dict:
        with self._lock:
            age = None if self.last_cmd_t is None else self.clock() - self.last_cmd_t
            reason = self.interlock()
            return {
                "active": self.active,
                "allowed": not reason,
                "interlock": reason,
                "busy": self.busy,
                "standing": standing if standing is not None else self.standing,
                "commanded": {
                    "vx": self.commanded[0],
                    "vy": self.commanded[1],
                    "wz": self.commanded[2],
                },
                "caps": dict(self.caps),
                "deadman_s": self.deadman_s,
                "cmd_age_s": age,
                "last_reject": self.last_reject,
                "last_event": self.last_event,
                "commands": self.commands,
                "rejected": self.rejected,
                "plan_executing": self.plan_executing(),
                "plan_paused": bool(getattr(self.executor, "paused", False)),
            }

    def notes(self, standing=None) -> str:
        """Short teleop summary for RuntimeGuardsMsg.notes."""
        st = self.state(standing)
        parts = [
            "teleop=" + ("active" if st["active"] else "idle"),
            "standing="
            + {True: "yes", False: "no", None: "unknown"}[st["standing"]],
        ]
        if st["busy"]:
            parts.append(f"busy={st['busy']}")
        if st["last_reject"]:
            parts.append(f"teleop_rejected={st['last_reject']}")
        return "; ".join(parts)
