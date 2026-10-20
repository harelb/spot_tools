"""Teleop through the executor: caps, deadman, plan interlock, stand/sit.
No robot: a recording fake Spot, a fake executor and a fake clock; one test
drives the real FakeSpot."""

from types import SimpleNamespace

import numpy as np
import pytest

from spot_executor.fake_spot import FakeSpot
from spot_executor.teleop import (
    HARD_MAX,
    TeleopController,
    send_body_velocity,
    standing_from_state,
)

OURS = "understanding-hamilton"
TABLET = "bosdyn.android.spotapp spot-tablet"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TwistSpot:
    """Real-Spot shaped: set_twist(duration_s) + stop + stand/sit."""

    def __init__(self, fail=None):
        self.twists = []
        self.calls = []
        self.fail = fail

    def set_twist(self, vx, vy, v_rot, duration_s=0.5):
        if self.fail:
            raise self.fail
        self.twists.append((vx, vy, v_rot, duration_s))

    def stop(self):
        self.calls.append("stop")

    def stand(self):
        self.calls.append("stand")

    def sit(self):
        self.calls.append("sit")

    def get_state(self):
        return SimpleNamespace()


def executor(owner=OURS, running=False, paused=False, taking_back=False):
    return SimpleNamespace(
        processing_action_sequence=running,
        paused=paused,
        lease_manager=SimpleNamespace(owner_name=owner, taking_back_lease=taking_back),
    )


def make(spot=None, ex=None, **kw):
    clock = Clock()
    spot = spot or TwistSpot()
    ex = ex or executor()
    return TeleopController(spot, ex, clock=clock, **kw), spot, ex, clock


# ------------------------------------------------------------------ caps
def test_velocity_is_clamped_to_caps():
    t, spot, _, _ = make()
    ok, _ = t.cmd_vel(5.0, -5.0, 9.0)
    assert ok
    vx, vy, wz, dur = spot.twists[-1]
    assert (vx, vy, wz) == (HARD_MAX["vx"], -HARD_MAX["vy"], HARD_MAX["wz"])
    assert dur == pytest.approx(0.5)


def test_caps_can_only_be_lowered():
    t, spot, _, _ = make(max_vx=2.0, max_vy=0.1, max_wz=-0.3)
    assert t.caps == {"vx": HARD_MAX["vx"], "vy": 0.1, "wz": 0.3}
    t.cmd_vel(1.0, 1.0, -1.0)
    assert spot.twists[-1][:3] == (HARD_MAX["vx"], 0.1, -0.3)


def test_within_caps_passes_through():
    t, spot, _, _ = make()
    t.cmd_vel(0.25, -0.1, 0.4)
    assert spot.twists[-1][:3] == (0.25, -0.1, 0.4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x"])
def test_non_finite_velocity_rejected(bad):
    t, spot, _, _ = make()
    ok, reason = t.cmd_vel(bad, 0, 0)
    assert not ok and spot.twists == []
    assert t.rejected == 1


# --------------------------------------------------------------- deadman
def test_deadman_stops_after_silence():
    t, spot, _, clock = make()
    t.cmd_vel(0.3, 0, 0)
    assert t.active
    clock.t += 0.4
    t.tick()
    assert t.active and spot.calls == []
    clock.t += 0.2  # 0.6 s since the last command
    t.tick()
    assert not t.active
    assert spot.calls == ["stop"]
    assert "deadman" in t.last_event
    assert t.state()["commanded"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}


def test_quiet_after_zero_is_idle_not_deadman():
    t, spot, _, clock = make()
    t.cmd_vel(0.3, 0, 0)
    t.cmd_vel(0, 0, 0)
    clock.t += 0.6
    t.tick()
    assert not t.active and t.last_event.startswith("teleop idle")


def test_steady_commands_keep_deadman_alive():
    t, spot, _, clock = make()
    for _ in range(20):
        t.cmd_vel(0.3, 0, 0)
        clock.t += 0.1
        t.tick()
    assert t.active and "stop" not in spot.calls


def test_idle_zeros_are_not_sent():
    t, spot, _, _ = make()
    ok, why = t.cmd_vel(0, 0, 0)
    assert ok and why == "idle"
    assert spot.twists == [] and not t.active


def test_zero_while_active_is_sent_and_refreshes_deadman():
    t, spot, _, clock = make()
    t.cmd_vel(0.3, 0, 0)
    clock.t += 0.4
    t.cmd_vel(0, 0, 0)
    assert spot.twists[-1][:3] == (0.0, 0.0, 0.0)
    clock.t += 0.4
    t.tick()
    assert t.active


def test_time_sync_once_before_first_command():
    synced = []
    spot = TwistSpot()
    spot.robot = SimpleNamespace(time_sync=SimpleNamespace(wait_for_sync=lambda: synced.append(1)))
    t, _, _, _ = make(spot=spot)
    t.cmd_vel(0.1, 0, 0)
    t.cmd_vel(0.2, 0, 0)
    assert synced == [1] and len(spot.twists) == 2


def test_robot_side_end_time_is_short():
    t, spot, _, _ = make()
    t.cmd_vel(0.1, 0, 0)
    assert spot.twists[-1][3] <= 0.5


# ------------------------------------------------------------- interlock
def test_rejected_while_plan_executing():
    t, spot, _, _ = make(ex=executor(running=True))
    ok, reason = t.cmd_vel(0.3, 0, 0)
    assert not ok and "plan executing" in reason
    assert spot.twists == []
    assert "teleop_rejected=plan executing" in t.notes()


def test_zeros_while_plan_executing_are_not_rejections():
    t, spot, _, _ = make(ex=executor(running=True))
    assert t.cmd_vel(0, 0, 0) == (True, "idle")
    assert t.rejected == 0


def test_allowed_while_plan_paused():
    t, spot, _, _ = make(ex=executor(running=True, paused=True))
    ok, _ = t.cmd_vel(0.3, 0, 0)
    assert ok and t.active


def test_rejected_without_lease():
    t, spot, _, _ = make(ex=executor(owner=TABLET))
    ok, reason = t.cmd_vel(0.3, 0, 0)
    assert not ok and "lease held by bosdyn" in reason


def test_rejected_while_lease_taken_back():
    t, _, _, _ = make(ex=executor(taking_back=True))
    ok, reason = t.cmd_vel(0.3, 0, 0)
    assert not ok and "taking the lease back" in reason


def test_resumed_plan_cancels_teleop_on_tick_without_fighting_it():
    t, spot, ex, _ = make(ex=executor(running=True, paused=True))
    t.cmd_vel(0.3, 0, 0)
    ex.paused = False  # plan resumed
    t.tick()
    assert not t.active
    assert spot.calls == []  # the plan owns the motion now
    assert "plan executing" in t.last_event


def test_new_plan_cancel_stops_motion():
    t, spot, _, _ = make()
    t.cmd_vel(0.3, 0, 0)
    t.cancel("new plan p1 received")
    assert not t.active and spot.calls == ["stop"]
    assert t.last_event == "new plan p1 received"


def test_cancel_when_idle_does_nothing_unless_forced():
    t, spot, _, _ = make()
    t.cancel("new plan")
    assert spot.calls == []
    t.cancel("stop requested", force_stop=True)
    assert spot.calls == ["stop"]


def test_forced_stop_leaves_tablet_alone():
    t, spot, _, _ = make(ex=executor(owner=TABLET))
    t.cancel("stop requested", force_stop=True)
    assert spot.calls == []


def test_lost_lease_while_active_cancels():
    t, spot, ex, _ = make()
    t.cmd_vel(0.3, 0, 0)
    ex.lease_manager.owner_name = TABLET
    t.tick()
    assert not t.active
    assert spot.calls == []  # not ours: no command


def test_command_failure_is_reported():
    t, _, _, _ = make(spot=TwistSpot(fail=RuntimeError("LeaseUseError")))
    ok, reason = t.cmd_vel(0.3, 0, 0)
    assert not ok and "LeaseUseError" in reason and not t.active


# ------------------------------------------------------------ stand / sit
def test_stand_and_sit():
    t, spot, _, _ = make()
    assert t.stand() == (True, "stand done")
    assert t.standing is True
    assert t.sit() == (True, "sit done")
    assert t.standing is False
    assert spot.calls == ["stand", "sit"]
    assert "standing=no" in t.notes()


def test_stand_rejected_during_plan():
    t, spot, _, _ = make(ex=executor(running=True))
    ok, reason = t.sit()
    assert not ok and "plan executing" in reason and spot.calls == []


def test_stand_halts_active_teleop_first():
    t, spot, _, _ = make()
    t.cmd_vel(0.3, 0, 0)
    t.sit()
    assert spot.calls == ["stop", "sit"] and not t.active


def test_stand_failure_is_reported():
    class Bad(TwistSpot):
        def stand(self):
            raise AssertionError("Robot not ready for motion.")

    t, _, _, _ = make(spot=Bad())
    ok, msg = t.stand()
    assert not ok and "not ready" in msg and t.busy == ""


def test_velocity_rejected_while_posture_busy():
    t, _, _, _ = make()
    t.busy = "stand"
    ok, reason = t.cmd_vel(0.3, 0, 0)
    assert not ok and "stand in progress" in reason


def test_standing_from_state():
    def st(n):
        return SimpleNamespace(behavior_state=SimpleNamespace(state=n))

    assert standing_from_state(st(3)) is True
    assert standing_from_state(st(4)) is True
    assert standing_from_state(st(1)) is False
    assert standing_from_state(st(2)) is None
    assert standing_from_state(SimpleNamespace()) is None


def test_state_dict():
    t, _, _, _ = make()
    t.cmd_vel(0.2, 0.1, 0.0)
    s = t.state(standing=True)
    assert s["active"] and s["allowed"] and s["standing"] is True
    assert s["commanded"]["vx"] == 0.2 and s["caps"]["vx"] == HARD_MAX["vx"]


# --------------------------------------------------------------- FakeSpot
def test_fake_spot_drives_and_stops():
    spot = FakeSpot(init_pose=np.array([0.0, 0.0, 0.0, 0.0]))
    t, _, _, clock = make(spot=spot)
    t.cmd_vel(0.5, 0.0, 0.0)
    spot.step(1.0)
    assert spot.get_pose()[0] == pytest.approx(0.5)
    clock.t += 1.0
    t.tick()  # deadman
    spot.step(1.0)
    assert spot.get_pose()[0] == pytest.approx(0.5)


def test_send_body_velocity_without_duration_kwarg():
    seen = []

    class Sim:
        def set_twist(self, vx, vy, v_rot):
            seen.append((vx, vy, v_rot))

    send_body_velocity(Sim(), 0.1, 0.2, 0.3, 0.5)
    assert seen == [(0.1, 0.2, 0.3)]
