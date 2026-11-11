"""Executor node <-> TeleopController wiring (no ROS graph): which executor
events end teleop. Handlers are called unbound on a stand-in node."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

from std_msgs.msg import Bool

from spot_tools_ros.spot_executor_ros import SpotExecutorRos


class Teleop:
    def __init__(self):
        self.cancels = []

    def cancel(self, reason, force_stop=False):
        self.cancels.append((reason, force_stop))

    @contextmanager
    def plan_control(self, reason):
        self.cancel(reason)
        yield


class Log:
    def info(self, *_):
        pass

    def error(self, *_):
        pass


def node(processing=False):
    calls = []
    ex = SimpleNamespace(
        processing_action_sequence=processing,
        set_paused=lambda v, fb: calls.append(("paused", v)),
        terminate_sequence=lambda fb: calls.append("terminate"),
    )
    return SimpleNamespace(
        teleop=Teleop(),
        spot_executor=ex,
        feedback_collector=SimpleNamespace(),
        background_thread=None,
        get_logger=lambda: Log(),
        _plan_id_lock=threading.Lock(),
        _current_plan_id=None,
    ), calls


def test_resume_of_running_plan_cancels_teleop():
    n, calls = node(processing=True)
    SpotExecutorRos.handle_resume(n, Bool(data=True))
    assert n.teleop.cancels == [("plan resumed", False)]
    assert calls == [("paused", False)]


def test_resume_without_plan_keeps_teleop():
    n, _ = node(processing=False)
    SpotExecutorRos.handle_resume(n, Bool(data=True))
    assert n.teleop.cancels == []


def test_stop_forces_a_stop_even_without_plan():
    n, calls = node()
    SpotExecutorRos.handle_stop(n, Bool(data=True))
    assert n.teleop.cancels == [("stop requested", True)]
    assert calls == []  # no plan thread to terminate


def test_duplicate_plan_does_not_cancel_teleop():
    n, _ = node()
    n._current_plan_id = "p1"
    SpotExecutorRos._process_action_sequence(n, SimpleNamespace(plan_id="p1"))
    assert n.teleop.cancels == []


def test_incoming_plan_reserves_motion_before_worker_starts(monkeypatch):
    from robot_executor_msgs.msg import ActionSequenceMsg
    from spot_executor.teleop import TeleopController

    class DeferredThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass  # Deliberately delay the worker to expose the handoff gap.

    n, _ = node()
    n.teleop = TeleopController(SimpleNamespace(), n.spot_executor)
    monkeypatch.setattr(
        "spot_tools_ros.spot_executor_ros.threading.Thread", DeferredThread
    )
    SpotExecutorRos._process_action_sequence(n, ActionSequenceMsg(plan_id="new"))
    assert n.spot_executor.processing_action_sequence
    assert not n.teleop.cmd_vel(0.3, 0.0, 0.0)[0]
    assert not n.teleop.sit()[0]
