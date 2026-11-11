"""A failed scheduled action cannot release other robots' dependent tasks."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Pick
from robot_executor_interface.plan_progress import PlanProgress

import spot_executor.spot_executor as se


class Feedback:
    def __init__(self):
        self.reports = []
        self.messages = []

    def print(self, level, message):
        self.messages.append((level, str(message)))

    def report_action_done(self, sequence, command, success):
        self.reports.append((command.task_id, success))


def executor(progress):
    stops = []
    spot = NS(
        robot=NS(time_sync=NS(wait_for_sync=lambda: None)),
        take_lease=lambda: None,
        stop=lambda: stops.append(True),
    )
    return se.SpotExecutor(spot, None, None, None, plan_progress=progress), stops


@pytest.mark.parametrize("fails_at", [0, 1])
def test_failed_task_never_announces_done_or_runs_its_remaining_actions(
    monkeypatch, fails_at
):
    sent = []
    progress = PlanProgress(publish=lambda *message: sent.append(message))
    ex, stops = executor(progress)
    pick = Pick("map", "cone", np.zeros(3), np.zeros(3), "o1", task_id="remove")
    carry = Follow("map", np.zeros((2, 3)), task_id="remove")
    successor = Follow(
        "map", np.zeros((2, 3)), task_id="cross", after_task_ids=["remove"]
    )
    calls = []

    def execute(command, *args, **kwargs):
        calls.append(command)
        return command is not [pick, carry][fails_at]

    ex.execute_pick = ex.execute_follow = execute
    monkeypatch.setattr(se.time, "sleep", lambda *_: None)
    feedback = Feedback()
    ex.process_action_sequence(
        ActionSequence("p", "robot", [pick, carry, successor]), feedback
    )
    assert len(calls) == fails_at + 3
    assert all(command is [pick, carry][fails_at] for command in calls[fails_at:])
    assert feedback.reports[-1] == ("remove", False)
    assert sent == []
    assert stops == [True]
    assert not ex.processing_action_sequence
    assert progress.blocked_on("p", successor, "partner") == [
        "task remove not finished"
    ]


def test_wait_warning_uses_supported_ros_log_level_and_keeps_waiting(monkeypatch):
    pytest.importorskip("rclpy")
    from spot_tools_ros.spot_executor_ros import RosFeedbackCollector

    progress = PlanProgress()
    ex, _ = executor(progress)
    command = Follow("map", np.zeros((2, 3)), task_id="b", after_task_ids=["a"])
    feedback = RosFeedbackCollector("map", "/tmp", "")
    warnings = []
    feedback.logger = NS(info=lambda *_: None, warning=warnings.append)
    sleeps = []
    times = iter([1.0, 32.0])
    monkeypatch.setattr(se.time, "time", lambda: next(times))

    def sleep(_):
        sleeps.append(True)
        if len(sleeps) == 2:
            progress.announce_done("p", "partner", "a")

    monkeypatch.setattr(se.time, "sleep", sleep)
    executed = []
    ex.execute_follow = lambda action, fb: executed.append(action) or True
    feedback.report_action_done = lambda *_: None
    ex.process_action_sequence(ActionSequence("p", "robot", [command]), feedback)
    assert len(warnings) == 1 and "Still waiting" in warnings[0]
    assert executed == [command]


def test_progress_reaches_both_shared_ros_and_radio_uplink():
    pytest.importorskip("rclpy")
    from builtin_interfaces.msg import Time
    from spot_tools_ros.spot_executor_ros import SpotExecutorRos

    shared, uplink = [], []
    node = NS(
        get_clock=lambda: NS(now=lambda: NS(to_msg=Time)),
        plan_progress_pub=NS(publish=shared.append),
        plan_progress_relay_pub=NS(publish=uplink.append),
    )
    SpotExecutorRos._publish_plan_progress(node, "plan", "robot", "task", "DONE")
    assert len(shared) == len(uplink) == 1
    assert shared[0] == uplink[0]
    assert (
        shared[0].plan_id,
        shared[0].robot_name,
        shared[0].task_id,
        shared[0].state,
    ) == ("plan", "robot", "task", "DONE")
