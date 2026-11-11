"""Follow actions the GRSTAPS-X plans produce: a leg that is one waypoint (the
robot's next task is where it stands), and an action that raises."""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import spot_executor.spot_executor as se

    from robot_executor_interface.action_descriptions import ActionSequence, Follow
except ImportError:  # spot_tools (bosdyn) not installed
    se = None

pytestmark = pytest.mark.skipif(se is None, reason="spot_tools not installed")


class IdentityRotation:
    x = y = z = 0.0
    w = 1.0


class Feedback:
    def __init__(self):
        self.reports = []

    def print(self, level, msg):
        pass

    def follow_path_feedback(self, path):
        pass

    def report_action_done(self, sequence, command, success):
        self.reports.append((type(command).__name__, success))


def make_executor(robot_xy=(0.0, 0.0)):
    spot = SimpleNamespace(
        is_fake=True,
        get_pose=lambda: np.array([robot_xy[0], robot_xy[1], 0.0]),
        robot=SimpleNamespace(time_sync=SimpleNamespace(wait_for_sync=lambda: None)),
        take_lease=lambda: None,
    )
    ex = se.SpotExecutor(
        spot,
        detector=None,
        transform_lookup=lambda parent, child: (np.zeros(3), IdentityRotation()),
        planner=None,
    )
    ex.lease_manager = None
    return ex


def test_a_one_waypoint_follow_drives_from_where_the_robot_is(monkeypatch):
    sent = []

    def follow(spot, path, *a, **k):
        sent.append(np.array(path))
        return True

    monkeypatch.setattr(se, "follow_trajectory_continuous", follow)
    ex = make_executor(robot_xy=(1.0, 2.0))
    assert ex.execute_follow(
        Follow(frame="map", path2d=np.array([[3.0, 4.0, 0.0]])), Feedback()
    )
    (path,) = sent
    assert path.shape[0] == 2
    assert path[0, :2].tolist() == [1.0, 2.0] and path[1, :2].tolist() == [3.0, 4.0]


def test_an_action_that_raises_is_reported_as_failed(monkeypatch):
    def follow(*a, **k):
        raise ValueError("point array must contain 0 or >1 elements")

    monkeypatch.setattr(se, "follow_trajectory_continuous", follow)
    ex = make_executor()
    feedback = Feedback()
    seq = ActionSequence(
        plan_id="p1",
        robot_name="testbot",
        actions=[
            Follow(frame="map", path2d=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        ],
    )
    with pytest.raises(ValueError):
        ex.process_action_sequence(seq, feedback)
    assert feedback.reports == [("Follow", False)]
