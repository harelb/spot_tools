"""Follow budgets include the robot's approach and never drop below five seconds."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import Follow

import spot_executor.spot_executor as se


@pytest.mark.parametrize(
    "path, seconds_per_meter, expected",
    [([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]], 10.0, 50.0), ([[0.0, 0.0, 0.0]], 10.0, 5.0)],
)
def test_follow_budget(monkeypatch, path, seconds_per_meter, expected):
    budgets = []
    spot = NS(get_pose=lambda: np.zeros(3))
    rotation = NS(x=0.0, y=0.0, z=0.0, w=1.0)
    ex = se.SpotExecutor(
        spot,
        None,
        lambda *_: (np.zeros(3), rotation),
        None,
        follow_timeout_per_meter=seconds_per_meter,
    )
    monkeypatch.setattr(
        se, "follow_trajectory_continuous", lambda *a, **k: budgets.append(a[4]) or True
    )
    feedback = NS(print=lambda *_: None, follow_path_feedback=lambda *_: None)
    assert ex.execute_follow(Follow("map", np.asarray(path)), feedback)
    assert budgets == [expected]
