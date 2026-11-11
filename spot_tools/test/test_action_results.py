"""Exceptional action failures reach the per-action result channel exactly once."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import ActionSequence, Follow

import spot_executor.spot_executor as se


@pytest.mark.parametrize("setup_fails", [False, True])
def test_exception_reports_failure_with_action_identity(setup_fails):
    def boom(*_):
        raise RuntimeError("skill failed")

    spot = NS(
        robot=NS(time_sync=NS(wait_for_sync=boom if setup_fails else lambda: None)),
        take_lease=lambda: None,
    )
    ex = se.SpotExecutor(spot, None, None, None)
    ex.execute_follow = boom
    results = []
    feedback = NS(
        print=lambda *_: None, action_result=lambda *args: results.append(args)
    )
    action = Follow("map", np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="skill failed"):
        ex.process_action_sequence(ActionSequence("plan", "robot", [action]), feedback)
    assert results == [(action, 0, "FAILED", "RuntimeError: skill failed")]
    assert not ex.processing_action_sequence


def test_broken_feedback_does_not_mask_skill_exception():
    def boom(*_):
        raise RuntimeError("skill failed")

    ex = se.SpotExecutor(
        NS(robot=NS(time_sync=NS(wait_for_sync=lambda: None)), take_lease=lambda: None),
        None,
        None,
        None,
    )
    ex.execute_follow = boom
    feedback = NS(print=lambda *_: None, action_result=boom)
    with pytest.raises(RuntimeError, match="skill failed"):
        ex.process_action_sequence(
            ActionSequence("p", "r", [Follow("map", np.zeros((2, 3)))]), feedback
        )
    assert not ex.processing_action_sequence
