from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from robot_executor_interface.action_descriptions import Gaze
from spot_executor.spot_executor import SpotExecutor
import spot_executor.spot_executor as module


def setup_gaze(monkeypatch):
    executor = SpotExecutor.__new__(SpotExecutor)
    executor.spot_interface = Mock()
    executor.spot_interface.get_pose.return_value = [2., 3., -.4]
    executor.transform_lookup = lambda *args: ([1., 2., 0.], SimpleNamespace(x=0., y=0., z=0., w=1.))
    turn, wait, arm = Mock(return_value=17), Mock(return_value=True), Mock(return_value=True)
    monkeypatch.setattr(module, 'turn_to_point', turn)
    monkeypatch.setattr(module, 'wait_for_navigation', wait)
    monkeypatch.setattr(module, 'gaze_at_vision_pose', arm)
    command = Gaze(frame='map', robot_point=np.array([1.,1.,0.]),
                   gaze_point=np.array([.5,1.,.6]), object_id='o1', stow_after=False)
    return executor, command, Mock(break_out_of_waiting_loop=False), turn, wait, arm


def test_pickup_gaze_aims_arm_without_turning_the_reviewed_body_stance(monkeypatch):
    executor, command, feedback, turn, wait, arm = setup_gaze(monkeypatch)
    assert executor.execute_gaze(command, feedback, pick_next=True)
    turn.assert_not_called()
    wait.assert_not_called()
    assert np.allclose(arm.call_args.args[1], [1.5, 3., .6])
    assert arm.call_args.kwargs['stow_after'] is False


def test_standalone_gaze_requires_successful_body_turn(monkeypatch):
    executor, command, feedback, turn, wait, arm = setup_gaze(monkeypatch)
    wait.return_value = False
    assert not executor.execute_gaze(command, feedback)
    turn.assert_called_once()
    wait.assert_called_once()
    arm.assert_not_called()


@pytest.mark.parametrize('pick_next', [False, True])
def test_cancelled_gaze_cannot_start_body_or_arm_motion(monkeypatch, pick_next):
    executor, command, feedback, turn, wait, arm = setup_gaze(monkeypatch)
    feedback.break_out_of_waiting_loop = True
    assert not executor.execute_gaze(command, feedback, pick_next=pick_next)
    turn.assert_not_called()
    arm.assert_not_called()
