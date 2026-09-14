from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from spot_skills.navigation_utils import follow_trajectory_continuous


@pytest.mark.parametrize('cancel_in_planner',[False,True])
def test_cancel_never_sends_another_waypoint(cancel_in_planner):
    spot=Mock();feedback=SimpleNamespace(break_out_of_waiting_loop=not cancel_in_planner)
    planner=Mock()
    def plan(*args):
        feedback.break_out_of_waiting_loop=True
        return True,None
    planner.plan_path.side_effect=plan
    assert not follow_trajectory_continuous(spot,np.array([[0,0],[1,0]]),.5,.1,30,planner,feedback=feedback)
    spot.command_client.robot_command.assert_called_once()
    command=spot.command_client.robot_command.call_args.args[0]
    assert command.full_body_command.HasField('stop_request')
    spot.get_pose.assert_not_called()
    assert planner.plan_path.call_count==int(cancel_in_planner)
