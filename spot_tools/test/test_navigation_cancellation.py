from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from spot_skills.navigation_utils import follow_trajectory_continuous,wait_for_navigation


def test_base_wait_cancellation_stops_before_requesting_arm():
    spot=Mock()
    assert not wait_for_navigation(spot,1,cancelled=lambda:True)
    spot.command_client.robot_command_feedback.assert_not_called()
    assert spot.command_client.robot_command.call_args.args[0].full_body_command.HasField('stop_request')


def test_base_wait_requires_at_goal_feedback():
    from bosdyn.api import robot_command_pb2,basic_command_pb2
    spot=Mock();response=robot_command_pb2.RobotCommandFeedbackResponse()
    response.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status=basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL
    spot.command_client.robot_command_feedback.return_value=response
    assert wait_for_navigation(spot,1)
    spot.command_client.robot_command_feedback.assert_called_once_with(1)


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


def test_final_waypoint_heading_is_executed_and_verified(monkeypatch):
    import shapely
    import spot_skills.navigation_utils as nav
    spot=Mock();spot.get_pose.return_value=[1.,0.,0.]
    feedback=Mock(break_out_of_waiting_loop=False)
    output=SimpleNamespace(path_shapely=shapely.LineString([[0,0],[1,0]]),path_waypoints_metric=[],target_point_metric=None)
    planner=Mock();planner.plan_path.return_value=(True,output)
    sent=[]
    monkeypatch.setattr(nav,'navigate_to_absolute_pose',lambda s,pose,*a,**k:sent.append(pose) or 17)
    wait=Mock(return_value=False);monkeypatch.setattr(nav,'wait_for_navigation',wait)
    assert not nav.follow_trajectory_continuous(spot,np.array([[0.,0.,0.],[1.,0.,1.2]]),.5,.1,30,planner,feedback=feedback)
    assert sent[0].angle==1.2
    assert wait.call_args.args==(spot,17)


@pytest.mark.parametrize("raster_path", [[[0,0],[1,0]], [[0,0],[.1,0],[.2,.05],[1,.05]]])
def test_cross_track_translation_preserves_straight_route_heading(monkeypatch,raster_path):
    import shapely
    import spot_skills.navigation_utils as nav
    spot=Mock();spot.get_pose.return_value=[0.,.05,0.]
    feedback=Mock(break_out_of_waiting_loop=False)
    output=SimpleNamespace(path_shapely=shapely.LineString(raster_path),path_waypoints_metric=[],target_point_metric=None)
    planner=Mock();planner.plan_path.return_value=(True,output)
    sent=[]
    def navigate(s,pose,*a,**k):
        sent.append(pose);feedback.break_out_of_waiting_loop=True;return 17
    monkeypatch.setattr(nav,'navigate_to_absolute_pose',navigate)
    monkeypatch.setattr(nav.time,'sleep',lambda _:None)
    assert not nav.follow_trajectory_continuous(spot,np.array([[0.,0.,0.],[1.,0.,0.]]),.15,.1,30,planner,feedback=feedback)
    assert sent[0].x>0
    assert sent[0].angle==pytest.approx(0.)
