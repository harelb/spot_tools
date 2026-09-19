import numpy as np
import pytest
from types import SimpleNamespace
from bosdyn.api import robot_state_pb2
from bosdyn.client.math_helpers import SE2Pose
from spot_skills.navigation_utils import navigation_timeout
from spot_skills import navigation_utils


def test_short_motion_and_in_place_turn_have_time_to_settle():
    assert navigation_timeout([[0,0,-.7],[0,.24,0]],15)==15
    assert navigation_timeout([[0,0,0],[0,0,np.pi]],15)>20
    assert navigation_timeout([[0,0,np.pi-.01],[0,0,-np.pi+.01]],15)==15
    assert navigation_timeout([[0,0,0],[2,0,0]],15)==30
    with pytest.raises(ValueError):navigation_timeout([[0,0,float('nan')]],15)


def test_slow_runtime_budget_preserves_default_and_wrap(monkeypatch):
    monkeypatch.setenv('SPOT_SKILL_NAVIGATION_TIMEOUT_S','30')
    assert navigation_timeout([[0,0,0],[0,0,2.2]],15)==30
    monkeypatch.setenv('SPOT_SKILL_NAVIGATION_TIMEOUT_S','45')
    assert navigation_timeout([[0,0,0],[.35,0,0]],15)==45
    monkeypatch.setenv('SPOT_SKILL_NAVIGATION_TIMEOUT_S','46')
    with pytest.raises(ValueError):navigation_timeout([[0,0,0]],15)


@pytest.mark.parametrize('value',['0','-1','nan','inf','46'])
def test_invalid_navigation_budget_fails_closed(monkeypatch,value):
    monkeypatch.setenv('SPOT_SKILL_NAVIGATION_TIMEOUT_S',value)
    with pytest.raises(ValueError):navigation_timeout([[0,0,0],[.35,0,0]],15)


@pytest.mark.parametrize('budget,expected',[(None,10.),('30',30.),('45',30.)])
@pytest.mark.parametrize('holding',[False,True])
def test_skill_budget_preserves_sdk_command_expiry(monkeypatch,budget,expected,holding):
    if budget is None:
        monkeypatch.delenv('SPOT_SKILL_NAVIGATION_TIMEOUT_S',raising=False)
    else:
        monkeypatch.setenv('SPOT_SKILL_NAVIGATION_TIMEOUT_S',budget)
    monkeypatch.setattr(navigation_utils.time,'time',lambda:1000.)
    issued=[]
    def command(**kwargs):
        issued.append(kwargs)
        return 17
    client=SimpleNamespace(robot_command=command)
    state=robot_state_pb2.RobotState()
    state.manipulator_state.is_gripper_holding_item=holding
    spot=SimpleNamespace(robot=SimpleNamespace(ensure_client=lambda _:client),
                         get_state=lambda:state)
    assert navigation_utils.navigate_to_absolute_pose(spot,SE2Pose(1.,2.,.3))==17
    assert issued[0]['end_time_secs']==1000.+expected
    trajectory=issued[0]['command'].synchronized_command.mobility_command.se2_trajectory_request
    assert trajectory.trajectory.points[0].pose.position.x==1.
    assert trajectory.trajectory.points[0].pose.position.y==2.
