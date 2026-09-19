import numpy as np
import pytest
from spot_skills.navigation_utils import navigation_timeout


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
