import pytest
from spot_skills.timeouts import skill_timeout


def test_robot_defaults_and_independent_deployment_budgets(monkeypatch):
    monkeypatch.delenv('SPOT_SKILL_ARM_TIMEOUT_S',raising=False)
    monkeypatch.delenv('SPOT_SKILL_GRASP_TIMEOUT_S',raising=False)
    assert skill_timeout(2.)==2.
    assert skill_timeout(15.,'grasp')==15.
    monkeypatch.setenv('SPOT_SKILL_ARM_TIMEOUT_S','45')
    assert skill_timeout(2.)==45.
    assert skill_timeout(15.,'grasp')==15.


@pytest.mark.parametrize('value',['0','-1','nan','inf','121'])
def test_invalid_budget_fails_closed(monkeypatch,value):
    monkeypatch.setenv('SPOT_SKILL_ARM_TIMEOUT_S',value)
    with pytest.raises(ValueError):skill_timeout(2.)
