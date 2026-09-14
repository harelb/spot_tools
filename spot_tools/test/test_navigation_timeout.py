import numpy as np
import pytest
from spot_skills.navigation_utils import navigation_timeout


def test_short_motion_and_in_place_turn_have_time_to_settle():
    assert navigation_timeout([[0,0,-.7],[0,.24,0]],15)==15
    assert navigation_timeout([[0,0,0],[0,0,np.pi]],15)>20
    assert navigation_timeout([[0,0,np.pi-.01],[0,0,-np.pi+.01]],15)==15
    assert navigation_timeout([[0,0,0],[2,0,0]],15)==30
    with pytest.raises(ValueError):navigation_timeout([[0,0,float('nan')]],15)
