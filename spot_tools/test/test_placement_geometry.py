import numpy as np
import pytest
from spot_skills.placement_geometry import placement_clearance


def test_observed_payload_below_hand_increases_release_clearance():
    transform=np.eye(4);transform[:3,3]=[.05,0,-.13]
    result=placement_clearance(dict(box=[.2,.16,.14],hand_T_box=transform),np.eye(3))
    assert result['minimum_hand_clearance_m']==pytest.approx(.2)
    assert result['recommended_hand_clearance_m']==pytest.approx(.23)


def test_clearance_uses_preserved_world_attitude_and_full_box_extent():
    transform=np.eye(4);transform[:3,3]=[.12,0,0]
    rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    result=placement_clearance(dict(box=[.16,.1,.1],hand_T_box=transform),rotation)
    assert result['minimum_hand_clearance_m']==pytest.approx(.2)
    with pytest.raises(ValueError):
        placement_clearance(dict(box=[2,.1,.1],hand_T_box=transform),rotation)
    with pytest.raises(ValueError):
        placement_clearance(dict(box=[.1,.1,.1],hand_T_box=transform),rotation*2)
