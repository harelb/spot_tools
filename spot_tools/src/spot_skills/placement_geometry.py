"""Placement clearance from an observed held envelope and grasp attitude."""
import numpy as np


def placement_clearance(observation, world_R_hand):
    half=np.asarray(observation['box'],dtype=float)/2
    transform=np.asarray(observation['hand_T_box'],dtype=float)
    rotation=np.asarray(world_R_hand,dtype=float)
    if (half.shape!=(3,) or transform.shape!=(4,4) or rotation.shape!=(3,3)
            or not np.isfinite(np.r_[half,transform.ravel(),rotation.ravel()]).all()
            or (half<=0).any() or (half>.3).any()
            or not np.allclose(transform[3],[0,0,0,1])):
        raise ValueError('Invalid observed held geometry')
    for r in [rotation,transform[:3,:3]]:
        if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1):
            raise ValueError('Held geometry rotation is invalid')
    center=rotation@transform[:3,3]
    extent=np.abs(rotation@transform[:3,:3])@half
    offset=max(0.,float(extent[2]-center[2]))
    return dict(minimum_hand_clearance_m=offset,
                recommended_hand_clearance_m=offset+.03,
                source='Observed held envelope at the preserved grasp attitude')
