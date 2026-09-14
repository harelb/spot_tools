import numpy as np
import pytest
from spot_tools_ros.robot_self_filter import mask_robot_depth


def observation():
    return dict(timestamp_ns=3,K=[[100.,0,3],[0,100.,3],[0,0,1]],
        position=[0,0,0],orientation_wxyz=[1,0,0,0],robot_self_filter=dict(timestamp_ns=3,
        boxes=[dict(low=[-.1,-.1,-.1],high=[.1,.1,.1],position=[0,0,1],orientation_wxyz=[1,0,0,0])]))


def test_only_measured_robot_returns_are_masked_without_inventing_background():
    meta=observation();depth=np.full((6,6),2.,np.float32);depth[2:4,2:4]=1.
    masked,count=mask_robot_depth(meta,depth)
    assert count==4 and np.all(masked[2:4,2:4]==0)
    assert np.all(masked[depth==2]==2) and np.all(depth[2:4,2:4]==1)


def test_stale_geometry_fails_and_offscreen_link_does_not_mask():
    meta=observation();meta['robot_self_filter']['timestamp_ns']=2
    with pytest.raises(ValueError,match='timestamp'):mask_robot_depth(meta,np.ones((6,6)))
    meta=observation();meta['robot_self_filter']['boxes'][0]['position']=[3,0,1]
    masked,count=mask_robot_depth(meta,np.ones((6,6)))
    assert count==0 and np.all(masked==1)


def test_missing_self_geometry_does_not_silently_map_the_robot():
    meta=observation();meta['robot_self_filter']['boxes']=[]
    with pytest.raises(ValueError,match='geometry'):mask_robot_depth(meta,np.ones((6,6)))


def test_empty_depth_roi_and_rotated_narrow_link():
    meta=observation()
    assert mask_robot_depth(meta,np.zeros((6,6)))[1]==0
    box=meta['robot_self_filter']['boxes'][0]
    box.update(low=[-.1,-.005,-.1],high=[.1,.005,.1],
               orientation_wxyz=[2**-.5,0,0,2**-.5])
    # A quarter turn makes the long axis vertical in the image.
    depth=np.ones((6,6));masked,count=mask_robot_depth(meta,depth,padding=0)
    assert count==6 and np.all(masked[:,3]==0)
    assert np.all(masked[:,0]==1)
