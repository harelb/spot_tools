import json
import struct
import numpy as np
import pytest
from spot_tools_ros.isaac_sensors import decode_rgbd


def packet(**overrides):
    meta=dict(protocol='spot-tools-isaac-v1',session_id='episode',height=2,width=3,
        frame_index=1,source='front_zed_color_image',timestamp_ns=123456,
        K=[[3,0,1.5],[0,3,1],[0,0,1]],position=[0,0,1],orientation_wxyz=[1,0,0,0],
        rgb_encoding='rgb8',depth_encoding='32FC1',depth_units='m',rgb_bytes=18,depth_bytes=24)
    meta.update(overrides);header=json.dumps(meta).encode()
    return struct.pack('<I',len(header))+header+bytes(range(18))+np.arange(6,dtype='<f4').tobytes()


def test_raw_snapshot_retains_pixel_and_depth_pairing():
    meta,rgb,depth=decode_rgbd(packet(),'episode')
    assert tuple(rgb[1,2])==(15,16,17) and depth[1,2]==5
    assert meta['timestamp_ns']==123456


@pytest.mark.parametrize('change',[dict(session_id='previous'),dict(depth_units='mm'),
    dict(rgb_bytes=2),dict(K=[[0,0,0],[0,0,0],[0,0,1]]),dict(orientation_wxyz=[0,0,0,0])])
def test_invalid_or_mismatched_snapshot_is_refused(change):
    with pytest.raises(ValueError):decode_rgbd(packet(**change),'episode')


def test_truncated_frame_is_refused():
    with pytest.raises(ValueError):decode_rgbd(packet()[:-1],'episode')
