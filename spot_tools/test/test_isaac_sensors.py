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


def sensor_client(monkeypatch, responses):
    from types import SimpleNamespace
    from io import BytesIO
    from urllib.error import HTTPError
    import spot_tools_ros.isaac_sensors as module
    client=module.SensorClient.__new__(module.SensorClient)
    client.transport=SimpleNamespace(url='http://127.0.0.1:9250',session='episode',
                                    call=lambda method:{'paused':False})
    client.source='front_zed_color_image';client.mask_robot=False
    client.index=-1;client.stamp=-1;client.stale_since=None;client.stale_responses=0
    def respond(*args,**kwargs):
        value=responses.pop(0)
        if isinstance(value,Exception):raise value
        result=BytesIO(value);result.status=200
        result.headers={'Content-Length':str(len(value))}
        return result
    monkeypatch.setattr(module,'urlopen',respond)
    return client,HTTPError


def test_transient_stale_frame_never_republishes_or_advances_cursor(monkeypatch):
    client,error=sensor_client(monkeypatch,[])
    responses=[packet(),error('',409,'RGB-D is stale',{},None),packet(frame_index=2,timestamp_ns=123457)]
    client,_=sensor_client(monkeypatch,responses)
    assert client.next()[0]['frame_index']==1
    assert client.next() is None
    assert (client.index,client.stamp)==(1,123456)
    assert client.stale_responses==1
    assert client.next()[0]['frame_index']==2
    assert client.stale_since is None


def test_persistent_stale_sensor_expires_and_wrong_episode_still_fails(monkeypatch):
    import spot_tools_ros.isaac_sensors as module
    client,error=sensor_client(monkeypatch,[])
    responses=[error('',409,'RGB-D is stale',{},None) for _ in range(2)]
    client,_=sensor_client(monkeypatch,responses)
    clock=iter([1.,11.]);monkeypatch.setattr(module.time,'monotonic',lambda:next(clock))
    assert client.next() is None
    with pytest.raises(RuntimeError,match='recovery expired'):client.next()
    client,_=sensor_client(monkeypatch,[error('',409,'unknown endpoint or episode',{},None)])
    with pytest.raises(error):client.next()


def test_recovery_rejects_duplicate_observation(monkeypatch):
    client,error=sensor_client(monkeypatch,[])
    client,_=sensor_client(monkeypatch,[packet(),error('',409,'RGB-D is stale',{},None),packet()])
    client.next();assert client.next() is None
    with pytest.raises(ValueError,match='stale, reordered'):client.next()
