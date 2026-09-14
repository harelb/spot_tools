from types import SimpleNamespace
import pytest
from spot_skills.placement_verification import observed_candidates,PlacementVerifier


def response(position=(1.,2.,3.)):
    source=dict(kind='live',episode_id='episode',timestamp_ns='101',calibration_sha256='a'*64)
    return dict(camera_model={'live_observation':dict(source,timestamp_ns=101,received_at=11.)},
                detections=[dict(observation_source=source,position=position,confidence=.9)])


def check(value):
    return observed_candidates(value,episode='episode',after_ns=100,target=[1.,2.,3.],requested_at=10.)


def test_verification_accepts_calibrated_new_capture_only():
    assert len(check(response())[0])==1
    assert check(response((4.,2.,3.)))[0]==[]
    for key,value in [('kind','prior'),('episode_id','old'),('timestamp_ns',100),
                      ('received_at',9.),('received_at',float('nan')),('calibration_sha256','')]:
        r=response();r['camera_model']['live_observation'][key]=value
        with pytest.raises(RuntimeError,match='stale'):check(r)
    r=response();r['detections'][0]['observation_source']['episode_id']='other'
    with pytest.raises(RuntimeError,match='match'):check(r)


def test_empty_gripper_alone_does_not_pass_and_duplicate_target_is_ambiguous():
    verifier=PlacementVerifier(None,None,'episode','front')
    verifier.state=lambda:(SimpleNamespace(manipulator_state=SimpleNamespace(is_gripper_holding_item=False)),100,None)
    context=dict(object_class='mug',target=[1.,2.,3.],before={})
    for matches in ([],[{},{}]):
        verifier.query=lambda *a:(matches,{})
        with pytest.raises(RuntimeError,match='uniquely'):verifier.verify(context)
    verifier.query=lambda *a:([{'position':[1.,2.,3.]}],{'timestamp_ns':101})
    assert verifier.verify(context)['verified']


def test_existing_target_prevents_ambiguous_release():
    verifier=PlacementVerifier(None,None,'episode','front')
    pose=SimpleNamespace(transform_point=lambda *p:p)
    verifier.state=lambda:(SimpleNamespace(manipulator_state=SimpleNamespace(is_gripper_holding_item=True)),100,pose)
    verifier.query=lambda *a:([{}],{})
    with pytest.raises(RuntimeError,match='already contains'):verifier.prepare('mug',[1.,2.,3.])
