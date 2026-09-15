from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from spot_executor.manual_control import ManualControl


def setup():
    now=[10.]
    spot=Mock()
    ex=SimpleNamespace(spot_interface=spot,processing_action_sequence=False,keep_going=True)
    control=ManualControl(ex,SimpleNamespace(break_out_of_waiting_loop=False),run_id='r',episode_id='e',clock=lambda:now[0],monotonic=lambda:now[0],start_watchdog=False)
    request=dict(run_id='r',episode_id='e',mode='manual')
    state=control.claim(request)
    request.update(control_token=state['control_token'],sequence=1,issued_at=10.,vx=.1,vy=0.,wz=0.)
    return control,request,spot,now


def test_deadman_stops_and_requires_new_claim():
    c,r,spot,now=setup();c.drive(r)
    spot.set_twist.assert_called_once_with(.1,0.,0.)
    now[0]=10.36;c.tick()
    assert c.token is None
    assert spot.command_client.robot_command.call_count==2
    assert 'input expired' in c.state()['error']
    with pytest.raises(ValueError,match='input expired'):c.drive(dict(r,sequence=2,issued_at=10.36))
    assert c.claim(r)['error'] is None


def test_wall_clock_adjustment_cannot_extend_driving():
    c,r,spot,now=setup()
    c.clock=lambda:10.
    c.drive(r)
    c.clock=lambda:-100.
    now[0]=10.36;c.tick()
    assert c.token is None and not c.state()['driving']


def test_reordered_stale_or_wrong_episode_cannot_move():
    c,r,spot,now=setup();c.drive(r)
    with pytest.raises(ValueError,match='Duplicate'):c.drive(r)
    with pytest.raises(ValueError,match='Stale'):c.drive(dict(r,episode_id='old',sequence=2))
    now[0]=11.
    with pytest.raises(ValueError,match='Expired'):c.drive(dict(r,sequence=2))
    assert spot.set_twist.call_count==1


def test_handoff_waits_for_skill_acknowledgement():
    c,r,spot,now=setup();c.executor.processing_action_sequence=True
    with pytest.raises(ValueError,match='Stopping'):c.claim(dict(r,mode='planned'))
    assert c.token is None and not c.executor.keep_going
    c.executor.processing_action_sequence=False
    assert c.claim(dict(r,mode='planned'))['mode']=='planned'
    with pytest.raises(ValueError,match='Acquire'):c.drive(r)


def test_unready_or_failed_hardware_never_accepts_drive():
    c,r,spot,now=setup();c.ready=lambda:False
    with pytest.raises(ValueError,match='Fresh'):c.drive(r)
    spot.set_twist.assert_not_called()
    c,r,spot,now=setup();spot.set_twist.side_effect=RuntimeError('obstructed')
    with pytest.raises(RuntimeError,match='obstructed'):c.drive(r)
    assert c.token is None
    assert c.state()['error']=='obstructed'
    with pytest.raises(ValueError,match='obstructed'):c.drive(dict(r,sequence=2))


def test_compute_holds_motion_and_never_reuses_old_lease():
    c,r,spot,now=setup();c.drive(r)
    reserved=c.reserve_compute(r)
    assert c.state()['compute_active'] and c.token is None
    with pytest.raises(ValueError,match='held'):c.claim(r)
    with pytest.raises(ValueError,match='Acquire'):c.drive(dict(r,sequence=2))
    with pytest.raises(ValueError,match='Stale'):c.release_compute(dict(r,compute_token='old'))
    c.release_compute(dict(r,**reserved))
    assert not c.state()['compute_active'] and c.token is None
    with pytest.raises(ValueError,match='Acquire'):c.drive(dict(r,sequence=2))
    c.claim(r)
    c.reserve_compute(r)
    now[0]+=126;c.tick()
    assert not c.state()['compute_active'] and c.token is None
