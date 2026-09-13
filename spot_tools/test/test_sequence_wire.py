import copy
import numpy as np
import pytest
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze, Pick, Place
from robot_executor_interface.sequence_wire import encode_sequence, decode_sequence, sequence_digest


def sequence():
    return ActionSequence('compiled-plan', 'hamilton', [
        Follow('map', np.array([[0.,0.,0.],[1.,2.,.3]])),
        Gaze('map', np.array([1.,2.,0.]), np.array([2.,3.,.7]), 'o1', True),
        Pick('map', 'mug', np.array([1.,2.,0.]), np.array([2.,3.,.7]), 'o1'),
        Place('map', 'mug', np.array([2.,2.,0.]), np.array([3.,3.,.8]), 'o1')])


def test_lossless_compiled_transport_and_ros_serializer():
    from robot_executor_interface_ros.action_descriptions_ros import to_msg, from_msg
    original = sequence()
    payload = encode_sequence(original)
    assert encode_sequence(decode_sequence(payload)) == payload
    roundtrip = encode_sequence(from_msg(to_msg(decode_sequence(payload))))
    np.testing.assert_allclose(roundtrip['actions'][0].pop('path2d'),
                               payload['actions'][0]['path2d'], atol=1e-12)
    expected = copy.deepcopy(payload)
    expected['actions'][0].pop('path2d')
    assert roundtrip == expected
    altered = copy.deepcopy(payload)
    altered['actions'][1]['stow_after'] = False
    assert sequence_digest(altered) != sequence_digest(payload)


@pytest.mark.parametrize('change', [
    lambda x: x['actions'][0]['path2d'][0].__setitem__(0, float('nan')),
    lambda x: x['actions'][2].pop('object_point'),
    lambda x: x['actions'][1].__setitem__('stow_after', 'false'),
    lambda x: x['actions'][0].__setitem__('kind', 'TELEPORT'),
])
def test_invalid_compiled_sequence_rejected(change):
    payload = encode_sequence(sequence())
    change(payload)
    with pytest.raises((ValueError, TypeError)):
        decode_sequence(payload)


def test_display_projection_cannot_disagree_with_execution():
    from open_set_navigation.live.contracts import Command, compiled_actions
    payload = encode_sequence(sequence())
    values = dict(robot_id='hamilton', mapping_session_id='session', frame='map',
                  position=(0.,0.,0.), scene_sha256='scene',
                  actions=compiled_actions(payload), compiled_sequence=payload)
    c = Command(**values)
    assert c.execution_plan_id == 'compiled-plan'
    values['actions'][2].target = (8.,9.,0.)
    with pytest.raises(ValueError, match='display actions differ'):
        Command(**values)
