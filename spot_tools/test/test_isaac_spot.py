"""SDK/skill contract tests; these do not claim physical Isaac execution."""
from types import SimpleNamespace
import base64
import pytest
from bosdyn.api import robot_state_pb2, robot_command_pb2, arm_command_pb2, gripper_command_pb2
from spot_executor.isaac_spot import IsaacSpot, IsaacTransport


class RecordingTransport:
    def __init__(self, complete=True):
        self.commands = []
        self.complete = complete

    def call(self, method, **args):
        if method == 'robot_command':
            self.commands.append(robot_command_pb2.RobotCommand.FromString(base64.b64decode(args['request'])))
            return {'command_id': len(self.commands)}
        raise NotImplementedError(method)

    def proto(self, method, response_type, request=None, **args):
        if method == 'robot_state':
            state = robot_state_pb2.RobotState()
            state.manipulator_state.is_gripper_holding_item = True
            edge = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['hand']
            edge.parent_frame_name = 'body'
            edge.parent_tform_child.rotation.w = 1
            state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['body'].parent_tform_child.rotation.w = 1
            return state
        if method == 'robot_command_feedback':
            result = robot_command_pb2.RobotCommandFeedbackResponse()
            command = self.commands[args['command_id'] - 1].synchronized_command
            if command.HasField('gripper_command'):
                result.feedback.synchronized_feedback.gripper_command_feedback.claw_gripper_feedback.status = gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_AT_GOAL
                return result
            feedback = result.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
            feedback.status = (arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE
                               if self.complete else arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_STALLED)
            return result
        raise NotImplementedError(method)


def test_actual_place_skill_uses_sdk_commands_without_fake_branch(monkeypatch):
    from spot_skills.grasp_utils import place_at_point
    transport = RecordingTransport()
    spot = IsaacSpot(transport=transport)
    assert spot.is_fake is False
    assert place_at_point(spot, [.6, 0, .1], lambda: False)
    assert len(transport.commands) == 6  # approach, place, open, retreat, stow, close
    assert transport.commands[0].synchronized_command.arm_command.HasField('arm_cartesian_command')
    assert transport.commands[2].synchronized_command.HasField('gripper_command')


def test_failed_arm_motion_does_not_release():
    from spot_skills.grasp_utils import place_at_point
    transport = RecordingTransport(complete=False)
    with pytest.raises(RuntimeError, match='did not reach'):
        place_at_point(IsaacSpot(transport=transport), [.6, 0, .1], lambda: False)
    assert len(transport.commands) == 1
    assert not transport.commands[0].synchronized_command.HasField('gripper_command')


def test_cancelled_place_sends_no_motion():
    from spot_skills.grasp_utils import place_at_point
    transport = RecordingTransport()
    with pytest.raises(RuntimeError, match='cancelled'):
        place_at_point(IsaacSpot(transport=transport), [.6, 0, .1], lambda: True)
    assert transport.commands == []


def test_isaac_refuses_physical_or_ambiguous_endpoints():
    for endpoint in ['http://192.168.80.3', 'https://localhost', 'http://user:password@localhost']:
        with pytest.raises(ValueError):
            IsaacTransport(endpoint)


def test_unsupported_sdk_service_is_explicit():
    with pytest.raises(NotImplementedError):
        IsaacSpot(transport=RecordingTransport()).ensure_client('anything-unknown')


def test_failed_stow_is_not_success():
    from spot_skills.arm_utils import stow_arm
    with pytest.raises(RuntimeError, match='arm command'):
        stow_arm(IsaacSpot(transport=RecordingTransport(complete=False)))


def test_pick_camera_session_closes_on_failure():
    class CameraTransport:
        def __init__(self):self.events=[]
        def call(self, method, **args):
            self.events.append((method,args))
            if method=='health':return {'pick_camera_ready':True}
            return {'active':args.get('enabled',False)}
    transport=CameraTransport()
    with pytest.raises(RuntimeError,match='cancel'):
        with IsaacSpot(transport=transport).pick_camera_session():
            raise RuntimeError('cancel')
    starts=[args for method,args in transport.events if method=='pick_cameras']
    assert [e['enabled'] for e in starts]==[True,False]
    assert starts[0]['request_id']==starts[1]['request_id']


def test_failed_action_does_not_execute_dependent_action(monkeypatch):
    import spot_executor.spot_executor as module
    from robot_executor_interface.action_descriptions import ActionSequence,Follow
    import numpy as np
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    state=SimpleNamespace(robot=SimpleNamespace(time_sync=SimpleNamespace(wait_for_sync=lambda:None)),take_lease=lambda:None)
    executor=module.SpotExecutor(state,None,None,None)
    calls=[];results=[]
    executor.execute_follow=lambda command,feedback:calls.append(command) or False
    feedback=SimpleNamespace(break_out_of_waiting_loop=False,print=lambda *a:None,
        action_result=lambda command,index,status,detail:results.append((index,status)))
    first=Follow('vision',np.array([[0.,0.,0.],[1.,0.,0.]]))
    second=Follow('vision',np.array([[1.,0.,0.],[2.,0.,0.]]))
    executor.process_action_sequence(ActionSequence('plan','hamilton',[first,second]),feedback)
    assert len(calls)==3 and all(c is first for c in calls)
    assert results==[(0,'FAILED')]


def test_real_pick_action_scopes_auxiliary_cameras(monkeypatch):
    import spot_executor.spot_executor as module
    from robot_executor_interface.action_descriptions import Pick
    import numpy as np
    class CameraTransport:
        def __init__(self):
            self.active = False
            self.events = []
        def call(self, method, **args):
            if method == 'pick_cameras':
                self.active = args['enabled']
                self.events.append(self.active)
                return {'active': self.active}
            if method == 'health':
                return {'pick_camera_ready': self.active}
            raise NotImplementedError(method)
    transport = CameraTransport()
    spot = IsaacSpot(transport=transport)
    executor = module.SpotExecutor(spot, object(), None, None,
                                  pick_image_source='front_zed_color_image')
    def grasp(interface, detector, **kwargs):
        assert interface is spot and not interface.is_fake and transport.active
        assert kwargs['image_source'] == 'front_zed_color_image'
        raise RuntimeError('injected grasp failure')
    monkeypatch.setattr(module, 'object_grasp', grasp)
    feedback = SimpleNamespace(print=lambda *args: None)
    with pytest.raises(RuntimeError, match='injected grasp failure'):
        executor.execute_pick(Pick(frame='vision', object_class='mug', robot_point=np.zeros(3),
                                   object_point=np.ones(3), object_id='o1'), feedback)
    assert transport.events == [True, False]
    assert transport.active is False
