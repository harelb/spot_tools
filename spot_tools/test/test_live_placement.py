"""Test release ordering without issuing SDK commands to a robot."""

from types import SimpleNamespace

import pytest

from spot_skills import grasp_utils


def rig(monkeypatch, arrived=True):
    import bosdyn.client.frame_helpers

    calls = []
    monkeypatch.setattr(
        bosdyn.client.frame_helpers,
        "get_a_tform_b",
        lambda *a: SimpleNamespace(rot=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)),
    )
    monkeypatch.setattr(grasp_utils, "block_until_arm_arrives", lambda *a: arrived)
    monkeypatch.setattr(grasp_utils, "open_gripper", lambda *a: calls.append("release"))
    monkeypatch.setattr(grasp_utils, "stow_arm", lambda *a: calls.append("stow"))
    monkeypatch.setattr(grasp_utils, "close_gripper", lambda *a: calls.append("close"))
    state = SimpleNamespace(
        manipulator_state=SimpleNamespace(is_gripper_holding_item=True),
        kinematic_state=SimpleNamespace(transforms_snapshot=None),
    )
    spot = SimpleNamespace(
        is_fake=False,
        state_client=SimpleNamespace(get_robot_state=lambda: state),
        command_client=SimpleNamespace(
            robot_command=lambda cmd: calls.append("move") or 1
        ),
    )
    return spot, calls


def test_failed_positioning_does_not_release(monkeypatch):
    spot, calls = rig(monkeypatch, arrived=False)
    with pytest.raises(RuntimeError, match="did not reach"):
        grasp_utils.place_at_point(spot, [0.5, 0, 0.2], lambda: False)
    assert calls == ["move"]


def test_cancel_before_release_holds_object(monkeypatch):
    spot, calls = rig(monkeypatch)
    with pytest.raises(RuntimeError, match="cancelled"):
        grasp_utils.place_at_point(spot, [0.5, 0, 0.2], lambda: len(calls) >= 2)
    assert "release" not in calls


def test_place_approaches_releases_and_retracts(monkeypatch):
    spot, calls = rig(monkeypatch)
    assert grasp_utils.place_at_point(spot, [0.5, 0, 0.2], lambda: False)
    assert calls == ["move", "move", "release", "move", "stow", "close"]


def test_place_restores_observed_grasp_attitude_after_base_turn(monkeypatch):
    import bosdyn.client.frame_helpers as frames
    from bosdyn.client.math_helpers import Quat
    spot,calls=rig(monkeypatch)
    spot.preserve_grasp_placement_orientation=True
    spot.placement_grasp_orientation=Quat.from_pitch(.7)
    body=Quat.from_yaw(.5)
    desired=body.inverse()*spot.placement_grasp_orientation
    monkeypatch.setattr(frames,'get_a_tform_b',lambda snapshot,a,b:
        SimpleNamespace(rot=body if b==frames.BODY_FRAME_NAME else Quat()))
    commands=[]
    spot.command_client.robot_command=lambda command:commands.append(command) or 1
    assert grasp_utils.place_at_point(spot,[.5,0,.2],lambda:False)
    actual=commands[0].synchronized_command.arm_command.arm_cartesian_command.pose_trajectory_in_task.points[0].pose.rotation
    assert [actual.w,actual.x,actual.y,actual.z]==pytest.approx([desired.w,desired.x,desired.y,desired.z])
    assert spot.placement_grasp_orientation is None


def test_required_missing_grasp_attitude_fails_before_motion(monkeypatch):
    spot,calls=rig(monkeypatch);spot.preserve_grasp_placement_orientation=True
    with pytest.raises(RuntimeError,match='observed grasp orientation'):
        grasp_utils.place_at_point(spot,[.5,0,.2],lambda:False)
    assert not calls
