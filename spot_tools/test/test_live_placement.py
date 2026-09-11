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
