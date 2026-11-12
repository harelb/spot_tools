"""Spot camera source names (body cameras vs. the arm's hand camera)."""

import pytest

spot_sensors = pytest.importorskip("spot_tools_ros.spot_sensors")


@pytest.mark.parametrize("name", ["frontleft", "frontright", "left", "right", "back"])
def test_body_camera_sources(name):
    d = spot_sensors.camera_defaults(name)
    assert f"{name}_{d['color_suffix']}" == f"{name}_fisheye_image"
    assert f"{name}_{d['depth_suffix']}" == f"{name}_depth_in_visual_frame"
    assert d["frame_id"] == ""  # the camera name


def test_hand_camera_sources():
    d = spot_sensors.camera_defaults("hand")
    assert f"hand_{d['color_suffix']}" == "hand_color_image"
    assert f"hand_{d['depth_suffix']}" == "hand_depth_in_hand_color_frame"
    assert d["frame_id"] == "hand_color_image_sensor"


def test_missing_sources_skips_hand_without_arm():
    available = {
        f"{c}_{s}"
        for c in ["frontleft", "frontright", "left", "right", "back"]
        for s in ["fisheye_image", "depth_in_visual_frame"]
    }
    hand = ["hand_color_image", "hand_depth_in_hand_color_frame"]
    assert spot_sensors.missing_sources(hand, available) == hand
    body = ["back_fisheye_image", "back_depth_in_visual_frame"]
    assert spot_sensors.missing_sources(body, available) == []


def test_active_cameras_only_with_subscribers_plus_always():
    names = ["frontleft", "frontright", "left", "right", "back", "hand"]
    subs = {"left": True, "back": False, "hand": True}
    active = spot_sensors.active_cameras(names, subs, spot_sensors.ALWAYS_REQUEST)
    assert active == ["frontleft", "frontright", "left", "hand"]


def test_active_cameras_empty_when_nothing_needed():
    names = ["left", "right"]
    assert spot_sensors.active_cameras(names, {}, []) == []
    # always-requested cameras the robot lacks (skipped at startup) never appear
    assert spot_sensors.active_cameras(names, {}, ["frontleft"]) == []
    assert spot_sensors.active_cameras(names, {"right": True}, []) == ["right"]
