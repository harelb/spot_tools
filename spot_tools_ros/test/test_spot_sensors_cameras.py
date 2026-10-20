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
