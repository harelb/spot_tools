"""Registered-depth pairing and request encoding for the approval panel. No robot needed."""

import types

import cv2
import numpy as np

from spot_skills.depth_utils import (
    aligned_depth_source,
    capture_aligned_depth,
    encode_depth_png,
    encode_jpeg,
)


def test_each_color_camera_maps_to_its_registered_depth_source():
    assert (
        aligned_depth_source("frontleft_fisheye_image")
        == "frontleft_depth_in_visual_frame"
    )
    assert aligned_depth_source("back_fisheye_image") == "back_depth_in_visual_frame"
    assert aligned_depth_source("hand_color_image") == "hand_depth_in_hand_color_frame"
    assert aligned_depth_source("hand_image") is None


def test_depth_png_is_lossless_and_jpeg_is_small():
    depth = (
        (np.arange(480 * 640, dtype=np.uint32) % 65536)
        .astype(np.uint16)
        .reshape(480, 640)
    )
    depth[0, 0], depth[1, 1] = 0, 65535
    restored = cv2.imdecode(
        np.frombuffer(encode_depth_png(depth), np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert restored.dtype == np.uint16 and np.array_equal(restored, depth)

    rng = np.random.default_rng(0)
    color = cv2.GaussianBlur(
        rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (9, 9), 0
    )
    jpeg = encode_jpeg(color)
    assert len(jpeg) < color.nbytes / 5
    assert (
        cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape
        == color.shape
    )


class FakeImageClient:
    def __init__(self, available, responses, fail=False):
        self.available, self.responses, self.fail = available, responses, fail
        self.requested = None

    def list_image_sources(self):
        return [types.SimpleNamespace(name=n) for n in self.available]

    def get_image_from_sources(self, names):
        if self.fail:
            raise RuntimeError("robot unreachable")
        self.requested = list(names)
        return [self.responses[n] for n in names]


def depth_response(name, rows=2, cols=3, scale=1000.0):
    from bosdyn.api import image_pb2

    data = np.arange(rows * cols, dtype="<u2").tobytes()
    image = types.SimpleNamespace(
        pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
        format=image_pb2.Image.FORMAT_RAW,
        rows=rows,
        cols=cols,
        data=data,
    )
    return types.SimpleNamespace(
        shot=types.SimpleNamespace(image=image),
        source=types.SimpleNamespace(name=name, depth_scale=scale),
    )


def test_capture_pairs_duplicates_skips_missing_and_never_raises(monkeypatch):
    import sys

    # Works with the real SDK or the stub installed by the grasp-loop tests.
    image_pb2 = sys.modules.get("bosdyn.api.image_pb2")
    if image_pb2 is None:
        from bosdyn.api import image_pb2
    from unittest import mock

    if isinstance(image_pb2, mock.MagicMock):  # only the stub needs real values
        monkeypatch.setattr(image_pb2.Image, "PIXEL_FORMAT_DEPTH_U16", 3)
        monkeypatch.setattr(image_pb2.Image, "FORMAT_RAW", 1)

    sources = [
        "frontleft_fisheye_image",
        "back_fisheye_image",
        "frontleft_fisheye_image",
        "hand_color_image",
    ]
    client = FakeImageClient(
        available=["frontleft_depth_in_visual_frame", "hand_depth_in_hand_color_frame"],
        responses={
            "frontleft_depth_in_visual_frame": depth_response(
                "frontleft_depth_in_visual_frame"
            ),
            "hand_depth_in_hand_color_frame": depth_response(
                "hand_depth_in_hand_color_frame", scale=0
            ),
        },
    )
    spot = types.SimpleNamespace(is_fake=False, image_client=client)
    result = capture_aligned_depth(spot, sources)
    assert client.requested == [
        "frontleft_depth_in_visual_frame",
        "hand_depth_in_hand_color_frame",
    ]
    assert result[1] is None, "back camera has no depth source on this robot"
    assert result[0][0].shape == (2, 3) and result[0][1] == 1000.0
    assert result[2] is result[0]
    assert result[3][1] == 1000.0, "zero depth_scale falls back to millimetres"

    messages = []
    broken = types.SimpleNamespace(
        is_fake=False, image_client=FakeImageClient(client.available, {}, fail=True)
    )
    assert capture_aligned_depth(broken, sources, log=messages.append) == [None] * 4
    assert messages and "robot unreachable" in messages[0]
    assert (
        capture_aligned_depth(types.SimpleNamespace(is_fake=True), sources)
        == [None] * 4
    )
