"""The operator can answer a grasp request with "walk to" any number of times.

Runs anywhere: the Boston Dynamics SDK, OpenCV, shapely and ultralytics are
replaced by stubs before spot_skills is imported, and the walking/stowing
primitives are patched, so only the approval loop in object_grasp is exercised.
"""

import sys
import types
from unittest import mock

import numpy as np
import pytest

STUBBED_MODULES = [
    "bosdyn",
    "bosdyn.api",
    "bosdyn.api.geometry_pb2",
    "bosdyn.api.image_pb2",
    "bosdyn.api.manipulation_api_pb2",
    "bosdyn.api.robot_state_pb2",
    "bosdyn.api.basic_command_pb2",
    "bosdyn.api.arm_command_pb2",
    "bosdyn.api.synchronized_command_pb2",
    "bosdyn.api.robot_command_pb2",
    "bosdyn.api.spot",
    "bosdyn.api.spot.robot_command_pb2",
    "bosdyn.client",
    "bosdyn.client.estop",
    "bosdyn.client.lease",
    "bosdyn.client.util",
    "bosdyn.client.frame_helpers",
    "bosdyn.client.math_helpers",
    "bosdyn.client.robot_command",
    "bosdyn.client.image",
    "bosdyn.client.manipulation_api_client",
    "bosdyn.client.robot_state",
    "cv2",
    "shapely",
    "shapely.geometry",
    "ultralytics",
]


def _install_stubs():
    for name in STUBBED_MODULES:
        if name not in sys.modules or isinstance(sys.modules[name], mock.MagicMock):
            module = mock.MagicMock(name=name)
            module.__spec__ = types.SimpleNamespace(name=name)
            sys.modules[name] = module
    # `from bosdyn.api import manipulation_api_pb2` reads the attribute of the
    # parent module, so every parent must point at the one registered child.
    for name in STUBBED_MODULES:
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, sys.modules[name])
    sys.modules["bosdyn.client.frame_helpers"].math_helpers = sys.modules[
        "bosdyn.client.math_helpers"
    ]


_install_stubs()

from spot_skills import grasp_utils  # noqa: E402
from spot_skills.approval import ApprovalDecision  # noqa: E402
from spot_skills.detection_utils import DetectionCandidate, Detector  # noqa: E402


class ScriptedDetector(Detector):
    """Returns a fresh candidate set per capture; counts captures."""

    def __init__(self):
        super().__init__(spot=None)
        self.captures = 0
        self.set_up_classes = []
        self.images = []

    def set_up_detector(self, semantic_class):
        self.set_up_classes.append(semantic_class)

    def return_centroid(self, img_source, semantic_class, debug):
        self.captures += 1
        image = mock.MagicMock(name=f"bosdyn_image_{self.captures}")
        self.images.append(image)
        candidates = [
            DetectionCandidate(
                image, np.zeros((4, 4, 3), np.uint8), (10, 20), img_source
            ),
            DetectionCandidate(
                mock.MagicMock(name=f"other_{self.captures}"),
                np.zeros((4, 4, 3), np.uint8),
                None,
                "frontright_fisheye_image",
            ),
        ]
        return 0, candidates


class ScriptedFeedback:
    """Hands out pre-decided operator answers, one per request."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.messages = []

    def bounding_box_detection_feedback(
        self, images, source_names, detection_index, det_x, det_y, semantic_class
    ):
        self.requests.append((len(images), source_names, detection_index, det_x, det_y))
        return self.decisions.pop(0)

    def print(self, level, *message):
        self.messages.append(" ".join(str(m) for m in message))


class FakeManipulationClient:
    def __init__(self):
        self.requests = []

    def manipulation_api_command(self, manipulation_api_request):
        self.requests.append(manipulation_api_request)
        return types.SimpleNamespace(manipulation_cmd_id=7)

    def manipulation_api_feedback_command(self, manipulation_api_feedback_request):
        manipulation_api_pb2 = sys.modules["bosdyn.api.manipulation_api_pb2"]
        return types.SimpleNamespace(
            current_state=manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
        )


def fake_spot():
    spot = types.SimpleNamespace()
    spot.is_fake = False
    spot.state_client = mock.MagicMock(name="state_client")
    spot.manipulation_api_client = FakeManipulationClient()
    spot.command_client = mock.MagicMock(name="command_client")
    spot.robot = mock.MagicMock(name="robot")
    return spot


@pytest.fixture
def patched(monkeypatch):
    calls = {"walk": [], "stow": 0, "force_stow": 0}
    monkeypatch.setattr(
        grasp_utils,
        "stow_arm",
        lambda spot: calls.__setitem__("stow", calls["stow"] + 1),
    )
    monkeypatch.setattr(
        grasp_utils,
        "force_stow_arm",
        lambda *a, **k: calls.__setitem__("force_stow", calls["force_stow"] + 1),
    )
    monkeypatch.setattr(grasp_utils, "execute_recovery_action", mock.MagicMock())
    monkeypatch.setattr(grasp_utils.time, "sleep", lambda *_: None)

    def walk_to_pixel(spot, image, pixel, stand_pixel=None, standoff=1.0, **kwargs):
        calls["walk"].append((image, pixel, stand_pixel, standoff))
        return True

    monkeypatch.setattr(grasp_utils, "walk_to_pixel", walk_to_pixel)
    return calls


def test_walk_to_repeats_detection_until_approval(patched):
    spot = fake_spot()
    detector = ScriptedDetector()
    feedback = ScriptedFeedback(
        [
            ApprovalDecision.walk((100, 200), 0),
            ApprovalDecision.walk((5, 6), 1, stand_pixel=(7, 8)),
            ApprovalDecision.approve((11, 12), 0),
        ]
    )

    success = grasp_utils.object_grasp(
        spot,
        detector,
        image_source="frontleft_fisheye_image",
        semantic_class="cone",
        feedback=feedback,
        carry_high=False,
        walk_to_standoff=0.8,
    )

    assert success is True
    assert detector.captures == 3, "new images after every walk"
    assert len(feedback.requests) == 3
    assert [pixel for _, pixel, _, _ in patched["walk"]] == [(100, 200), (5, 6)]
    assert patched["walk"][1][2] == (7, 8), "stand point passed through"
    assert all(standoff == 0.8 for *_, standoff in patched["walk"])
    assert patched["walk"][1][0]._mock_name == "other_2", (
        "walk uses the chosen candidate image"
    )
    assert patched["stow"] >= 2, "arm stowed before each walk"
    # Exactly one grasp, at the approved pixel in the approved (third) capture.
    assert len(spot.manipulation_api_client.requests) == 1
    geometry_pb2 = sys.modules["bosdyn.api.geometry_pb2"]
    geometry_pb2.Vec2.assert_called_once_with(x=11, y=12)
    grasp = sys.modules["bosdyn.api.manipulation_api_pb2"].PickObjectInImage
    approved_image = detector.images[2]
    assert grasp.call_args.kwargs["camera_model"] is approved_image.source.pinhole
    assert (
        grasp.call_args.kwargs["transforms_snapshot_for_camera"]
        is approved_image.shot.transforms_snapshot
    )


def test_reject_after_walk_abandons_without_grasping(patched):
    spot = fake_spot()
    detector = ScriptedDetector()
    feedback = ScriptedFeedback(
        [ApprovalDecision.walk((1, 2), 0), ApprovalDecision.reject()]
    )

    assert grasp_utils.object_grasp(spot, detector, feedback=feedback) is False
    assert detector.captures == 2
    assert len(patched["walk"]) == 1
    assert spot.manipulation_api_client.requests == []


def test_preemption_during_walk_stops_the_attempt(patched):
    spot = fake_spot()
    detector = ScriptedDetector()
    feedback = ScriptedFeedback([ApprovalDecision.walk((1, 2), 0)])
    stopped = {"value": False}

    def walk(*args, **kwargs):
        stopped["value"] = True
        return False

    grasp_utils.walk_to_pixel = walk
    result = grasp_utils.object_grasp(
        spot, detector, feedback=feedback, should_stop=lambda: stopped["value"]
    )
    assert result is False
    assert detector.captures == 1
    assert spot.manipulation_api_client.requests == []


def test_legacy_tuple_feedback_still_works(patched):
    spot = fake_spot()
    detector = ScriptedDetector()

    class LegacyFeedback(ScriptedFeedback):
        def bounding_box_detection_feedback(self, *args):
            return (True, (3, 4), 0)

    assert grasp_utils.object_grasp(spot, detector, feedback=LegacyFeedback([])) is True
    sys.modules["bosdyn.api.geometry_pb2"].Vec2.assert_called_with(x=3, y=4)
