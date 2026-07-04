"""Tests for YOLODetector's configurable confidence threshold and class-synonym
translation (Task 6c). No real YOLOE weights are loaded -- `ultralytics.YOLOE`
is replaced with a lightweight fake that records how it was called.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spot_skills.detection_utils import YOLODetector


class FakeBox:
    """Stand-in for an ultralytics `Boxes` single-box view."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeResult:
    """Stand-in for a single ultralytics `Results` object."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLOEInnerModel:
    """Stand-in for `YOLOE(...).model`, whose `.names` and `.set_classes` are
    used directly by `set_up_detector`."""

    def __init__(self, names):
        self.names = names

    def set_classes(self, classes):
        self.names = list(classes)


class FakeYOLOE:
    """Stand-in for the top-level `ultralytics.YOLOE` object."""

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model = FakeYOLOEInnerModel(names=[])
        self.last_conf = "UNSET"
        self._results = []

    def set_classes(self, classes):
        self.model.names = list(classes)

    def set_results(self, results):
        self._results = results

    def __call__(self, model_input, conf=None):
        self.last_conf = conf
        return self._results


def make_detector(class_synonyms=None, conf=None):
    """Build a YOLODetector wired to a FakeYOLOE instead of a real model."""
    fake_yolo = FakeYOLOE("weights.pt")
    kwargs = {}
    if conf is not None:
        kwargs["conf"] = conf
    with patch("spot_skills.detection_utils.YOLOE", return_value=fake_yolo):
        detector = YOLODetector(
            spot=MagicMock(),
            yolo_world_path="weights.pt",
            class_synonyms=class_synonyms,
            **kwargs,
        )
    return detector, fake_yolo


def make_img():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# --- synonym translation -----------------------------------------------------


def test_synonyms_register_prompt_phrase_not_canonical_name():
    _, fake_yolo = make_detector(class_synonyms={"bag": "cement bag"})

    assert "cement bag" in fake_yolo.model.names
    assert "bag" not in fake_yolo.model.names
    # untouched classes still pass through verbatim
    assert "cone" in fake_yolo.model.names
    assert "pipe" in fake_yolo.model.names


def test_set_up_detector_translates_through_synonym_map():
    detector, fake_yolo = make_detector(class_synonyms={"bag": "cement bag"})

    # "cement bag" is already registered (from __init__'s seed list), so
    # asking to set up canonical "bag" should be a no-op, not a duplicate add.
    fake_yolo.model.set_classes = MagicMock(wraps=fake_yolo.model.set_classes)
    detector.set_up_detector("bag")
    fake_yolo.model.set_classes.assert_not_called()


def test_detection_under_translated_label_matches_canonical_query():
    detector, fake_yolo = make_detector(class_synonyms={"bag": "cement bag"})

    box = FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 20, 20])
    fake_yolo.set_results([FakeResult(boxes=[box], names={0: "cement bag"})])

    xy = detector._get_centroid(make_img(), "bag", rotate=0, debug=False)

    assert xy is not None


def test_missing_synonym_entry_does_not_match_translated_label():
    """Sanity check for the failure mode task 6b flagged: if the synonym map
    is absent, a model reporting the translated label must NOT silently
    match a canonical query -- it should read as "not found", not crash."""
    detector, fake_yolo = make_detector(class_synonyms=None)

    box = FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 20, 20])
    fake_yolo.set_results([FakeResult(boxes=[box], names={0: "cement bag"})])

    xy = detector._get_centroid(make_img(), "bag", rotate=0, debug=False)

    assert xy is None


# --- no-map behavior identical to today --------------------------------------


def test_no_synonym_map_registers_classes_verbatim():
    _, fake_yolo = make_detector(class_synonyms=None)

    assert fake_yolo.model.names == ["", "bag", "cone", "pipe"]


def test_no_synonym_map_set_up_detector_appends_verbatim():
    detector, fake_yolo = make_detector(class_synonyms=None)

    detector.set_up_detector("crate")

    assert "crate" in fake_yolo.model.names


def test_no_synonym_map_detection_matches_verbatim_label():
    detector, fake_yolo = make_detector(class_synonyms=None)

    box = FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 20, 20])
    fake_yolo.set_results([FakeResult(boxes=[box], names={0: "bag"})])

    xy = detector._get_centroid(make_img(), "bag", rotate=0, debug=False)

    assert xy is not None


# --- confidence threshold -----------------------------------------------------


def test_default_conf_matches_ultralytics_default():
    detector, fake_yolo = make_detector()

    assert detector.conf == 0.25

    fake_yolo.set_results([])
    detector._get_centroid(make_img(), "bag", rotate=0, debug=False)

    assert fake_yolo.last_conf == 0.25


def test_custom_conf_is_passed_through_to_predict_call():
    detector, fake_yolo = make_detector(conf=0.02)

    assert detector.conf == 0.02

    fake_yolo.set_results([])
    detector._get_centroid(make_img(), "bag", rotate=0, debug=False)

    assert fake_yolo.last_conf == 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
