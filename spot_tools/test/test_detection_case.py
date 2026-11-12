"""YOLODetector matches detected classes case-insensitively and registers new
classes through YOLOE.set_classes (which computes their text embeddings).
No real YOLOE weights are loaded: ultralytics.YOLOE is replaced by a fake."""

from unittest.mock import MagicMock, patch

import numpy as np

from spot_skills.detection_utils import YOLODetector


class FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLOE:
    def __init__(self, weights_path):
        self.model = MagicMock()
        self.model.names = []
        self.set_classes_calls = []
        self._results = []

    def set_classes(self, classes):
        self.set_classes_calls.append(list(classes))
        self.model.names = list(classes)

    def __call__(self, model_input, **kwargs):
        return self._results


def make_detector():
    fake = FakeYOLOE("weights.pt")
    with patch("spot_skills.detection_utils.YOLOE", return_value=fake):
        detector = YOLODetector(spot=MagicMock(), yolo_world_path="weights.pt")
    return detector, fake


def test_mixed_case_query_matches_lowercase_detection():
    detector, fake = make_detector()
    box = FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 20, 20])
    fake._results = [FakeResult(boxes=[box], names={0: "bag"})]
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector._get_centroid(img, "Bag", rotate=0, debug=False) is not None


def test_new_class_is_registered_through_yoloe_set_classes():
    detector, fake = make_detector()
    fake.model.set_classes = MagicMock()
    detector.set_up_detector("Crate")
    assert fake.set_classes_calls[-1][-1] == "crate"
    fake.model.set_classes.assert_not_called()


def test_known_class_is_not_registered_again():
    detector, fake = make_detector()
    calls = len(fake.set_classes_calls)
    detector.set_up_detector("BAG")
    assert len(fake.set_classes_calls) == calls
