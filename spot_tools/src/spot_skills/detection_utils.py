from copy import copy
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from bosdyn.api import image_pb2
from ultralytics import YOLOE


@dataclass
class DetectionCandidate:
    """A candidate image from a single camera source for object detection."""

    bosdyn_image: image_pb2.ImageResponse
    cv2_image: np.ndarray
    detection_xy: Optional[tuple[int, int]]


class Detector:
    def __init__(self, spot):
        self.spot = spot


class YOLODetector(Detector):
    def __init__(
        self,
        spot,
        yolo_world_path,
        conf: float = 0.25,
        class_synonyms: Optional[dict] = None,
        load_on_demand: bool = False,
    ):
        super().__init__(spot)

        print("Initializing YOLOWorld model. ")
        if not yolo_world_path:
            raise ValueError("YOLOWorld model path must be provided.")

        # `conf` is passed explicitly to every predict call below. 0.25 matches
        # the ultralytics library default, so leaving this unset preserves
        # today's production behavior exactly.
        self.conf = conf

        # Maps a canonical semantic class name (e.g. "bag", used everywhere else
        # in the stack) to the prompt phrase given to the detector's open-vocab
        # text encoder (e.g. "cement bag"). Classes without an entry are used
        # verbatim, so an empty/None map reproduces today's behavior exactly.
        # Prompt phrases are lowercase-normalized once here so that class
        # registration (set_up_detector lowercases appended classes) and the
        # detection match in _get_centroid operate in the same case space --
        # a mixed-case synonym value must not silently never match.
        self.class_synonyms = {
            key: value.lower() for key, value in (class_synonyms or {}).items()
        }

        self._model_path = yolo_world_path
        self._yolo_model = None
        import threading
        self._model_lock = threading.Lock()
        if not load_on_demand:
            self.yolo_model

    @property
    def yolo_model(self):
        with self._model_lock:
            if self._yolo_model is None:
                model = YOLOE(self._model_path)
                model.set_classes([self._to_prompt(cls) for cls in ["", "bag", "cone", "pipe"]])
                self._yolo_model = model
            return self._yolo_model

    def _to_prompt(self, semantic_class):
        """Translate a canonical semantic class name to the prompt phrase
        registered with / reported by the detector model."""
        return self.class_synonyms.get(semantic_class, semantic_class)

    def set_up_detector(self, semantic_class):
        prompt_class = self._to_prompt(semantic_class)

        # Get list of current classes recognized by YOLO World model
        recognized_classes = self.yolo_model.model.names

        # If the classes were never set by the users, the recognized classes (model.model.names)
        # will be a dictionary. Otherwise, it'll be a list.
        if isinstance(recognized_classes, dict):
            recognized_classes = list(recognized_classes.values())

        # Check if the class exists in the list (lowercase for consistency)
        if prompt_class.lower() not in [cls.lower() for cls in recognized_classes]:
            updated_classes = recognized_classes + [prompt_class.lower()]
            self.yolo_model.set_classes(updated_classes)
            print(f"Updated recognized classes: {updated_classes}")

    def return_centroid(self, img_source, semantic_class, debug):
        detection_candidates = []  # List of DetectionCandidates -- one per camera source on Spot
        detection_index = None

        # Get the primary image source & run the detector; Add to detection_candidates
        primary_bosdyn_image, primary_cv2_image = self.spot.get_image_RGB(
            view=img_source
        )
        primary_xy = self._get_centroid(
            primary_cv2_image, semantic_class, rotate=0, debug=debug
        )
        detection_candidates.append(
            DetectionCandidate(primary_bosdyn_image, primary_cv2_image, primary_xy)
        )

        # If primary image has a detection, just get the images from the other image sources; Otherwise, look for detections
        if primary_xy is not None:
            detection_index = 0
            secondary_candidates = self._get_candidate_detection_images()
        else:
            secondary_candidates, secondary_detection_index = self._look_for_object(
                semantic_class, debug=debug
            )
            if secondary_detection_index is not None:
                detection_index = (
                    secondary_detection_index + 1
                )  # offset of the primary candidate
        # Collect all of the detection candidates
        detection_candidates.extend(secondary_candidates)
        return detection_index, detection_candidates

    def _get_centroid(self, img, semantic_class, rotate, debug):
        if rotate == 0:
            model_input = copy(img)
        elif rotate == 1:
            model_input = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 2:
            model_input = cv2.rotate(img, cv2.ROTATE_180)

        results = self.yolo_model(model_input, conf=self.conf)

        best_box = None
        best_confidence = -1.0
        # Compare in lowercase space, matching how classes are registered
        # (synonym values are lowered in __init__; set_up_detector lowercases
        # appended classes).
        prompt_class = self._to_prompt(semantic_class).lower()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = float(box.conf[0])

                # Check if the class name matches the semantic class we're looking for and if box is not too big
                box_height = (
                    box.xyxy[0][3] - box.xyxy[0][1]
                )  # height of the bounding box
                if box_height > 0.95 * img.shape[0]:
                    continue
                box_width = box.xyxy[0][2] - box.xyxy[0][0]  # width of the bounding box
                if (
                    box_width > 0.95 * img.shape[1]
                ):  # If the box is more than 95% the width of the image, skip it
                    continue

                if class_name.lower() == prompt_class and confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box

        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            class_id = int(best_box.cls[0])

            mid_x = x1 + (x2 - x1) // 2
            mid_y = y1 + (y2 - y1) // 2

            if rotate == 0:
                centroid_x = mid_x
                centroid_y = mid_y

            elif rotate == 1:
                centroid_x = mid_y
                centroid_y = img.shape[0] - mid_x

            elif rotate == 2:
                centroid_x = img.shape[1] - mid_x
                centroid_y = img.shape[0] - mid_y

            print("The centroid of the bounding box is at:", centroid_x, centroid_y)

            return (centroid_x, centroid_y)  # Return the centroid of the bounding box

        else:
            return None

    def _get_candidate_detection_images(self):
        detection_candidates = []
        sources = self.spot.image_client.list_image_sources()
        sources = [
            s for s in sources if "depth" not in s.name and "hand_image" != s.name
        ]
        for source in sources:
            bosdyn_image, cv2_image = self.spot.get_image_RGB(view=source.name)
            detection_candidates.append(
                DetectionCandidate(bosdyn_image, cv2_image, None)
            )
        return detection_candidates

    def _look_for_object(self, semantic_class, debug):
        sources = self.spot.image_client.list_image_sources()

        candidates = []
        detection_index = None

        for source in sources:
            if (
                "depth" in source.name or source.name == "hand_image"
            ):  # "hand_image" is only in greyscale, "hand_color_image" is RGB
                continue

            image_source = source.name
            print("Getting image from source:", image_source)
            image, img = self.spot.get_image_RGB(view=image_source)

            rotate = 0

            if (
                "frontleft_fisheye_image" in image_source
                or "frontright_fisheye_image" in image_source
            ):
                rotate = 1  # cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            elif "right_fisheye_image" in image_source:
                rotate = 2  # cv2.rotate(img, cv2.ROTATE_180)

            xy = self._get_centroid(img, semantic_class, rotate=rotate, debug=debug)
            print("Found object centroid:", xy)
            if xy is None:
                print(f"Object not found in {image_source}.")

            candidates.append(DetectionCandidate(image, img, xy))
            if xy is not None and detection_index is None:
                detection_index = len(candidates) - 1

        return candidates, detection_index


class SemanticDetector(Detector):
    def __init__(self, spot, labelspace_map):
        super().__init__(spot)

        if self.spot.semantic_name_to_id is None:
            raise Exception(
                "Semantic names must be mapped to their id's (semantic_name_to_id is undefined)."
            )

        self.semantic_ids_to_grab = []

    def set_up_detector(self, semantic_class):
        if self.spot.labelspace_map is not None:
            self.semantic_ids_to_grab = (
                self.spot.labelspace_map[semantic_class]
                + self.spot.labelspace_map["clothes"]
                + self.spot.labelspace_map["bag"]
            )
        else:
            self.semantic_ids_to_grab = [
                self.spot.semantic_name_to_id[semantic_class],
                self.spot.semantic_name_to_id["bag"],
                self.spot.semantic_name_to_id["clothes"],
            ]

    def return_centroid(self, img_source, semantic_class, debug):
        image, img = self.spot.get_image_alt(view=img_source)
        semantic_image = self.spot.segment_image(img)

        # Convert to grayscale if needed
        if len(semantic_image.shape) == 3:
            gray = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = semantic_image

        # Create a binary mask where the class of interest is white and the rest is black
        mask = np.zeros_like(gray)
        # mask[gray == class_index] = 255
        mask[np.isin(gray, self.semantic_ids_to_grab)] = 255

        xy = self._get_centroid(semantic_image, self.semantic_ids_to_grab, img)

        if xy is None:
            print("Object not found in first image. Looking around!")
            xy, image, img, image_source = self._look_for_object(
                self.semantic_ids_to_grab
            )
            if xy is None:
                print("Object not found near robot.")

        return xy, image, img

    def _get_centroid(self, segmented_image, class_indices, image):
        """Get the centroid of a class in a segmented image."""

        # Convert to grayscale if needed
        if len(segmented_image.shape) == 3:
            gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = segmented_image

        # Create a binary mask where the class of interest is white and the rest is black
        mask = np.zeros_like(gray)
        # mask[gray == class_index] = 255
        mask[np.isin(gray, class_indices)] = 255

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # No contour found for the class

        # Assuming we take the largest contour if multiple are found
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments for the largest contour
        M = cv2.moments(largest_contour)

        # Calculate centroid using moments
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            seg_image_size = segmented_image.shape
            image_size = image.shape
            x_scale = image_size[1] / seg_image_size[1]
            y_scale = image_size[0] / seg_image_size[0]

            return (cX * x_scale, cY * y_scale)
        else:
            return None  # Centroid calculation failed

    def _look_for_object(self, semantic_ids):
        """Look for an object in the image sources. Return the centroid of the object, and the image source."""

        sources = self.spot.image_client.list_image_sources()

        for source in sources:
            if "depth" in source.name:
                continue
            image_source = source.name
            image, img = self.spot.get_image_alt(view=image_source)

            rotate = 0
            if "front" in image_source or "hand_image" in image_source:
                rotate = 1
            elif "right_fisheye_image" in image_source:
                rotate = 2

            semantic_image = self.spot.segment_image(img, rotate=rotate, show=False)
            xy = self.get_class_centroid(semantic_image, semantic_ids, img)
            print("Found object centroid:", xy)
            if xy is None:
                print(f"Object not found in {image_source}.")
                continue
            else:
                return xy, image, img, image_source

        return None, None, None, None
