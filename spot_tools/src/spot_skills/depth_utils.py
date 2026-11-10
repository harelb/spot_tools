"""Depth registered to Spot's color cameras, for the operator approval panel.

Spot publishes, for each body fisheye camera, a depth image already projected
into that camera's pixel grid (``<camera>_depth_in_visual_frame``), and for the
gripper camera ``hand_depth_in_hand_color_frame``. Pairing each candidate color
image with its registered depth lets the operator see how far a pixel is before
choosing a grasp or walk-to point. Depth is a convenience: any failure here
returns ``None`` entries and never blocks the pick.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Spot's depth images are 16-bit; these raw values carry no measurement.
INVALID_DEPTH_VALUES = (0, 65535)

DepthImage = Tuple[np.ndarray, float]  # (uint16 HxW raw depth, raw units per meter)


def aligned_depth_source(color_source: str) -> Optional[str]:
    """Name of the depth source registered to `color_source`, if Spot has one."""
    if color_source == "hand_color_image":
        return "hand_depth_in_hand_color_frame"
    if color_source.endswith("_fisheye_image"):
        return color_source[: -len("_fisheye_image")] + "_depth_in_visual_frame"
    return None


def decode_depth_response(response) -> Optional[DepthImage]:
    """Turn a bosdyn ImageResponse holding DEPTH_U16 into (array, scale)."""
    from bosdyn.api import image_pb2

    image = response.shot.image
    if image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        return None
    if image.format == image_pb2.Image.FORMAT_RAW:
        depth = np.frombuffer(image.data, dtype="<u2").reshape(image.rows, image.cols)
    else:
        depth = cv2.imdecode(
            np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if depth is None or depth.dtype != np.uint16:
            return None
    scale = float(response.source.depth_scale) or 1000.0
    return np.ascontiguousarray(depth), scale


def capture_aligned_depth(
    spot, color_sources: Sequence[str], log=print
) -> List[Optional[DepthImage]]:
    """Fetch registered depth for each color source in one image request.

    Returns a list parallel to `color_sources`; entries are None when the robot
    is fake, has no matching depth source, or the request fails.
    """
    result: List[Optional[DepthImage]] = [None] * len(color_sources)
    if getattr(spot, "is_fake", False):
        return result
    try:
        available = {source.name for source in spot.image_client.list_image_sources()}
        wanted = {}
        for index, color in enumerate(color_sources):
            depth_source = aligned_depth_source(color)
            if depth_source in available:
                wanted.setdefault(depth_source, []).append(index)
        if not wanted:
            return result
        responses = spot.image_client.get_image_from_sources(list(wanted))
        for response in responses:
            decoded = decode_depth_response(response)
            for index in wanted.get(response.source.name, []):
                result[index] = decoded
    except Exception as ex:  # depth must never stop the grasp flow
        log(f"Depth capture for the approval panel failed: {ex}")
        return [None] * len(color_sources)
    return result


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


def encode_depth_png(depth: np.ndarray) -> bytes:
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Depth must be a 2D uint16 array")
    ok, buffer = cv2.imencode(".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buffer.tobytes()
