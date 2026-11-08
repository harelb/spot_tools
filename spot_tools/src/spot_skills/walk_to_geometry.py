"""Pure geometry for walking toward a clicked image point.

Kept free of bosdyn imports so it can be unit tested anywhere. Frames: every
3D quantity is expressed in one fixed world frame (Spot's "vision" frame in
practice); yaw is the body heading in that frame.
"""

from typing import Optional, Tuple

import numpy as np


def pixel_ray(pixel, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Unit direction, in the camera frame, of the ray through `pixel`.

    Pinhole convention: +z forward, +x right, +y down (as in the Spot API).
    """
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal length must be positive")
    u, v = pixel
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return d / np.linalg.norm(d)


def intersect_ray_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    min_range: float = 0.2,
    max_range: float = 15.0,
) -> Optional[np.ndarray]:
    """Where the ray meets the plane, or None if it misses or is implausible.

    A ray parallel to the plane, pointing away from it, or hitting it closer
    than `min_range` / farther than `max_range` is treated as a miss so a
    click on the sky or on the camera housing does not become a goal.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    denominator = float(np.dot(direction, normal))
    if abs(denominator) < 1e-6:
        return None
    t = float(np.dot(plane_point - origin, normal)) / denominator
    if t < min_range or t > max_range:
        return None
    return origin + t * direction


def standoff_pose(
    target_xy,
    robot_xy,
    standoff: float,
    stand_xy=None,
    min_move: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (x, y, yaw): where to stand and which way to face.

    With `stand_xy` the robot stands there facing the target (operator-chosen
    heading). Otherwise it stops `standoff` meters short of the target along the
    line from its current position, facing the target; if it is already closer
    than that it stays put and only turns.
    """
    target = np.asarray(target_xy, dtype=float)[:2]
    robot = np.asarray(robot_xy, dtype=float)[:2]

    if stand_xy is not None:
        stand = np.asarray(stand_xy, dtype=float)[:2]
        facing = target - stand
        if np.linalg.norm(facing) < 1e-3:
            # Standing on the target: keep the robot's current line of approach.
            facing = target - robot
        yaw = float(np.arctan2(facing[1], facing[0]))
        return float(stand[0]), float(stand[1]), yaw

    to_target = target - robot
    distance = float(np.linalg.norm(to_target))
    if distance < 1e-3:
        raise ValueError("Target coincides with the robot position")
    yaw = float(np.arctan2(to_target[1], to_target[0]))
    if distance - standoff < min_move:
        return float(robot[0]), float(robot[1]), yaw
    stand = target - to_target / distance * standoff
    return float(stand[0]), float(stand[1]), yaw


def heading_error(a: float, b: float) -> float:
    """Smallest signed difference a - b, wrapped to [-pi, pi]."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)
