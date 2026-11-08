"""Walk toward a point the operator clicked in a candidate image.

Used by the pick flow when the operator answers a manipulation approval request
with "walk to" instead of approve/reject: the clicked pixel is lifted to a 3D
point on the ground plane, a standoff pose is computed, the robot walks there,
and the caller captures fresh images and asks again.
"""

import time
from typing import Optional, Tuple

import numpy as np
from bosdyn.api import basic_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_se2_a_tform_b,
)

from spot_skills.navigation_utils import navigate_to_absolute_pose
from spot_skills.walk_to_geometry import (
    heading_error,
    intersect_ray_plane,
    pixel_ray,
    plan_walk,
)

# How close counts as "arrived" when the trajectory feedback is not conclusive.
ARRIVAL_POSITION_TOLERANCE = 0.25  # meters
ARRIVAL_HEADING_TOLERANCE = 0.15  # radians


def _log(feedback, message):
    print(message)
    if feedback is not None:
        feedback.print("INFO", message)


def ground_plane_in_vision(robot_state) -> Tuple[np.ndarray, np.ndarray]:
    """Return (point, normal) of Spot's ground plane estimate in the vision frame."""
    snapshot = robot_state.kinematic_state.transforms_snapshot
    vision_T_gpe = get_a_tform_b(snapshot, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    if vision_T_gpe is None:
        raise ValueError("Robot state has no ground plane estimate frame")
    point = np.array([vision_T_gpe.x, vision_T_gpe.y, vision_T_gpe.z])
    normal = np.asarray(vision_T_gpe.rot.to_matrix()) @ np.array([0.0, 0.0, 1.0])
    return point, normal


def camera_pose_in_vision(image_response) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, rotation matrix) of the image sensor in the vision frame."""
    shot = image_response.shot
    if not shot.frame_name_image_sensor:
        raise ValueError("Image has no sensor frame name")
    vision_T_cam = get_a_tform_b(
        shot.transforms_snapshot, VISION_FRAME_NAME, shot.frame_name_image_sensor
    )
    if vision_T_cam is None:
        raise ValueError(
            f"Image snapshot cannot relate {shot.frame_name_image_sensor} to vision"
        )
    origin = np.array([vision_T_cam.x, vision_T_cam.y, vision_T_cam.z])
    rotation = np.asarray(vision_T_cam.rot.to_matrix())
    return origin, rotation


def pixel_to_ground_point(image_response, pixel, robot_state) -> Optional[np.ndarray]:
    """Lift an image pixel to a 3D point on the ground plane, in the vision frame.

    Returns None when the geometry is unusable: no pinhole model, no sensor
    frame, or a ray that misses the ground (sky, horizon, implausible range).
    """
    source = image_response.source
    if not source.HasField("pinhole"):
        raise ValueError(f"Image source {source.name} has no pinhole camera model")
    intrinsics = source.pinhole.intrinsics
    direction_cam = pixel_ray(
        pixel,
        intrinsics.focal_length.x,
        intrinsics.focal_length.y,
        intrinsics.principal_point.x,
        intrinsics.principal_point.y,
    )
    origin, rotation = camera_pose_in_vision(image_response)
    direction = rotation @ direction_cam
    plane_point, plane_normal = ground_plane_in_vision(robot_state)
    return intersect_ray_plane(origin, direction, plane_point, plane_normal)


def robot_xy_yaw_in_vision(robot_state) -> Tuple[float, float, float]:
    snapshot = robot_state.kinematic_state.transforms_snapshot
    vision_T_body = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    return vision_T_body.x, vision_T_body.y, vision_T_body.angle


def _trajectory_status(command_feedback):
    try:
        return command_feedback.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
    except AttributeError:
        return None


def _drive_to(spot, goal: math_helpers.SE2Pose, timeout, feedback, should_stop) -> bool:
    """Issue the SE2 goal and wait until Spot arrives, is stopped, or times out."""
    AT_GOAL = basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL
    NEAR_GOAL = basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL

    deadline = time.time() + timeout
    cmd_id = navigate_to_absolute_pose(
        spot, goal, frame_name=VISION_FRAME_NAME, end_time=timeout
    )
    near_since = None
    while time.time() < deadline:
        if should_stop():
            _log(feedback, "Walk-to interrupted.")
            return False

        if spot.is_fake:
            # The fake command client steps the pose once per command, so keep
            # re-issuing the goal until the fake robot gets there.
            x, y, _ = spot.get_pose()[:3]
            if np.hypot(goal.x - x, goal.y - y) < ARRIVAL_POSITION_TOLERANCE:
                return True
            navigate_to_absolute_pose(
                spot, goal, frame_name=VISION_FRAME_NAME, end_time=timeout
            )
            continue

        status = _trajectory_status(spot.command_client.robot_command_feedback(cmd_id))
        if status == AT_GOAL:
            return True
        if status == NEAR_GOAL:
            # Spot can hover at NEAR_GOAL indefinitely on uneven ground; accept
            # it once it has stayed there briefly and the pose is close enough.
            near_since = near_since or time.time()
            x, y, yaw = robot_xy_yaw_in_vision(spot.get_state())
            close = (
                np.hypot(goal.x - x, goal.y - y) < ARRIVAL_POSITION_TOLERANCE
                and abs(heading_error(goal.angle, yaw)) < ARRIVAL_HEADING_TOLERANCE
            )
            if close or time.time() - near_since > 3.0:
                return True
        else:
            near_since = None
        time.sleep(0.2)

    _log(feedback, "Walk-to timed out before reaching the standoff pose.")
    return False


def walk_to_pixel(
    spot,
    image_response,
    pixel,
    stand_pixel=None,
    standoff: float = 1.5,
    timeout: float = 25.0,
    feedback=None,
    should_stop=lambda: False,
    max_distance: float = 5.0,
) -> bool:
    """Walk to a standoff pose for the ground/object point under `pixel`.

    `stand_pixel`, when given, chooses where to stand (and hence the heading)
    instead of the default "stop `standoff` meters short, facing the target".
    No single walk translates more than `max_distance` meters: the ground-plane
    estimate grows unreliable for far, shallow rays, so a far target is
    approached in steps. Returns True once the robot is at the pose. The caller
    should re-detect afterwards; this function never grasps.
    """
    if spot.is_fake:
        # No camera model on the fake robot: pretend the click was 2 m ahead.
        x, y, yaw = spot.get_pose()[0], spot.get_pose()[1], spot.get_pose()[3]
        target = np.array([x + 2.0 * np.cos(yaw), y + 2.0 * np.sin(yaw)])
        stand = None
        if stand_pixel is not None:
            stand = target + np.array([-np.sin(yaw), np.cos(yaw)]) * standoff
        gx, gy, gyaw, _ = plan_walk(
            target, (x, y), standoff, stand_xy=stand, max_distance=max_distance
        )
        _log(feedback, f"[fake] Walking to ({gx:.2f}, {gy:.2f}, yaw {gyaw:.2f})")
        return _drive_to(
            spot, math_helpers.SE2Pose(gx, gy, gyaw), timeout, feedback, should_stop
        )

    robot_state = spot.get_state()
    target = pixel_to_ground_point(image_response, pixel, robot_state)
    if target is None:
        _log(
            feedback,
            f"Walk-to pixel {tuple(pixel)} does not project onto the ground; staying put.",
        )
        return False

    stand = None
    if stand_pixel is not None:
        stand = pixel_to_ground_point(image_response, stand_pixel, robot_state)
        if stand is None:
            _log(
                feedback,
                f"Stand pixel {tuple(stand_pixel)} does not project onto the ground; "
                "using the default standoff instead.",
            )

    rx, ry, _ = robot_xy_yaw_in_vision(robot_state)
    try:
        gx, gy, gyaw, capped = plan_walk(
            target, (rx, ry), standoff, stand_xy=stand, max_distance=max_distance
        )
    except ValueError as ex:
        _log(feedback, f"Walk-to goal invalid: {ex}")
        return False

    _log(
        feedback,
        f"Walk-to target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) vision; "
        f"standing at ({gx:.2f}, {gy:.2f}) heading {np.degrees(gyaw):.0f} deg; "
        f"ground-plane distance {np.hypot(target[0] - rx, target[1] - ry):.2f} m, "
        f"walking {np.hypot(gx - rx, gy - ry):.2f} m"
        + (f" (capped at {max_distance:.1f} m)" if capped else ""),
    )
    walk_feedback = getattr(feedback, "walk_to_feedback", None)
    if walk_feedback is not None:
        walk_feedback(target, (gx, gy, gyaw))

    return _drive_to(
        spot, math_helpers.SE2Pose(gx, gy, gyaw), timeout, feedback, should_stop
    )


def walk_preview_geometry(spot, image_responses):
    """Geometry a viewer needs to reproduce pixel_to_ground_point and plan_walk.

    Returns a dict with, per image, ``cameras[i] = (fx, fy, cx, cy, x, y, z, qw,
    qx, qy, qz)`` (sensor pose in the vision frame) or None, plus ``ground``
    (point, normal) and ``robot`` (x, y, yaw), each None when unavailable. Never
    raises: a missing preview must not block the pick.
    """
    preview = {"cameras": [None] * len(image_responses), "ground": None, "robot": None}
    if getattr(spot, "is_fake", False):
        return preview
    for index, response in enumerate(image_responses):
        try:
            source, shot = response.source, response.shot
            if not source.HasField("pinhole") or not shot.frame_name_image_sensor:
                continue
            k = source.pinhole.intrinsics
            pose = get_a_tform_b(
                shot.transforms_snapshot,
                VISION_FRAME_NAME,
                shot.frame_name_image_sensor,
            )
            if pose is None or k.focal_length.x <= 0 or k.focal_length.y <= 0:
                continue
            preview["cameras"][index] = (
                k.focal_length.x,
                k.focal_length.y,
                k.principal_point.x,
                k.principal_point.y,
                pose.x,
                pose.y,
                pose.z,
                pose.rot.w,
                pose.rot.x,
                pose.rot.y,
                pose.rot.z,
            )
        except Exception:
            continue
    try:
        state = spot.get_state()
        point, normal = ground_plane_in_vision(state)
        preview["ground"] = (tuple(map(float, point)), tuple(map(float, normal)))
        preview["robot"] = tuple(map(float, robot_xy_yaw_in_vision(state)))
    except Exception:
        pass
    return preview
