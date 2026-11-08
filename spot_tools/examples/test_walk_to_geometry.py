"""Geometry behind the operator's walk-to-before-pick request. No robot needed."""

import numpy as np
import pytest

from spot_skills.approval import ApprovalDecision
from spot_skills.walk_to_geometry import (
    heading_error,
    intersect_ray_plane,
    pixel_ray,
    standoff_pose,
)

GROUND = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def test_pixel_ray_center_points_forward_and_offsets_follow_image_axes():
    center = pixel_ray((320, 240), fx=400, fy=400, cx=320, cy=240)
    assert np.allclose(center, [0, 0, 1])
    right_down = pixel_ray((720, 640), fx=400, fy=400, cx=320, cy=240)
    assert right_down[0] > 0 and right_down[1] > 0 and right_down[2] > 0
    assert np.isclose(np.linalg.norm(right_down), 1.0)
    with pytest.raises(ValueError):
        pixel_ray((0, 0), fx=0, fy=400, cx=0, cy=0)


def test_ray_hits_ground_ahead_and_misses_sky_and_backwards():
    origin = np.array([0.0, 0.0, 0.6])
    ahead_down = np.array([2.0, 0.0, -0.6])
    ahead_down /= np.linalg.norm(ahead_down)
    hit = intersect_ray_plane(origin, ahead_down, *GROUND)
    assert np.allclose(hit, [2.0, 0.0, 0.0])

    assert intersect_ray_plane(origin, np.array([1.0, 0.0, 0.0]), *GROUND) is None
    assert intersect_ray_plane(origin, np.array([1.0, 0.0, 0.3]), *GROUND) is None
    far = np.array([30.0, 0.0, -0.6])
    assert intersect_ray_plane(origin, far / np.linalg.norm(far), *GROUND) is None


def test_default_standoff_stops_short_facing_target():
    x, y, yaw = standoff_pose(target_xy=(4.0, 3.0), robot_xy=(0.0, 0.0), standoff=1.0)
    assert np.isclose(np.hypot(4.0 - x, 3.0 - y), 1.0)
    assert np.isclose(yaw, np.arctan2(3.0, 4.0))
    # Already closer than the standoff: stay put, only turn.
    x, y, yaw = standoff_pose(target_xy=(0.5, 0.0), robot_xy=(0.0, 0.0), standoff=1.0)
    assert (x, y) == (0.0, 0.0) and np.isclose(yaw, 0.0)
    with pytest.raises(ValueError):
        standoff_pose(target_xy=(0.0, 0.0), robot_xy=(0.0, 0.0), standoff=1.0)


def test_operator_stand_point_sets_position_and_heading():
    x, y, yaw = standoff_pose(
        target_xy=(2.0, 2.0), robot_xy=(0.0, 0.0), standoff=1.0, stand_xy=(2.0, 0.0)
    )
    assert (x, y) == (2.0, 0.0)
    assert np.isclose(yaw, np.pi / 2)


def test_heading_error_wraps():
    assert np.isclose(heading_error(np.pi - 0.1, -np.pi + 0.1), -0.2)
    assert np.isclose(heading_error(0.3, 0.1), 0.2)


def test_decision_kinds_are_exclusive():
    assert ApprovalDecision.approve((1, 2), 0).approved
    walk = ApprovalDecision.walk((1, 2), 1, stand_pixel=(3, 4))
    assert walk.walk_to and not walk.approved and not walk.rejected
    assert walk.stand_pixel == (3, 4)
    assert ApprovalDecision.reject().rejected
