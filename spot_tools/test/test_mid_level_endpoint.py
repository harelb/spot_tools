from unittest.mock import Mock

import numpy as np
import pytest

from robot_executor_interface.mid_level_planner import MidLevelPlanner, OccupancyMap


def planner_at(start, *, yaw=0, lookahead=50):
    origin = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    origin[:2, :2] = [[c, -s], [s, c]]
    origin[:2, 3] = [29, 34]
    pose = origin.copy()
    pose[:, 3] = origin @ np.array([*start, 0, 1])
    grid = OccupancyMap(Mock(), np.zeros((20, 20), dtype=np.int8), .1, origin,
                        inflate_radius_meters=0)
    grid.robot_pose = pose
    planner = MidLevelPlanner(grid, Mock(), lookahead_distance_grid=lookahead)
    return planner, origin


def world_path(origin, points):
    return np.array([(origin @ np.array([*p, 0, 1]))[:2] for p in points])


@pytest.mark.parametrize('yaw', [0, .4])
@pytest.mark.parametrize('start', [[.15, .15], [.54, .65]])
def test_free_final_cell_preserves_exact_reviewed_endpoint(yaw, start):
    planner, origin = planner_at(start, yaw=yaw)
    route = world_path(origin, [start, [.57, .68]])
    success, output = planner.plan_path(route)
    assert success
    np.testing.assert_allclose(output.path_waypoints_metric[-1], route[-1], atol=1e-12)
    np.testing.assert_allclose(output.path_shapely.coords[-1], route[-1], atol=1e-12)
    np.testing.assert_allclose(output.target_point_metric[:2, 0], route[-1], atol=1e-12)


@pytest.mark.parametrize('occupancy', [-1, 100])
def test_projected_unknown_or_blocked_goal_does_not_restore_exact_endpoint(occupancy):
    planner, origin = planner_at([.15, .15])
    planner.occupancy_map[6, 5] = occupancy
    route = world_path(origin, [[.15, .15], [.57, .68]])
    success, output = planner.plan_path(route)
    assert success
    assert np.linalg.norm(output.path_waypoints_metric[-1] - route[-1]) > .01
    cell = planner.global_position_to_grid_cell(output.target_point_metric)
    assert planner.is_free(cell)


def test_intermediate_lookahead_does_not_skip_to_route_end():
    planner, origin = planner_at([.15, .15], lookahead=2)
    route = world_path(origin, [[.15, .15], [1.57, 1.68]])
    success, output = planner.plan_path(route)
    assert success
    assert np.linalg.norm(output.path_waypoints_metric[-1] - route[-1]) > 1


def test_failed_raster_plan_stays_failed():
    planner, origin = planner_at([.15, .15])
    planner.a_star = Mock(return_value=None)
    success, output = planner.plan_path(world_path(origin, [[.15, .15], [.57, .68]]))
    assert not success
    assert output.path_waypoints_metric == []
