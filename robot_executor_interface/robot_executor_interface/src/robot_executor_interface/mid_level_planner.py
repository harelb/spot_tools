import heapq
import threading

import numpy as np
import shapely
from attr import dataclass
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt


def invert_pose(p):
    p_inv = np.zeros((4, 4))
    p_inv[:3, :3] = p[:3, :3].T
    p_inv[:3, 3] = -p[:3, :3].T @ p[:3, 3]
    p_inv[3, 3] = 1
    return p_inv


class OccupancyMap:
    """
    OccupancyMap is designed to be asynchronously updated by a ROS callback. To safely access the map, you do

    ```
        om = OccupancyMap(...)
        with om:
            # run the planner
    ```

    """

    def __init__(
        self,
        feedback,
        occupancy_map=None,
        map_resolution=None,
        map_origin=None,
        inflate_radius_meters=0.5,
        use_cost_map=False,
        safe_distance=0.5,
        nearest_obstacle_cost=5.0,
    ):
        self.feedback = feedback
        self.occupancy_map = occupancy_map
        self.map_resolution = map_resolution
        self.map_origin = map_origin
        self.robot_pose = None  # <robot>/odom frame
        self.inflate_radius_meters = inflate_radius_meters
        self.use_cost_map = use_cost_map
        self.safe_distance = safe_distance  # meters
        self.nearest_obstacle_cost = nearest_obstacle_cost
        self.proximity_cost_map = None
        self.map_lock = threading.Lock()  # used for async updates
        self.last_update_time = None

    def __enter__(self):
        self.map_lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.map_lock.release()

    def get_grid(self):
        return self.occupancy_map

    def clone_grid(self):
        with self:
            return self.occupancy_map.copy() if self.occupancy_map is not None else None

    def set_grid(self, grid, resolution, origin, robot_pose, update_time):
        with self:
            self.last_update_time = update_time
            self.occupancy_map = grid
            self.map_resolution = resolution
            self.map_origin = origin
            self.robot_pose = robot_pose
            self.occupancy_map = self.inflate_obstacles(
                self.occupancy_map, self.inflate_radius_meters
            )
            if self.use_cost_map:
                self.proximity_cost_map = self._compute_proximity_cost_map(
                    self.occupancy_map
                )
        self.feedback.print(
            "DEBUG",
            f"Occupancy map updated: shape={self.occupancy_map.shape}, resolution={self.map_resolution}, origin=\n{self.map_origin}, robot_pose=\n{self.robot_pose}",
        )

    def global_position_to_grid_cell(self, pose):
        """
        Input:
            - pose: (4,1) numpy array in map (global) frame
        Output:
            - (x, y): tuple in grid frame

        indexing: (i, j) = (row, col) = (y, x)
        """
        pose_in_grid_frame = invert_pose(self.map_origin) @ pose  # (4,1)

        # Convert the pose to grid coordinates
        grid_j = int(pose_in_grid_frame[0, 0] / self.map_resolution)
        grid_i = int(pose_in_grid_frame[1, 0] / self.map_resolution)

        return (grid_i, grid_j)

    def grid_cell_to_global_pose(self, cell_indx):
        """
        Input:
            - cell: (x, y) tuple in grid frame
        Output:
            - pose: (4,1) numpy array in map (global) frame
        indexing: (i, j) = (row, col) = (y, x)
        """
        i, j = cell_indx
        pose_in_grid_frame = np.array(
            [j * self.map_resolution, i * self.map_resolution, 0, 1]
        ).reshape(4, 1)
        pose_in_global_frame = self.map_origin @ pose_in_grid_frame
        return pose_in_global_frame

    def inflate_obstacles(self, grid, inflate_radius_meters):
        """
        Inflates obstacles in the occupancy grid by a given radius in meters.

        Input:
            - inflate_radius_meters: float, the radius to inflate obstacles by (in meters)

        Output:
            - inflated_grid: numpy array, the occupancy grid with inflated obstacles

        Note: This function creates a copy of the grid and returns it without modifying
              the original self.occupancy_map. Call set_grid() to update the grid.
        """

        # Convert radius from meters to grid cells
        inflate_radius_cells = int(np.ceil(inflate_radius_meters / self.map_resolution))

        if inflate_radius_cells <= 0:
            self.feedback.print(
                "INFO",
                f"Inflate radius too small ({inflate_radius_meters}m = {inflate_radius_cells} cells), returning original grid",
            )
            return grid.copy()

        # Create binary mask of obstacles (occupied cells with value > 0)
        obstacle_mask = grid > 0

        # Create circular structuring element for dilation
        y, x = np.ogrid[
            -inflate_radius_cells : inflate_radius_cells + 1,
            -inflate_radius_cells : inflate_radius_cells + 1,
        ]
        structuring_element = x**2 + y**2 <= inflate_radius_cells**2

        # Dilate the obstacle mask
        inflated_mask = binary_dilation(obstacle_mask, structure=structuring_element)

        # Create inflated grid (copy of original)
        inflated_grid = grid.copy()

        # Update cells that were free (0) but are now in inflated zone
        # Preserve unknown (-1) cells and already occupied cells
        newly_occupied = inflated_mask & (grid == 0)
        inflated_grid[newly_occupied] = 100

        return inflated_grid

    def _compute_proximity_cost_map(self, inflated_grid):
        """
        Computes a per-cell proximity cost based on Euclidean distance to the nearest obstacle.

        For each free cell, if its distance to the nearest obstacle is within safe_distance,
        a Gaussian (RBF) cost is assigned: cost = nearest_obstacle_cost * exp(-d^2 / (2*sigma^2))
        where sigma = safe_distance_cells / 3, so cost drops to ~1% of max at the safe_distance
        boundary. Cells beyond safe_distance get zero cost.

        Returns a float32 array with the same shape as inflated_grid.
        """
        free_mask = inflated_grid == 0
        dist_cells = distance_transform_edt(free_mask)  # distance in grid cells
        safe_dist_cells = (
            self.safe_distance / self.map_resolution
        )  # convert safe distance to grid cells
        sigma = safe_dist_cells / 3.0
        raw_cost = self.nearest_obstacle_cost * np.exp(
            -(dist_cells**2) / (2 * sigma**2)
        )
        proximity_cost_map = np.where(dist_cells < safe_dist_cells, raw_cost, 0.0)
        return proximity_cost_map.astype(np.float32)


@dataclass
class MidLevelPlannerOutput:
    target_point_metric: np.ndarray
    path_shapely: shapely.LineString
    path_waypoints_metric: list
    global_path_target_point_metric: np.ndarray


class MidLevelPlanner:
    def __init__(
        self,
        occupancy_map: OccupancyMap,
        feedback,
        use_fake_path_planner=False,
        lookahead_distance_grid=50,
    ):
        self.feedback = feedback
        # poses are 4x4 homogeneous transformation matrix
        self.robot_pose = None  # <robot>/odom frame
        # high level plan is in <robot>/odom frame
        self.lookahead_distance_grid = lookahead_distance_grid  # grid cells

        self.occupancy_map_obj = occupancy_map

        self.use_fake_path_planner = use_fake_path_planner

        self.set_robot_pose()
        self.global_position_to_grid_cell = (
            self.occupancy_map_obj.global_position_to_grid_cell
        )
        self.grid_cell_to_global_pose = self.occupancy_map_obj.grid_cell_to_global_pose

        self.feedback.print(
            "INFO", f"MidLevelPlanner initialized, {self.use_fake_path_planner=}"
        )
        if self.use_fake_path_planner:
            self.feedback.print("INFO", "NOTE: See path in <robot>odom frame")

    def set_robot_pose(self):
        self.robot_pose = self.occupancy_map_obj.robot_pose

    @property
    def occupancy_map(self) -> np.ndarray:
        return self.occupancy_map_obj.get_grid()

    def get_progress_along_path(self, path_shapely, current_point):
        self.feedback.print("DEBUG", f"Current point: {current_point}")
        current_point_shapely = shapely.Point(
            current_point[0], current_point[1]
        )  # (x,y) = (col, row)
        return shapely.line_locate_point(path_shapely, current_point_shapely)

    def plan_path(self, high_level_path_metric):
        """
        Input: high level path in <robot>/odom frame, Nx2 numpy array
        Output: (bool, path) -> (success, path in odom frame)
        """

        ## First get target point along the path
        # 1. project to current path distance
        # Assume self.robot_pose is in vision coordinate frame

        with self.occupancy_map_obj:  # TODO: too ugly, any better way?
            # get robot pose
            self.set_robot_pose()

            # convert poses to grid cells
            high_level_path_grid = [
                self.global_position_to_grid_cell(
                    np.array([pt[0], pt[1], 0, 1]).reshape(4, 1)
                )
                for pt in high_level_path_metric[:, :2]
            ]  # convert high level path position to grid cells indexes
            high_level_path_grid = np.array(high_level_path_grid).reshape(-1, 2)
            self.feedback.print(
                "DEBUG", f"High level path (grid cells): {high_level_path_grid}"
            )
            current_point_grid = self.global_position_to_grid_cell(
                self.robot_pose[:, 3].reshape(4, 1)
            )
            current_point_grid = self.project_goal_to_grid_naive(current_point_grid)
            self.feedback.print(
                "DEBUG", f"Current point (grid cell): {current_point_grid}"
            )

            # convert poses to shapely points/lines
            high_level_path_shapely = shapely.LineString(high_level_path_grid)
            # where are we along the path?
            progress_distance_shapely = self.get_progress_along_path(
                high_level_path_shapely, current_point_grid
            )
            self.feedback.print(
                "INFO",
                f"Current point: {current_point_grid}, progress distance along path: {progress_distance_shapely}, lookahead distance: {self.lookahead_distance_grid}",
            )

            # get line point at lookahead distance
            target_distance_shapely = (
                progress_distance_shapely + self.lookahead_distance_grid
            )
            target_point_shapely = shapely.line_interpolate_point(
                high_level_path_shapely, target_distance_shapely
            )

            # find the target in grid cell coordinates (make it free)
            target_point_grid = (
                int(target_point_shapely.x),
                int(target_point_shapely.y),
            )
            target_point_grid_proj = self.project_goal_to_grid(
                target_point_grid, high_level_path_shapely
            )

            output = MidLevelPlannerOutput(
                target_point_metric=self.grid_cell_to_global_pose(
                    target_point_grid_proj
                ),
                path_shapely=shapely.LineString(high_level_path_metric[:, :2]),
                path_waypoints_metric=[],
                global_path_target_point_metric=self.grid_cell_to_global_pose(
                    target_point_grid
                ),
            )

            # plan using a_star
            a_star_path_grid = self.a_star(current_point_grid, target_point_grid_proj)

            if a_star_path_grid is None:
                # fall back using the high level path directly, and return empty visualization
                self.feedback.print(
                    "INFO", "A* failed, falling back to high level path"
                )
                return False, output

            # convert a_star path to metric coordinates
            a_star_path_metric = [
                self.grid_cell_to_global_pose((pt[0], pt[1])) for pt in a_star_path_grid
            ]
            a_star_path_metric = np.array(a_star_path_metric).reshape(-1, 4)
            a_star_path_metric = a_star_path_metric[:, :2].reshape(-1, 2)
            # TODO(multy) temp fix for visualization if a star only output one point
            # in fact, this is probably reaching a local min we plan can't go farther.
            if len(a_star_path_metric) == 1:
                a_star_path_metric = np.vstack(
                    (a_star_path_metric[0], a_star_path_metric)
                )
            a_star_path_execute = a_star_path_metric
            output.path_shapely = shapely.LineString(a_star_path_execute)
            output.path_waypoints_metric = a_star_path_metric
            return True, output

    def project_goal_to_grid_naive(self, cell):
        h, w = self.occupancy_map.shape
        bound_i, bound_j = [0, h - 1], [0, w - 1]

        projected_cell = (
            max(bound_i[0], min(cell[0], bound_i[1])),
            max(bound_j[0], min(cell[1], bound_j[1])),
        )
        if self.occupancy_map[projected_cell[0], projected_cell[1]] != 0:
            # if the goal is in an obstacle, find the nearest free cell
            free_cells = np.argwhere(self.occupancy_map == 0)
            if free_cells.size == 0:
                return None
            dists = np.linalg.norm(free_cells - np.array(projected_cell).T, axis=1)
            projected_cell = tuple(free_cells[np.argmin(dists)])

        # debug the cell projection is correct
        self.feedback.print("DEBUG", f"Projected cell: {projected_cell}")
        return projected_cell

    def get_frontier_cells(self):
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0

        # Unknown neighbor count for each cell
        unknown_neighbors = convolve(
            (self.occupancy_map == -1).astype(np.uint8), kernel, mode="constant", cval=1
        )

        # Edge mask (True for border cells)
        edge_mask = np.zeros_like(self.occupancy_map, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True

        # Frontier = free and (has unknown neighbor OR on edge)
        frontier_mask = (self.occupancy_map == 0) & (
            (unknown_neighbors > 0) | edge_mask
        )

        frontier_cells = np.argwhere(frontier_mask)

        return frontier_cells

    def is_free(self, cell):
        return self.occupancy_map[cell[0], cell[1]] == 0

    def project_goal_observed(self, goal, path_shapely, epsilon=0):
        if self.is_free(goal):
            return goal
        free_cells = np.argwhere(self.occupancy_map == 0)
        if free_cells.size == 0:
            return None

        dists_free = np.linalg.norm(free_cells - np.array(goal).T, axis=1)

        free_cell_candidate = tuple(free_cells[np.argmin(dists_free)])

        return free_cell_candidate

    def project_goal_observed_with_frontier(self, goal, path_shapely, epsilon=0):
        if self.is_free(goal):
            return goal
        free_cells = np.argwhere(self.occupancy_map == 0)
        frontier_cells = self.get_frontier_cells()
        if free_cells.size == 0:
            return None

        dists_free = np.linalg.norm(free_cells - np.array(goal).T, axis=1)

        free_cell_candidate = tuple(free_cells[np.argmin(dists_free)])
        free_cell_progress = self.get_progress_along_path(
            path_shapely, free_cell_candidate
        )

        if frontier_cells.size == 0:
            return free_cell_candidate

        dists_frontier = np.array(
            [
                self.get_progress_along_path(path_shapely, frontier_cell)
                + np.linalg.norm(np.array(frontier_cell) - np.array(goal).T)
                for frontier_cell in frontier_cells
            ]
        )
        frontier_cell_candidate = tuple(frontier_cells[np.argmax(dists_frontier)])
        frontier_cell_progress = self.get_progress_along_path(
            path_shapely, frontier_cell_candidate
        )

        if frontier_cell_progress + epsilon < free_cell_progress:
            return free_cell_candidate

        return frontier_cell_candidate

    def project_goal_unknown_frontier(self, goal):
        # Define a 4-connected kernel
        frontier_cells = self.get_frontier_cells()
        if frontier_cells.size == 0:
            return None
        dists = np.linalg.norm(frontier_cells - np.array(goal).T, axis=1)
        return tuple(frontier_cells[np.argmin(dists)])

    def project_goal_to_grid(self, goal, path_shapely):
        ## assumes that goal is in same coordinate frame as the occupancy grid
        # heurestic for right now will be clamping it to the occupancy grid.
        h, w = self.occupancy_map.shape
        bound_i, bound_j = [0, h - 1], [0, w - 1]

        def within_bounds(cell):
            return (bound_i[0] <= cell[0] <= bound_i[1]) and (
                bound_j[0] <= cell[1] <= bound_j[1]
            )

        if within_bounds(goal):
            return self.project_goal_observed(goal, path_shapely)

        return self.project_goal_unknown_frontier(goal)

    def a_star(self, start, goal, diagonal_movement=True):
        """
        Generated A* path planning algorithm.
        Programed by Gemini 2.5 Pro
        """
        if diagonal_movement:
            connectivity = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]  # 8-way connectivity
        else:
            connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-way connectivity

        if self.occupancy_map is None:
            return None

        rows, cols = self.occupancy_map.shape

        def is_valid(cell):
            return 0 <= cell[0] < rows and 0 <= cell[1] < cols

        def is_obstacle(cell):
            return self.occupancy_map[cell[0], cell[1]] > 0  # threshold

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        if (
            not is_valid(start)
            or not is_valid(goal)
            or is_obstacle(start)
            or is_obstacle(goal)
        ):
            return None

        open_set = [(0, start)]  # (f_cost, position)
        came_from = {}
        g_cost = {(r, c): float("inf") for r in range(rows) for c in range(cols)}
        g_cost[start] = 0

        f_cost = {(r, c): float("inf") for r in range(rows) for c in range(cols)}
        f_cost[start] = heuristic(start, goal)

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in connectivity:
                neighbor = (current[0] + dr, current[1] + dc)

                if not is_valid(neighbor) or is_obstacle(neighbor):
                    continue

                prox_cost = (
                    self.occupancy_map_obj.proximity_cost_map[neighbor[0], neighbor[1]]
                    if self.occupancy_map_obj.proximity_cost_map is not None
                    else 0.0
                )
                tentative_g_cost = g_cost[current] + 1 + prox_cost
                if tentative_g_cost < g_cost[neighbor]:
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g_cost
                    f_cost[neighbor] = g_cost[neighbor] + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_cost[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found


class IdentityPlanner:
    def __init__(self, feedback):
        self.feedback = feedback
        self.feedback.print("INFO", "IdentityPlanner initialized")

    def plan_path(self, high_level_path_metric):
        """
        Input: high level path in <robot>/odom frame, Nx2 numpy array
        Output: (bool, path) -> (success, path in odom frame)
        """
        output = MidLevelPlannerOutput(
            target_point_metric=None,
            path_shapely=shapely.LineString(high_level_path_metric[:, :2]),
            path_waypoints_metric=None,
            global_path_target_point_metric=None,
        )
        return True, output
