"""Interface for moving the spot frame."""

import time
from typing import Tuple

import numpy as np
import shapely
from bosdyn.api import geometry_pb2
from bosdyn.api.spot import robot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from numpy.typing import ArrayLike
from spot_skills.timeouts import skill_timeout

# Global constants to define spot motion speed
MAX_LINEAR_VEL = 0.75
MAX_ROTATION_VEL = 0.65


def navigate_to_relative_pose(
    spot,
    body_tform_goal: math_helpers.SE2Pose,
    max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
    timeout: float = 20.0,
):
    # Get an SE2 pose for vision_tform_body to convert the body-based command to a non-moving frame
    # that can be issued to the robot.
    robot_state = spot.get_state()
    transforms = robot_state.kinematic_state.transforms_snapshot

    assert str(transforms) != ""

    vision_tform_body = get_se2_a_tform_b(
        transforms, VISION_FRAME_NAME, BODY_FRAME_NAME
    )
    vision_tform_goal = vision_tform_body * body_tform_goal

    navigate_to_absolute_pose(spot, waypoint=vision_tform_goal)


def navigate_to_absolute_pose(
    spot, waypoint, frame_name=VISION_FRAME_NAME, stairs=False
):
    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )

    params = RobotCommandBuilder.mobility_params(
        stair_hint=stairs, locomotion_hint=robot_command_pb2.LocomotionHint.HINT_AUTO
    )

    max_vel_linear = geometry_pb2.Vec2(x=MAX_LINEAR_VEL, y=MAX_LINEAR_VEL)
    max_vel_se2 = geometry_pb2.SE2Velocity(
        linear=max_vel_linear, angular=MAX_ROTATION_VEL
    )
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params.vel_limit.CopyFrom(vel_limit)

    state = spot.get_state()
    manipulator_state = state.manipulator_state

    gripper_holding = manipulator_state.is_gripper_holding_item

    if gripper_holding:
        arm_joint_freeze_command = RobotCommandBuilder.arm_joint_freeze_command()
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=waypoint.x,
            goal_y=waypoint.y,
            goal_heading=waypoint.angle,
            frame_name=frame_name,
            params=params,
            build_on_command=arm_joint_freeze_command,
        )
    else:
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=waypoint.x,
            goal_y=waypoint.y,
            goal_heading=waypoint.angle,
            frame_name=frame_name,
            params=params,
        )
    end_time = skill_timeout(10., 'navigation')
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + end_time
    )
    return cmd_id


def navigation_timeout(path, seconds_per_meter):
    """Allow short paths to settle and budget rotation even at zero distance."""
    path=np.asarray(path,dtype=float)
    if path.ndim!=2 or path.shape[1]!=3 or len(path)<1 or not np.isfinite(path).all():
        raise ValueError('Navigation timeout requires finite x/y/yaw waypoints')
    if not np.isfinite(seconds_per_meter) or seconds_per_meter<=0:
        raise ValueError('Invalid navigation time budget')
    distance=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum())
    yaw=np.diff(path[:,2]);rotation=float(abs(np.arctan2(np.sin(yaw),np.cos(yaw))).sum())
    return max(skill_timeout(15., 'navigation'),distance*seconds_per_meter+rotation*max(5.,seconds_per_meter/2))


def follow_trajectory_continuous(
    spot,
    waypoints_list: ArrayLike,
    lookahead_distance: float,
    goal_tolerance: float,
    timeout: float,
    mid_level_planner,
    frame_name=VISION_FRAME_NAME,
    stairs=False,
    feedback=None,
) -> bool:
    """
    Follows a trajectory by commanding the robot to move to each waypoint in the specified frame.
    Args:
        waypoints_list (ArrayLike): List of list of positions in the format [x, y].
        frame_name (str): Desired frame for the trajectory.
        robot_command_client (RobotCommandClient): Client for sending robot commands.
        robot_state_client (RobotStateClient): Client for receiving robot state.
        occupancy_grid_subscriber (Union[spot_ros_utils.OccupancyGridSubscriber, None], optional): Subscriber for occupancy grid updates. Defaults to None.
        stairs (bool, optional): Flag indicating whether the robot is navigating stairs. Defaults to False.
    Returns:
        bool: True if the trajectory is successfully followed, False otherwise.
    """

    spot.robot.ensure_client(RobotCommandClient.default_service_name)

    end_pt = waypoints_list[-1, :2]
    reference_path = shapely.LineString(waypoints_list[:, :2])
    t0 = time.time()
    rate = 10
    def cancelled():
        return bool(feedback is not None and getattr(feedback,'break_out_of_waiting_loop',False))

    def stop():
        spot.command_client.robot_command(RobotCommandBuilder.stop_command())

    # TODO: reactive loop, yeild out the loop to get info
    while 1:
        if cancelled():
            stop()
            return False
        if time.time() - t0 > timeout:
            stop()
            return False
        tform_body_in_vision = spot.get_pose()
        distance_from_end = np.linalg.norm(
            end_pt - np.array([tform_body_in_vision[0], tform_body_in_vision[1]])
        )
        if distance_from_end < goal_tolerance:
            feedback.print("INFO", "Spot reached end of path")
            endpoint = math_helpers.SE2Pose(
                x=tform_body_in_vision[0],
                y=tform_body_in_vision[1],
                angle=float(waypoints_list[-1,2]) if waypoints_list.shape[1]>=3 else tform_body_in_vision[2],
            )
            command_id=navigate_to_absolute_pose(spot, endpoint, frame_name, stairs=stairs)
            return wait_for_navigation(spot,command_id,cancelled=cancelled,timeout=max(.1,min(skill_timeout(15., 'navigation'),timeout-(time.time()-t0))))


        # if mid_level_planner is not None:
        # update path every (couple?) loop
        mlp_success, planning_output = mid_level_planner.plan_path(
            waypoints_list[:, :2]
        )
        if cancelled():
            stop()
            return False
        path = planning_output.path_shapely
        path_wp = planning_output.path_waypoints_metric
        target_point_metric = planning_output.target_point_metric

        if feedback is not None and target_point_metric is not None:
            feedback.print("INFO", f"target_point_metric: {target_point_metric}")
            feedback.path_follow_MLP_feedback(path_wp, target_point_metric)

        if not mlp_success:
            feedback.print(
                "INFO", "Mid-level planner failed, following high-level path directly"
            )
            if time.time() - t0 > timeout:
                stop()
                return False

            path = shapely.LineString(waypoints_list[:, :2])
            continue

        if time.time() - t0 > timeout:
            # TODO: I think we need to tell Spot to stop?
            # TODO: Also, we should probably have a finer-grained
            # check about making progress
            stop()
            return False
        # 1. project to current path distance
        current_point = shapely.Point(tform_body_in_vision[0], tform_body_in_vision[1])
        progress_distance = shapely.line_locate_point(path, current_point)
        progress_point = shapely.line_interpolate_point(path, progress_distance)
        # 2. get line point at lookahead
        target_distance = progress_distance + lookahead_distance
        target_point = shapely.line_interpolate_point(path, target_distance)

        # Spot is holonomic. Cross-track translation must not turn the whole
        # footprint toward a nearby rounded grid cell in a narrow corridor.
        # Align with the route tangent while translating back onto the route.
        # Use the reviewed route for body heading. The local raster planner
        # changes translation targets at cell boundaries and may add a final
        # diagonal correction even along a straight reviewed corridor route.
        heading_distance=shapely.line_locate_point(reference_path,current_point)+lookahead_distance
        before=shapely.line_interpolate_point(reference_path,max(0.,min(reference_path.length,heading_distance)-.1))
        after=shapely.line_interpolate_point(reference_path,min(reference_path.length,heading_distance+.1))
        delta=np.array([after.x-before.x,after.y-before.y])
        yaw_angle=np.arctan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-6 else tform_body_in_vision[2]

        if feedback is not None:
            # get data back out
            # TODO: new function for MLP
            feedback.path_following_progress_feedback(progress_point, target_point)

        # 3. send command
        current_waypoint = math_helpers.SE2Pose(
            x=target_point.x, y=target_point.y, angle=yaw_angle
        )
        feedback.print("INFO", f"Navigating to waypoint {current_waypoint}")

        if cancelled():
            stop()
            return False
        command_id=navigate_to_absolute_pose(spot, current_waypoint, frame_name, stairs=stairs)
        time.sleep(1 / rate)
        # Read terminal hardware failures instead of repeatedly replacing a
        # rejected/stalled motion while reporting that the action is running.
        spot.command_client.robot_command_feedback(command_id)
    return True


def turn_to_point(spot, current_position, target_position):
    d = [
        target_position[0] - current_position[0],
        target_position[1] - current_position[1],
    ]
    angle = np.arctan2(d[1], d[0])
    waypoint = math_helpers.SE2Pose(
        x=current_position[0], y=current_position[1], angle=angle
    )
    return navigate_to_absolute_pose(spot, waypoint, "vision", stairs=False)


def wait_for_navigation(spot, command_id, cancelled=lambda:False, timeout=None):
    """Wait for actual base arrival before an arm operation begins."""
    from bosdyn.api import basic_command_pb2
    deadline=time.monotonic()+(skill_timeout(15., 'navigation') if timeout is None else timeout)
    try:
        while time.monotonic()<deadline:
            if cancelled():return False
            response=spot.command_client.robot_command_feedback(command_id)
            result=response.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback
            if result.status==basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:return True
            time.sleep(.05)
        return False
    finally:
        spot.command_client.robot_command(RobotCommandBuilder.stop_command())
