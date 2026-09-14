"""Interface for moving the spot hand."""

import time
from spot_skills.timeouts import skill_timeout

import numpy as np
from bosdyn.api import gripper_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient


def move_hand_to_relative_pose(spot, body_tform_goal: math_helpers.SE3Pose) -> None:
    """Move the spot hand.

    The target pose is relative to the robot's body.
    """

    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    # Build the arm command.
    cmd = RobotCommandBuilder.arm_pose_command(
        body_tform_goal.x,
        body_tform_goal.y,
        body_tform_goal.z,
        body_tform_goal.rot.w,
        body_tform_goal.rot.x,
        body_tform_goal.rot.y,
        body_tform_goal.rot.z,
        BODY_FRAME_NAME,
        2.0,
    )
    # Send the request.
    full_cmd = RobotCommandBuilder.build_synchro_command(cmd)
    cmd_id = robot_command_client.robot_command(full_cmd)

    # Wait until the arm arrives at the goal.
    if not block_until_arm_arrives(robot_command_client, cmd_id, skill_timeout(2.0)):
        raise RuntimeError("arm did not reach requested pose")


def gaze_at_relative_pose(
    spot, gaze_target: math_helpers.Vec3, duration: float = 2.0
) -> None:
    """Gaze at a point relative to the robot's body frame."""

    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    # Transform the gaze target from the body frame to the odom frame because
    # the gaze command results in shaking in the body frame.
    robot_state_client = spot.robot.ensure_client(RobotStateClient.default_service_name)
    robot_state = robot_state_client.get_robot_state()
    odom_tform_body = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        BODY_FRAME_NAME,
    )
    gaze_target = odom_tform_body.transform_vec3(gaze_target)
    # Build the arm command.
    cmd = RobotCommandBuilder.arm_gaze_command(
        gaze_target.x, gaze_target.y, gaze_target.z, ODOM_FRAME_NAME
    )
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    if not block_until_arm_arrives(robot_command_client, cmd_id, skill_timeout(duration)):
        raise RuntimeError("arm command stalled, expired or was cancelled")
    time.sleep(1.0)


def gaze_at_vision_pose(spot, gaze_target, duration=2, stow_after=False):
    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )

    # Build the arm command.
    cmd = RobotCommandBuilder.arm_gaze_command(
        gaze_target[0], gaze_target[1], gaze_target[2], VISION_FRAME_NAME
    )
    # Send the request.
    cmd_id = robot_command_client.robot_command(command=cmd)
    # Wait until the arm arrives at the goal.
    if not block_until_arm_arrives(robot_command_client, cmd_id, skill_timeout(duration)):
        raise RuntimeError("arm command stalled, expired or was cancelled")
    time.sleep(1.0)
    if stow_after:
        stow_arm(spot)
    return True


def stow_arm(spot, duration: float = 2.0) -> None:
    """Stow the spot arm."""

    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    # Build the arm command.
    cmd = RobotCommandBuilder.arm_stow_command()
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    if not block_until_arm_arrives(robot_command_client, cmd_id, skill_timeout(duration)):
        raise RuntimeError("arm command stalled, expired or was cancelled")
    return True


def arm_to_carry(spot, duration: float = 2.0) -> None:
    """Stow the spot arm."""

    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    # Build the arm command.
    cmd = RobotCommandBuilder.arm_carry_command()
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    if not block_until_arm_arrives(robot_command_client, cmd_id, skill_timeout(duration)):
        raise RuntimeError("arm command stalled, expired or was cancelled")
    return True


def arm_to_drop(spot, duration: float = 2.0) -> None:
    """Stow the spot arm."""

    looking_down_and_right_and_rotated_right_pose = math_helpers.SE3Pose(
        x=0.8,
        y=0.0,
        z=0.1,
        rot=math_helpers.Quat.from_pitch(np.pi / 2)
        * math_helpers.Quat.from_roll(np.pi / 2),
    )

    move_hand_to_relative_pose(spot, looking_down_and_right_and_rotated_right_pose)
    # gaze_at_relative_pose(spot, math_helpers.Vec3(x=-0.5, y=-3, z=-1))

    # set joint angles
    # robot_state = spot.state_client.get_robot_state()
    # joint_angles = robot_state.kinematic_state.joint_angles
    # joint_angles = robot_state.kinematic_state.joint_states
    # joint_angles[-1] = -1.0  # Adjust the last value for desired pitch

    # Send joint angle command
    # arm_joint_move_command = arm_command_pb2.ArmCommand.ArmJointMoveCommand(
    #    trajectory_points=[
    #        trajectory_pb2.ArmJointTrajectoryPoint(
    #            position=joint_angles,
    #        )
    #    ]
    # )
    # arm_cmd_id = spot.command_client.robot_command(arm_joint_move_command)
    # block_until_arm_arrives(spot.command_client, arm_cmd_id, skill_timeout(duration))

    # Send arm joint command


#     robot_command_client = spot.robot.ensure_client(
#         RobotCommandClient.default_service_name
#     )
#     # Build the arm command.
#     cmd = RobotCommandBuilder.arm_carry_command()
#     # Send the request.
#     cmd_id = robot_command_client.robot_command(cmd)
#     # Wait until the arm arrives at the goal.
#     block_until_arm_arrives(robot_command_client, cmd_id, skill_timeout(duration))

#     # Define the desired pose relative to the body frame
#     x = 0.3  # Slightly in front of the robot
#     y = 0.3  # To the left of the robot
#     z = 0.0  # At the same height as carry position
#     rotation
#     rotation = Quat.from_yaw(math.radians(45))  # Rotate 45 degrees to the left

#     hand_pose = math_helpers.SE3Pose(x, y, z, rotation)

#     # Create the arm command
#     arm_cartesian_command = arm_command_pb2.ArmCommand.ArmCartesianCommand(
#         root_frame_name="body",
#         pose_trajectory_in_task=trajectory_pb2.SE3Trajectory(points=[
#             trajectory_pb2.SE3TrajectoryPoint(
#                 pose=hand_pose.to_proto()#, time_since_reference=duration_pb2.Duration()
#             )
#         ])
#     )
#     arm_cmd = arm_command_pb2.ArmCommand(arm_cartesian_command=arm_cartesian_command)
#     arm_cmd_id = robot_command_client.robot_command(arm_cmd)
#     block_until_arm_arrives(robot_command_client, arm_cmd_id, skill_timeout(duration))

#     return True

# def drop_object(spot, duration: float = 2.0) -> None:
#     #get current joint angles

#     robot_state = spot.state_client.get_robot_state()
#     joint_angles = robot_state.kinematic_state.joint_angles
#     joint_angles = [0.0, -0.5, 1.5, 0.0, -1.0, -0.5]  # Adjust the last value for desired pitch

#     # Create the arm command
#     arm_joint_move_command = arm_command_pb2.ArmCommand.ArmJointMoveCommand(
#         trajectory_points=[
#             trajectory_pb2.ArmJointTrajectoryPoint(
#                 position=joint_angles,
#                 # time_since_reference=time_utils.duration_to_duration_proto(0)
#             )
#         ]
#     )

#     # Create the full robot command
#     command = RobotCommandBuilder.build_synchro_command(
#         arm_command_pb2.ArmCommand(arm_joint_move_command=arm_joint_move_command)
#     )

#     # Send the command
#     command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)
#     cmd_id = command_client.robot_command(command)


def change_gripper(spot, fraction: float, duration: float = 2.0) -> None:
    """Change the spot gripper angle."""

    assert 0.0 <= fraction <= 1.0
    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    # Build the command.
    cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(fraction)
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Gripper commands have gripper feedback, never arm feedback.
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        reply = robot_command_client.robot_command_feedback(cmd_id)
        feedback = reply.feedback.synchronized_feedback.gripper_command_feedback
        if feedback.HasField("claw_gripper_feedback"):
            status = feedback.claw_gripper_feedback.status
            completed = status == gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_AT_GOAL
            contacting = (fraction == 0 and status ==
                          gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_APPLYING_FORCE)
            if completed or contacting:
                return
        time.sleep(0.05)
    raise RuntimeError("gripper did not reach the requested state")


def open_gripper(spot, duration: float = 2.0) -> None:
    """Open the spot gripper."""

    return change_gripper(spot, fraction=1.0, duration=duration)


def close_gripper(spot, duration: float = 2.0) -> None:
    """Close the spot gripper."""

    return change_gripper(spot, fraction=0.0, duration=duration)
