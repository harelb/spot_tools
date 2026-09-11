"""
Grasping an object with the arm (likely a backpack)
"""

import argparse
import sys
import time
from copy import copy

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import (
    geometry_pb2,
    manipulation_api_pb2,
    robot_state_pb2,
)
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives

from spot_skills.arm_utils import (
    close_gripper,
    move_hand_to_relative_pose,
    open_gripper,
    stow_arm,
)
from spot_skills.detection_utils import DetectionCandidate, Detector
from spot_skills.primitives import execute_recovery_action

g_image_click = None
g_image_display = None


def wait_until_grasp_state_updates(grasp_override_command, robot_state_client):
    updated = False
    has_grasp_override = grasp_override_command.HasField("api_grasp_override")
    has_carry_state_override = grasp_override_command.HasField("carry_state_override")

    while not updated:
        robot_state = robot_state_client.get_robot_state()

        grasp_state_updated = (
            robot_state.manipulator_state.is_gripper_holding_item
            and (
                grasp_override_command.api_grasp_override.override_request
                == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING
            )
        ) or (
            not robot_state.manipulator_state.is_gripper_holding_item
            and grasp_override_command.api_grasp_override.override_request
            == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_NOT_HOLDING
        )
        carry_state_updated = has_carry_state_override and (
            robot_state.manipulator_state.carry_state
            == grasp_override_command.carry_state_override.override_request
        )
        updated = (not has_grasp_override or grasp_state_updated) and (
            not has_carry_state_override or carry_state_updated
        )
        time.sleep(0.1)


def force_stow_arm(manipulation_client, state_client, command_client):
    # grasp_override = manipulation_api_pb2.ApiGraspOverride()
    # grasp_override.override = manipulation_api_pb2.ApiGraspOverride.CARRIABLE_AND_STOWABLE

    ## Build the stow command with the grasp override
    # stow_command = RobotCommandBuilder.arm_stow_command()
    # stow_command.synchronized_command.arm_command.arm_cartesian_command.root_frame_name = 'body'
    # stow_command.synchronized_command.arm_command.arm_cartesian_command.wrist_tform_tool.rotation.w = 1
    # stow_command.synchronized_command.gripper_command.grasp_override.CopyFrom(grasp_override)

    ## Send the command
    # command_id = client.robot_command(stow_command)

    ## Optionally, you can block until the command is complete
    ##client.robot_command_feedback(command_id)

    ## Wait until the arm arrives at the goal.
    # block_until_arm_arrives(client, command_id, 2.0)

    carriable_and_stowable_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
        override_request=robot_state_pb2.ManipulatorState.CARRY_STATE_CARRIABLE_AND_STOWABLE
    )

    grasp_holding_override = manipulation_api_pb2.ApiGraspOverride(
        override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING
    )

    override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
        api_grasp_override=grasp_holding_override,
        carry_state_override=carriable_and_stowable_override,
    )
    manipulation_client.grasp_override_command(override_request)
    # Wait for the override to take effect before trying to move the arm.
    wait_until_grasp_state_updates(override_request, state_client)

    robot_cmd = RobotCommandBuilder.arm_stow_command()
    cmd_id = command_client.robot_command(robot_cmd)
    block_until_arm_arrives(command_client, cmd_id)



def place_at_point(spot, position, cancelled):
    """Stationary-base placement; never release if positioning fails."""
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME
    point = np.asarray(position, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("invalid placement target")
    if spot.is_fake:
        return not cancelled()
    state = spot.state_client.get_robot_state()
    if not state.manipulator_state.is_gripper_holding_item:
        raise RuntimeError("robot is not holding an object")
    from bosdyn.client.frame_helpers import get_a_tform_b, HAND_FRAME_NAME
    orientation = get_a_tform_b(state.kinematic_state.transforms_snapshot,
                               BODY_FRAME_NAME, HAND_FRAME_NAME).rot
    def move(target):
        if cancelled():
            raise RuntimeError("placement cancelled")
        cmd = RobotCommandBuilder.arm_pose_command(
            *map(float, target), orientation.w, orientation.x, orientation.y,
            orientation.z, BODY_FRAME_NAME, 2.0)
        command_id = spot.command_client.robot_command(cmd)
        if not block_until_arm_arrives(spot.command_client, command_id, 5.0):
            raise RuntimeError("arm did not reach approved placement pose")
        if cancelled():
            raise RuntimeError("placement cancelled")
    above = point + [0, 0, .1]
    move(above)
    move(point)
    if cancelled():
        raise RuntimeError("placement cancelled before release")
    open_gripper(spot)
    move(above)
    stow_arm(spot)
    close_gripper(spot)
    return True

def object_place(spot, semantic_class="bag", position=None, cancelled=lambda: False):
    """Place at an approved body-frame point, or retain legacy drop behavior."""
    if position is not None:
        return place_at_point(spot, position, cancelled)

    if spot.is_fake:
        return True

    time.sleep(0.25)

    carry_cmd = RobotCommandBuilder.arm_carry_command()
    spot.command_client.robot_command(carry_cmd)
    time.sleep(1)

    open_gripper(spot)
    stow_arm(spot)
    close_gripper(spot)
    execute_recovery_action(
        spot,
        recover_arm=False,
        relative_pose=math_helpers.SE2Pose(x=-1.0, y=0.0, angle=0),
    )
    time.sleep(1)
    execute_recovery_action(
        spot,
        recover_arm=False,
        relative_pose=math_helpers.SE2Pose(x=0.0, y=-1.0, angle=0),
    )
    time.sleep(1)
    return True


def object_grasp(
    spot,
    detector,
    image_source="hand_color_image",
    user_input=False,
    semantic_class="bag",
    grasp_constraint=None,
    debug=False,
    feedback=None,
    carry_high=True,
):
    debug_images = []

    """Using the Boston Dynamics API to command Spot's arm"""
    if not isinstance(detector, Detector):
        raise Exception("You need to define a valid detector to pick up an object.")

    print(f'Grasping object of class "{semantic_class}"')
    if spot.is_fake:
        semantic_class = "bag"

    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client

    attempts = 0
    success = False

    # Set up the detector (e.g., for YOLOWorld, this may mean updating recognized classes)
    detector.set_up_detector(semantic_class)

    candidates = None
    while attempts < 2 and not success:
        attempts += 1

        if not user_input:
            # Try to get the centroid using the detector passed into the function.
            detection_index, candidates = detector.return_centroid(
                image_source, semantic_class, debug=debug
            )

            # If the detector fails to return the centroid, then try again until max_attempts
            if detection_index is None:
                continue

            else:
                success = True

        else:
            image, img = spot.get_image_RGB(view=image_source)
            xy = get_user_grasp_input(spot, img)
            candidates = [DetectionCandidate(image, img, xy)]
            detection_index = 0
            print("Found object centroid:", xy)

    if candidates is None:
        if feedback is not None:
            feedback.print(
                "INFO",
                "Failed to capture any camera images. Please check the detector or user input.",
            )
        execute_recovery_action(
            spot,
            recover_arm=False,
            relative_pose=math_helpers.SE2Pose(
                x=0.0, y=0.0, angle=np.random.choice([-0.5, 0.5])
            ),
        )
        time.sleep(1)
        return False

    # Display all candidate images in the approval panel
    if feedback is not None:
        cv2_images = [copy(c.cv2_image) for c in candidates]

        if detection_index is not None:
            xy = candidates[detection_index].detection_xy
            det_x = xy[0] if xy else None
            det_y = xy[1] if xy else None
        else:
            det_x = None
            det_y = None

        approved, updated_xy, selected_index = feedback.bounding_box_detection_feedback(
            cv2_images,
            detection_index,
            det_x,
            det_y,
            semantic_class,
        )

        if approved is not None and not approved:
            feedback.print("INFO", "User requested abort.")
            return False

        # Use selected camera image and pixel (panel always sends the correct selection)
        image = candidates[selected_index].bosdyn_image
        xy = updated_xy
    else:
        if detection_index is None:
            execute_recovery_action(
                spot,
                recover_arm=False,
                relative_pose=math_helpers.SE2Pose(
                    x=0.0, y=0.0, angle=np.random.choice([-0.5, 0.5])
                ),
            )
            time.sleep(1)
            return False
        image = candidates[detection_index].bosdyn_image
        xy = candidates[detection_index].detection_xy

    pick_vec = geometry_pb2.Vec2(x=xy[0], y=xy[1])
    stow_arm(spot)

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(grasp_constraint, grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request
    )

    loop_timer = time.time()

    # Reset success --> agent is successful only if it detects the object and picks it up
    success = False
    # Get feedback from the robot
    while True:
        current_time = time.time()
        if current_time - loop_timer > 15:
            if feedback is not None:
                feedback.print("INFO", "The pick skill timed out!")
            print("The pick skill timed out!")
            break
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        current_state = manipulation_api_pb2.ManipulationFeedbackState.Name(
            response.current_state
        )
        print(f"Current state: {current_state}")

        failed_states = [
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE,
        ]

        if response.current_state in failed_states:
            if feedback is not None:
                feedback.print("INFO", "GRASP FAILED")
            break

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            success = True
            break

    if feedback is not None:
        current_state = manipulation_api_pb2.ManipulationFeedbackState.Name(
            response.current_state
        )
        feedback.print("INFO", f"POST-LOOP STATE: {current_state}")

    # If the robot was not successful, send it back to the location it started the skill at
    if not success:
        execute_recovery_action(
            spot,
            recover_arm=True,
            relative_pose=math_helpers.SE2Pose(
                x=np.random.uniform(-1, -0.5), y=0.0, angle=0.0
            ),
        )
        if debug:
            return success, debug_images
        return success

    close_cmd = RobotCommandBuilder.claw_gripper_close_command(
        build_on_command=None,
        max_acc=None,
        max_vel=None,
        disable_force_on_contact=False,
        max_torque=20,
    )
    spot.command_client.robot_command(close_cmd)
    time.sleep(0.25)

    # Move the arm to a carry position.
    print("Grasp finished, carrying object.")
    carry_cmd = RobotCommandBuilder.arm_carry_command()
    spot.command_client.robot_command(carry_cmd)
    time.sleep(1)

    print("Force stowing arm!")
    force_stow_arm(manipulation_api_client, robot_state_client, spot.command_client)
    time.sleep(1)

    if carry_high:
        print("Carrying high")
        body_tform_goal = math_helpers.SE3Pose(
            x=0.6, y=0.0, z=0.6, rot=math_helpers.Quat()
        )
        move_hand_to_relative_pose(spot, body_tform_goal)
        time.sleep(1)

    print("Finished grasp.")

    if debug:
        return success, debug_images
    return success


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = "Click to grasp"
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def get_user_grasp_input(spot, img):
    """Show the image to the user and wait for them to click on a pixel"""

    spot.robot.logger.info("Click on an object to start grasping...")
    image_title = "Click to grasp"
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    global g_image_click, g_image_display
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            # Quit
            print('"q" pressed, exiting.')
            exit(0)
    spot.robot.logger.info(
        f"Picking object at image location ({g_image_click[0]}, {g_image_click[1]})"
    )
    spot.robot.logger.info(
        "Picking object at image location (%s, %s)", g_image_click[0], g_image_click[1]
    )
    return g_image_click


def add_grasp_constraint(grasp_constraint, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = False
    if (
        grasp_constraint == "force_top_down_grasp"
        or grasp_constraint == "force_horizontal_grasp"
    ):
        use_vector_constraint = True

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if grasp_constraint == "force_top_down_grasp":
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        elif grasp_constraint == "force_horizontal_grasp":
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        else:
            assert False, "Invalid grasp constraint"

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo
        )

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif grasp_constraint == "force_45_angle_grasp":
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif grasp_constraint == "force_squeeze_grasp":
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

    elif grasp_constraint is not None:
        assert False, "Invalid grasp constraint"


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-i",
        "--image-source",
        help="Get image from source",
        default="frontleft_fisheye_image",
    )
    parser.add_argument(
        "-t",
        "--force-top-down-grasp",
        help="Force the robot to use a top-down grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force-horizontal-grasp",
        help="Force the robot to use a horizontal grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--force-45-angle-grasp",
        help="Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--force-squeeze-grasp",
        help="Force the robot to use a squeeze grasp",
        action="store_true",
    )
    options = parser.parse_args()

    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print("Error: cannot force more than one type of grasp.  Choose only one.")
        sys.exit(1)

    try:
        # Assuming object_grasp is the correct function to call
        object_grasp(options)
        return True
    except Exception:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
