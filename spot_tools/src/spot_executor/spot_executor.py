import threading
import time
from contextlib import nullcontext

import numpy as np
import skimage as ski
from bosdyn.api.robot_state_pb2 import BehaviorFault
from bosdyn.client.exceptions import LeaseUseError
from bosdyn.client.robot_command import BehaviorFaultError
from robot_executor_interface.action_descriptions import (
    Follow,
    Gaze,
    Pick,
    Place,
    Carry,
    Stow,
)
from scipy.spatial.transform import Rotation

from spot_skills.arm_utils import gaze_at_vision_pose,arm_to_carry
from spot_skills.grasp_utils import object_grasp, object_place, stow_arm
from spot_skills.navigation_utils import (
    follow_trajectory_continuous,
    turn_to_point,
    wait_for_navigation,
)


def _report_action_result(feedback, command, action_index, status, detail=""):
    """Executor -> planner return channel (PR B5), duck-typed on feedback.

    A feedback collector that implements ``action_result(command,
    action_index, status, detail)`` (RosFeedbackCollector publishes an
    ActionResultMsg) gets each action's outcome; collectors that don't (the
    pyplot one, test fakes) are silently unaffected. The report must never
    break execution — the action already ran.
    """
    hook = getattr(feedback, "action_result", None)
    if hook is None:
        return
    try:
        hook(command, action_index, status, detail)
    except Exception as e:  # noqa: BLE001 -- reporting is best-effort
        feedback.print("INFO", f"action_result feedback failed: {e}")


def transform_command_frame(tf_trans, tf_q, command, feedback=None):
    # command is Nx3 numpy array

    R = Rotation.from_quat([tf_q.x, tf_q.y, tf_q.z, tf_q.w])
    _, _, yaw = R.as_euler("xyz", degrees=False)

    for ix in range(len(command)):
        c = command[ix]
        x, y = c[0:2]

        command[ix, 0] = np.cos(yaw) * x - np.sin(yaw) * y + tf_trans[0]
        command[ix, 1] = np.sin(yaw) * x + np.cos(yaw) * y + tf_trans[1]
        command[ix, 2] += yaw
    return command


# The lease manager should run in a separate thread to handle the exchange
# of the lease e.g., when the tablet takes control of the robot.
class LeaseManager:
    def __init__(self, spot_interface, feedback=None):
        self.spot_interface = spot_interface
        self.monitoring_thread = None
        self.feedback = feedback

        self.stopping = threading.Event()
        self.error = None
        self.taking_back_lease = False

        leases = self.spot_interface.lease_client.list_leases()
        self.owner = leases[0].lease_owner
        self.owner_name = self.owner.client_name
        self.initialize_thread()

    def close(self):
        self.stopping.set()
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=6)

    def initialize_thread(self):
        def monitor_lease():
            while not self.stopping.is_set():
                try:
                    leases = self.spot_interface.lease_client.list_leases()

                    # owner of the full lease
                    self.owner = leases[0].lease_owner
                    self.owner_name = self.owner.client_name

                    # If nobody owns the lease, then the owner string is empty.
                    # We should try to take the lease back in that case.
                    if self.owner_name == "":
                        # We should set the feedback's break_out_of_waiting_loop to True
                        # so that the pick skill gets immediately cancelled if it is running.
                        if self.feedback is not None:
                            self.feedback.break_out_of_waiting_loop = True

                        self.taking_back_lease = True
                        if self.feedback is not None:
                            self.feedback.print(
                                "INFO",
                                "LEASE MANAGER THREAD: Trying to take lease back, since nobody owns it.",
                            )
                            self.feedback.log_lease_takeover("spot_executor_takes_lease")
                        self.spot_interface.take_lease()
                        try:
                            stow_arm(self.spot_interface)
                            self.spot_interface.stand()
                        except BehaviorFaultError:
                            fault_ids = []
                            for fault in (
                                self.spot_interface.get_state().behavior_fault_state.faults
                            ):
                                if fault.cause == BehaviorFault.CAUSE_LEASE_TIMEOUT:
                                    fault_ids.append(fault.behavior_fault_id)
                            for fault_id in fault_ids:
                                if self.feedback is not None:
                                    self.feedback.print(
                                        "INFO",
                                        f"LEASE MANAGER THREAD: Clearing behavior fault {fault_id}",
                                    )
                                self.spot_interface.command_client.clear_behavior_fault(
                                    fault_id
                                )

                            if (
                                len(
                                    self.spot_interface.get_state().behavior_fault_state.faults
                                )
                                == 0
                            ):
                                self.spot_interface.stand()
                            else:
                                if self.feedback is not None:
                                    self.feedback.print(
                                        "WARN",
                                        "LEASE MANAGER THREAD: Could not clear all behavior faults, cannot stand.",
                                    )
                        self.stopping.wait(1)
                        if self.feedback is not None:
                            self.feedback.break_out_of_waiting_loop = False
                        self.taking_back_lease = False
                    self.error = None
                except Exception as exc:
                    # A lost endpoint or unsupported recovery action must not kill
                    # the guard monitor or leave a stale claimed lease owner.
                    self.owner_name = ""
                    self.error = str(exc)
                    if self.feedback is not None:
                        self.feedback.break_out_of_waiting_loop = True
                        self.feedback.print("ERROR", f"Lease monitor: {exc}")
                finally:
                    self.taking_back_lease = False
                self.stopping.wait(0.5)

        self.monitoring_thread = threading.Thread(target=monitor_lease, daemon=True)
        self.monitoring_thread.start()


class SpotExecutor:
    def __init__(
        self,
        spot_interface,
        detector,
        transform_lookup,
        planner,
        follower_lookahead=2,
        goal_tolerance=2.8,
        feedback=None,
        use_fake_path_planner=False,
        follow_timeout_per_meter=6.0,
        pick_image_source="frontleft_fisheye_image",
    ):
        self.debug = False
        self.spot_interface = spot_interface
        self.transform_lookup = transform_lookup
        self.follower_lookahead = follower_lookahead
        self.goal_tolerance = goal_tolerance
        self.detector = detector
        self.pick_image_source = pick_image_source
        self.keep_going = True
        self.cancel_event = threading.Event()
        self.processing_action_sequence = False
        self.mid_level_planner = planner
        self.use_fake_path_planner = use_fake_path_planner
        self.follow_timeout_per_meter = float(follow_timeout_per_meter)
        if self.follow_timeout_per_meter <= 0:
            raise ValueError("follow_timeout_per_meter must be positive")

        self.lease_manager = None

    def initialize_lease_manager(self, feedback):
        self.lease_manager = LeaseManager(self.spot_interface, feedback)

    def terminate_sequence(self, feedback):
        # Tell the actions sequence to break
        self.keep_going = False
        self.cancel_event.set()

        # Blocking the thread so that it terminates cleanly by
        # terminating the pick action and waiting for processing to end
        feedback.break_out_of_waiting_loop = True

        # Block until action sequence is done executing
        while self.processing_action_sequence:
            feedback.print(
                "INFO",
                "Waiting for previous action sequence to terminate. You must release the lease!",
            )
            time.sleep(1)

    def process_action_sequence(self, sequence, feedback):
        self.processing_action_sequence = True
        self.keep_going = not self.cancel_event.is_set()

        try:
            feedback.print("INFO", "Would like to execute: ")
            for command in sequence.actions:
                feedback.print("INFO", command)

            self.spot_interface.robot.time_sync.wait_for_sync()
            self.spot_interface.take_lease()

            ix = 0
            inner_loop_attempts = 0
            while ix < len(sequence.actions) and not self.cancel_event.is_set():
                # If the lease manager is actively taking back the lease and getting the
                # robot to stand back up, we don't want to send it any commands. It will break.
                if (
                    self.lease_manager is not None
                    and self.lease_manager.taking_back_lease
                ):
                    feedback.print(
                        "INFO",
                        "Waiting for lease manager to finish taking back lease...",
                    )
                    time.sleep(1)
                    continue

                # If we don't own the lease, we don't try to take any actions
                if (
                    self.lease_manager is not None
                    and not self.lease_manager.owner_name.startswith("understanding")
                ):
                    time.sleep(0.5)
                    continue

                command = sequence.actions[ix]

                if not self.keep_going or feedback.break_out_of_waiting_loop:
                    feedback.print("INFO", "Action sequence was pre-empted.")
                    _report_action_result(
                        feedback, command, ix, "PREEMPTED",
                        "sequence pre-empted before this action completed",
                    )
                    break
                pick_next = False
                if ix < len(sequence.actions) - 1:
                    pick_next = type(sequence.actions[ix + 1]) is Pick
                feedback.print("INFO", "\n")
                feedback.print("INFO", "Spot executor executing command: ")
                feedback.print("INFO", command)

                success = False
                try:
                    if type(command) is Follow:
                        success = self.execute_follow(command, feedback)
                        feedback.print(
                            "INFO", f"Finished `follow` command with return {success}"
                        )

                    elif type(command) is Gaze:
                        success = self.execute_gaze(
                            command, feedback, pick_next=pick_next
                        )

                    elif type(command) is Pick:
                        success = self.execute_pick(command, feedback)

                    elif type(command) is Place:
                        success = self.execute_place(command, feedback)

                    elif type(command) in (Carry,Stow):
                        skill_name=type(command).__name__.lower()
                        feedback.print("INFO", f"Executing `{skill_name}` command")
                        operation=arm_to_carry if type(command) is Carry else stow_arm
                        success=operation(self.spot_interface,duration=30.)
                        if feedback.break_out_of_waiting_loop:success=False
                        feedback.print("INFO", f"Finished `{skill_name}` command with return {success}")

                    else:
                        raise Exception(
                            f"SpotExecutor received unknown command type {type(command)}"
                        )
                    if success or inner_loop_attempts > 1:
                        # The action is being advanced past: either it
                        # succeeded, or it exhausted its retries (advanced
                        # with success=False). Report it on the executor ->
                        # planner return channel (PR B5).
                        _report_action_result(
                            feedback, command, ix,
                            "SUCCESS" if success else "FAILED",
                            "" if success
                            else f"gave up after {inner_loop_attempts + 1} attempts",
                        )
                        if not success:
                            # Dependent actions cannot execute after their prerequisite
                            # failed. Replanning requires a newly reviewed sequence.
                            break
                        ix += 1
                        inner_loop_attempts = 0
                    else:
                        inner_loop_attempts += 1
                        time.sleep(1)

                except LeaseUseError:
                    feedback.print("INFO", "Lost lease, stopping action sequence.")
                    feedback.log_lease_takeover("manual_intervention")
                    # Wait until the lease manager has taken the lease back
                    time.sleep(2)

        except Exception as ex:
            if "command" in locals():
                _report_action_result(feedback, command, ix,
                    "PREEMPTED" if feedback.break_out_of_waiting_loop else "FAILED", str(ex))
            self.processing_action_sequence = False
            raise ex

        self.processing_action_sequence = False

    def execute_gaze(self, command, feedback, pick_next=False):
        from scipy.spatial.transform import Rotation
        translation, rotation = self.transform_lookup("<spot_vision_frame>", command.frame)
        gaze_point = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).apply(command.gaze_point) + np.asarray(translation)
        feedback.print("INFO", "Executing `gaze` command")
        current_pose = self.spot_interface.get_pose()
        if feedback.break_out_of_waiting_loop:
            return False
        # Pickup navigation already selected the body stance and heading.
        # Gaze prepares the hand view without adding an unplanned body sweep
        # into the object's support. Standalone inspection keeps its body turn.
        if not pick_next:
            command_id=turn_to_point(self.spot_interface, current_pose, gaze_point)
            if not wait_for_navigation(self.spot_interface,command_id,
                cancelled=lambda:feedback.break_out_of_waiting_loop):
                return False
        if feedback.break_out_of_waiting_loop:
            return False
        # stow_after = command.stow_after
        stow_after = not pick_next
        success = gaze_at_vision_pose(
            self.spot_interface, gaze_point, stow_after=stow_after
        )
        feedback.gaze_feedback(current_pose, command.gaze_point)
        feedback.print("INFO", f"Finished `gaze` command with return {success}")
        return success

    def execute_pick(self, command, feedback):
        feedback.print("INFO", "Executing `pick` command")

        symbolic_grasp = getattr(self.spot_interface, "execute_symbolic_grasp", None)
        if callable(symbolic_grasp):
            success = symbolic_grasp()
        else:
            session = getattr(self.spot_interface, "pick_camera_session", nullcontext)
            with session():
                success = object_grasp(
                    self.spot_interface,
                    self.detector,
                    image_source=self.pick_image_source,
                    user_input=False,
                    semantic_class=command.object_class,
                    feedback=feedback,
                )

        if self.debug and not callable(symbolic_grasp):
            success, debug_images = success
            sem_img = ski.util.img_as_ubyte(debug_images[0])
            feedback.print(
                "INFO",
                "looking for classes: ",
                self.spot_interface.labelspace_map[command.object_class],
            )
            feedback.print("INFO", "unique semantic labels: ", np.unique(sem_img))
            outline_img = ski.util.img_as_ubyte(debug_images[1])

            feedback.pick_image_feedback(sem_img, outline_img)

        if success:
            # Update object holding state
            feedback.set_robot_holding_state(True, command.object_id.upper())

        feedback.print("INFO", f"Finished `pick` command with return {success}")
        feedback.print("INFO", f"Pick skill success: {success}")
        return success

    def execute_place(self, command, feedback):
        feedback.print("INFO", "Executing `place` command")
        placement = getattr(feedback, "placement_feedback", None)
        position = placement(command.object_class) if placement else None
        verifier = getattr(feedback, "placement_verifier", None)
        verification = verifier.prepare(command.object_class, position) if verifier else None
        success = object_place(self.spot_interface, semantic_class=command.object_class,
                               position=position, cancelled=lambda: feedback.break_out_of_waiting_loop)

        if success:
            # Update object holding state
            feedback.set_robot_holding_state(False, command.object_id.upper())
            if verifier:
                receipt = verifier.verify(verification)
                feedback.print("INFO", f"Placement observation verification: {receipt}")

        feedback.print("INFO", f"Finished `place` command with return {success}")
        return success

    def execute_follow(self, command, feedback):
        feedback.print("INFO", "Executing `follow` command")
        feedback.print(
            "INFO", f"transforming path from {command.frame} to <spot_vision_frame>"
        )
        # <spot_vision_frame> gets remapped to the actual robot odom frame name
        # by the transform_lookup function.
        t, r = self.transform_lookup("<spot_vision_frame>", command.frame)
        command_to_send = transform_command_frame(
            t, r, command.path2d.copy(), feedback=feedback
        )

        path_distance = np.sum(
            np.linalg.norm(np.diff(command_to_send[:, :2], axis=0), axis=1)
        )
        from spot_skills.navigation_utils import navigation_timeout
        timeout = navigation_timeout(command_to_send, self.follow_timeout_per_meter)
        feedback.print(
            "INFO",
            f"Using continous follower with params:\n\tlookahead: {self.follower_lookahead}\n\tgoal tolerance: {self.goal_tolerance}\n\ttimeout: {timeout}",
        )

        feedback.follow_path_feedback(command_to_send)

        if self.mid_level_planner is not None and self.use_fake_path_planner:
            # this only publish the path but does not actually command the spot to follow it
            # TODO: need to refactor this part
            ret = False
            mlp_success, planning_output = self.mid_level_planner.plan_path(
                command_to_send[:, :2]
            )
            path_wp = planning_output.path_waypoints_metric
            target_point_metric = planning_output.target_point_metric
            if not mlp_success:
                feedback.print("INFO", "Mid-level planner failed to find a path")
            if target_point_metric is not None:
                feedback.path_follow_MLP_feedback(path_wp, target_point_metric)
        else:
            ret = follow_trajectory_continuous(
                self.spot_interface,
                command_to_send,
                self.follower_lookahead,
                self.goal_tolerance,
                timeout,
                self.mid_level_planner,
                feedback=feedback,
            )
        return ret
