import logging
import os
import threading
import time
import traceback

import numpy as np
import rclpy
import rclpy.time
import spot_executor as se
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point as PointMsg
from heracles_ros_interfaces.srv import UpdateHoldingState
from nav_msgs.msg import Path
from nlu_interface_rviz.msg import (
    ManipulationApprovalRequest,
    ManipulationApprovalResponse,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface_ros.action_descriptions_ros import from_msg
from robot_executor_msgs.msg import (
    ActionDoneMsg,
    ActionMsg,
    ActionResultMsg,
    ActionSequenceMsg,
    RuntimeGuardsMsg,
)
from ros_system_monitor_msgs.msg import NodeInfoMsg
from sensor_msgs.msg import Image
from shapely.geometry import Point
from spot_executor.fake_spot import FakeSpot
from spot_executor.spot import Spot
from spot_skills.detection_utils import YOLODetector
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from robot_executor_interface.action_descriptions import Follow, Gaze, Pick, Place
from robot_executor_interface.mid_level_planner import (
    IdentityPlanner,
    MidLevelPlanner,
    OccupancyMap,
)
from spot_tools_ros.fake_spot_ros import FakeSpotRos
from spot_tools_ros.occupancy_grid_ros_updater import OccupancyGridROSUpdater
from spot_tools_ros.utils import get_tf_pose, waypoints_to_path


def load_inverse_semantic_id_map_from_label_space(fn):
    with open(fn, "r") as fo:
        labelspace_yaml = yaml.safe_load(fo)
    label_to_id = {e["name"]: e["label"] for e in labelspace_yaml["label_names"]}
    return label_to_id


def pt_to_marker(pt, ns, mid, color, fid="vision"):
    m = Marker()
    m.header.frame_id = fid
    m.header.stamp = rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()
    m.ns = ns
    m.id = mid
    m.type = m.SPHERE
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.pose.position.x = pt.x
    m.pose.position.y = pt.y
    m.scale.x = 0.6
    m.scale.y = 0.6
    m.scale.z = 0.6
    m.color.a = 1.0
    m.color.r = float(color[0])
    m.color.g = float(color[1])
    m.color.b = float(color[2])

    return m


def parse_robot_color(value):
    """``robot_color`` param -> (r, g, b) floats, or None for the default colors.

    Accepts "#rrggbb", "rrggbb" or a "prefix:#rrggbb" form (the generated params
    use "color:#rrggbb" so the substituted hex is not read as a YAML comment).
    Anything else ("", "none", "color:none", garbage) means defaults.
    """
    h = str(value or "").rpartition(":")[2].strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None


def shade(rgb, k):
    """Scale an (r, g, b) color toward black (k < 1) or white (k > 1)."""
    if k <= 1.0:
        return [c * k for c in rgb]
    return [c + (1.0 - c) * (k - 1.0) for c in rgb]


def build_markers(pts, namespaces, frames, colors):
    ma = MarkerArray()
    for i, pt in enumerate(pts):
        m = pt_to_marker(pt, namespaces[i], i, colors[i], fid=frames[i])
        ma.markers.append(m)
    return ma


class RosFeedbackCollector:
    def __init__(
        self,
        odom_frame: str,
        output_dir: str,
        log_to_file_level: str = "",
        robot_color=None,
    ):
        self.pick_confirmation_event = threading.Event()
        # self.pick_confirmation_response = False

        self.pick_confirmation_approved = False
        self.pick_confirmation_xy = [0, 0]
        self.pick_confirmation_image_index = 0

        self.break_out_of_waiting_loop = False
        self.plan_valid = True
        self.odom_frame = odom_frame

        # Executor -> planner return channel (PR B5): publisher + the
        # dispatched sequence's identity, set by process_action_sequence
        # before execution starts so every per-action result can echo them.
        self.action_result_pub = None
        self._clock = None
        self.current_plan_id = ""
        self.current_robot_name = ""

        self.output_dir = output_dir
        self.log_to_file_level = log_to_file_level

        if log_to_file_level != "":
            os.makedirs(self.output_dir, exist_ok=True)
            self.str_log_file = os.path.join(self.output_dir, "str_log.txt")
            with open(self.str_log_file, "w") as f:
                f.write("time - level - message\n")

            match log_to_file_level:
                case "DEBUG":
                    logging_level = logging.DEBUG
                case "INFO":
                    logging_level = logging.INFO
                case "WARNING":
                    logging_level = logging.WARNING
                case "ERROR":
                    logging_level = logging.ERROR
                case _:
                    raise ValueError(f"Invalid log level {log_to_file_level}")

            # Create a custom logger
            self.file_logger = logging.getLogger(__name__)
            self.file_logger.setLevel(logging_level)

            # Create a file handler
            file_handler = logging.FileHandler(self.str_log_file)
            file_handler.setLevel(logging_level)

            # Create a formatter and add it to the handler
            formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            # Add the handler to the logger
            self.file_logger.addHandler(file_handler)

        # Path-following markers: progress point / lookahead target. With a robot
        # color the target is a darker shade of it, so several robots stay apart.
        if robot_color is None:
            self.progress_color, self.target_color = [0, 1, 1], [1, 0, 1]
            self.subgoal_color = [0, 0, 1]
        else:
            self.progress_color = list(robot_color)
            self.target_color = shade(robot_color, 0.55)
            self.subgoal_color = shade(robot_color, 0.3)

    def bounding_box_detection_feedback(
        self,
        detection_imgs,
        source_names,
        detection_index,
        centroid_x,
        centroid_y,
        semantic_class,
    ):
        bridge = CvBridge()

        request_msg = ManipulationApprovalRequest()
        request_msg.images = [
            bridge.cv2_to_imgmsg(img, encoding="passthrough") for img in detection_imgs
        ]
        request_msg.image_source_names = source_names
        request_msg.has_detection = detection_index is not None
        request_msg.detection_image_index = (
            detection_index if detection_index is not None else 0
        )
        request_msg.image_x = centroid_x if centroid_x is not None else 0
        request_msg.image_y = centroid_y if centroid_y is not None else 0
        self.detection_img_pub.publish(request_msg)

        self.pick_confirmation_event.clear()

        # Wait until input is received and self.pick_confirmation_response is set
        while (
            not self.break_out_of_waiting_loop
            and not self.pick_confirmation_event.is_set()
        ):
            self.logger.info("Waiting for user to confirm pick action...")
            self.pick_confirmation_event.wait(timeout=5)

        if self.break_out_of_waiting_loop:
            self.logger.info("ROBOT WAS PREEMPTED")
            self.pick_confirmation_approved = False
        else:
            self.logger.info(
                f"Pick Confirmation Response Received: approved ({self.pick_confirmation_approved}), xy ({self.pick_confirmation_xy}), image_index ({self.pick_confirmation_image_index})"
            )

        return (
            self.pick_confirmation_approved,
            self.pick_confirmation_xy,
            self.pick_confirmation_image_index,
        )

    def pick_image_feedback(self, semantic_image, mask_image):
        bridge = CvBridge()
        semantic_hand_msg = bridge.cv2_to_imgmsg(semantic_image, encoding="passthrough")
        mask_img_msg = bridge.cv2_to_imgmsg(mask_image, encoding="passthrough")
        self.semantic_hand_pub.publish(semantic_hand_msg)
        self.semantic_mask_pub.publish(mask_img_msg)

    def follow_path_feedback(self, path):
        path_debug_viz = waypoints_to_path("vision", path)
        self.smooth_path_publisher.publish(path_debug_viz)

    def path_following_progress_feedback(self, progress_point, target_point):
        pts = [progress_point, target_point]
        namespaces = ["path_progress"] * 2
        colors = [self.progress_color, self.target_color]
        frames = [self.odom_frame] * 2
        self.progress_point_pub.publish(build_markers(pts, namespaces, frames, colors))

    def path_follow_MLP_feedback(
        self,
        path,
        target_point_metric,
        target_point_global_traj_metric,
        subgoal_target_point_metric,
    ):
        self.mlp_path_publisher.publish(waypoints_to_path(self.odom_frame, path))
        if (
            target_point_metric is None
            or target_point_global_traj_metric is None
            or subgoal_target_point_metric is None
        ):
            return
        target_point_metric_flattened = Point([p[0] for p in target_point_metric[:3]])
        target_point_global_traj_metric_flattened = Point(
            [p[0] for p in target_point_global_traj_metric[:3]]
        )
        subgoal_target_point_metric_flattened = Point(subgoal_target_point_metric[:3])

        pts = [
            target_point_global_traj_metric_flattened,
            target_point_metric_flattened,
            subgoal_target_point_metric_flattened,
        ]
        namespaces = [
            "projected target point",
            "actual target point",
            "subgoal target point",
        ]
        colors = [self.target_color, self.progress_color, self.subgoal_color]
        frames = [self.odom_frame] * 3
        self.mlp_target_publisher.publish(
            build_markers(pts, namespaces, frames, colors)
        )

    def gaze_feedback(self, pose, gaze_point):
        pass

    def print(self, level, string):
        # rclpy keys its logging context on CallerId -- (function, file, line) --
        # and refuses to log a second severity from a call site it has already
        # seen. Binding log_fn and calling it from one shared line made every
        # level share a call site, so the first WARNING after an INFO raised
        # "Logger severity cannot be changed between calls." Each severity needs
        # its own line.
        message = str(string)
        match level:
            case "DEBUG":
                self.logger.debug(message)
            case "INFO":
                self.logger.info(message)
            case "WARNING":
                self.logger.warning(message)
            case "ERROR":
                self.logger.error(message)
            case _:
                raise ValueError(f"Invalid log level {level}")

        # TODO(multy): quick logic to log everything we print in the executor
        if self.log_to_file_level != "":
            match level:
                case "DEBUG":
                    file_logger_fn = self.file_logger.debug
                case "INFO":
                    file_logger_fn = self.file_logger.info
                case "WARNING":
                    file_logger_fn = self.file_logger.warning
                case "ERROR":
                    file_logger_fn = self.file_logger.error
                case _:
                    raise ValueError(f"Invalid log level {level}")
            file_logger_fn(str(string))

    def feedback_viz_2(self, y):
        pass

    def action_result(self, command, action_index, status, detail=""):
        """Publish one action's outcome (PR B5). Called by
        SpotExecutor._report_action_result; must never raise into the
        execution loop (the caller guards, this stays defensive anyway)."""
        if self.action_result_pub is None:
            return
        msg = ActionResultMsg()
        if self._clock is not None:
            msg.header.stamp = self._clock.now().to_msg()
        msg.robot_name = self.current_robot_name
        msg.plan_id = self.current_plan_id
        msg.action_type = type(command).__name__.upper()
        msg.action_index = int(action_index)
        msg.status = str(status)
        msg.detail = str(detail)
        self.action_result_pub.publish(msg)

    def register_publishers(self, node):
        self.logger = node.get_logger()
        self._clock = node.get_clock()

        # ROS 2 transient local QoS for "latching" behavior
        latching_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.smooth_path_publisher = node.create_publisher(
            Path, "~/smooth_path_publisher", qos_profile=latching_qos
        )

        self.progress_visualizer_pub = node.create_publisher(
            MarkerArray, "~/path_progress_visualizer", qos_profile=latching_qos
        )

        self.semantic_hand_pub = node.create_publisher(
            Image, "~/semantic_hand_image", qos_profile=latching_qos
        )

        self.semantic_mask_pub = node.create_publisher(
            Image, "~/semantic_mask_image", qos_profile=latching_qos
        )

        self.progress_point_pub = node.create_publisher(
            MarkerArray, "~/progress_point_visualizer", qos_profile=latching_qos
        )

        self.mlp_path_publisher = node.create_publisher(
            Path, "~/mlp_path_publisher", qos_profile=latching_qos
        )

        self.mlp_target_publisher = node.create_publisher(
            MarkerArray, "~/mlp_target_publisher", qos_profile=latching_qos
        )

        self.detection_img_pub = node.create_publisher(
            ManipulationApprovalRequest,
            "~/manipulation_request",
            qos_profile=latching_qos,
        )

        self.lease_takeover_publisher = node.create_publisher(String, "~/takeover", 10)

        # PR B5: per-action results back to the planner. Plain queue QoS —
        # consumers want the stream, not just the last value.
        self.action_result_pub = node.create_publisher(
            ActionResultMsg, "~/action_result", 10
        )

        node.create_subscription(
            ManipulationApprovalResponse,
            "~/pick_confirmation",
            self.pick_confirmation_callback,
            10,
        )

        self.holding_client = node.create_client(
            UpdateHoldingState, "update_holding_state"
        )

        # Fleet-wide, like the plan itself: the planner reads every robot's
        # progress from here.
        self.action_done_pub = node.create_publisher(ActionDoneMsg, "/action_done", 10)

        # TODO(aaron): Once we switch logging to python logger,
        # should move into init
        self.logger.info(f"Logging to: {self.output_dir}")
        if not os.path.exists(self.output_dir):
            self.logger.info(f"Making {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        log_fn = os.path.join(self.output_dir, "lease_log.txt")
        with open(log_fn, "w") as fo:
            fo.write("time,event\n")

    def report_action_done(self, sequence, command, success):
        msg = ActionDoneMsg()
        msg.header.stamp = self._clock.now().to_msg()
        msg.plan_id = sequence.plan_id or ""
        msg.robot_name = sequence.robot_name or ""
        msg.success = bool(success)
        msg.object_id = str(getattr(command, "object_id", "") or "")
        if isinstance(command, Follow):
            msg.action_type = ActionMsg.FOLLOW
            points = command.path2d
        elif isinstance(command, Gaze):
            msg.action_type = ActionMsg.GAZE
            points = []
        elif isinstance(command, Pick):
            msg.action_type = ActionMsg.PICK
            points = []
        elif isinstance(command, Place):
            msg.action_type = ActionMsg.PLACE
            points = [command.object_point]
        else:
            return
        msg.points = [
            PointMsg(
                x=float(p[0]),
                y=float(p[1]),
                z=float(p[2]) if len(p) > 2 else 0.0,
            )
            for p in points
        ]
        self.action_done_pub.publish(msg)

    def set_robot_holding_state(
        self, is_holding: bool, object_id: str, position=None, timeout=5
    ):
        req = UpdateHoldingState.Request()
        req.is_holding = is_holding
        req.id = object_id
        if position is not None:
            # Where the object was put down, in the map frame. Without it the
            # state updater falls back to the robot's own position.
            req.has_position = True
            req.position = PointMsg(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]) if len(position) > 2 else 0.0,
            )

        if not self.holding_client.wait_for_service(timeout_sec=0.5):
            self.logger.warning("UpdateHoldingState service not available")
            return False

        future = self.holding_client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                self.logger.error("UpdateHoldingState call timed out")
                return False

        return future.result().success

    def pick_confirmation_callback(self, msg):
        # if msg.data:
        #    self.logger.info("Detection is valid. Continuing pick action!")
        #    self.pick_confirmation_response = True
        # else:
        #    self.logger.warn("Detection is invalid. Discontinuing pick action.")
        #    self.pick_confirmation_response = False

        # self.pick_confirmation_event.set()

        # If not approved, discontinue
        # If approved, check whether the detection is overwritten
        if not msg.approve:
            self.logger.warn("Detection is invalid. Discontinuing pick action.")
            self.pick_confirmation_approved = False
        else:
            self.pick_confirmation_approved = True
            self.pick_confirmation_image_index = msg.image_index
            self.pick_confirmation_xy[0] = msg.image_x
            self.pick_confirmation_xy[1] = msg.image_y
            self.logger.warn("Detection is valid. Continuing pick action!")
        self.pick_confirmation_event.set()

    def log_lease_takeover(self, event: str):
        log_fn = os.path.join(self.output_dir, "lease_log.txt")
        t = time.time()
        with open(log_fn, "a") as fo:
            fo.write(f"{t},{event}\n")

        msg = String()
        msg.data = f"{t},{event}"
        self.lease_takeover_publisher.publish(msg)


def resolve_spot_interface(spot_interface: str, use_fake_spot_interface: bool) -> str:
    """Back-compat resolution: explicit spot_interface wins; the legacy
    use_fake_spot_interface flag maps to 'fake'."""
    if spot_interface not in ("", "real", "fake", "sim"):
        raise ValueError(f"Invalid spot_interface: {spot_interface}")
    if spot_interface:
        return spot_interface
    return "fake" if use_fake_spot_interface else "real"


def take_lease_report(lease_manager) -> tuple[bool, str]:
    """Force-take the lease via ``lease_manager``; ``(success, message)`` with the
    owner before and after.  Never raises (a service callback must not)."""
    try:
        before, after = lease_manager.force_take()
    except Exception as e:  # noqa: BLE001
        return False, f"take failed: {type(e).__name__}: {e}"
    ours = str(after).startswith("understanding")
    msg = f"lease owner before: {before or '(nobody)'}; after: {after or '(nobody)'}"
    return ours, msg if ours else f"{msg} (not ours after take)"


class SpotExecutorRos(Node):
    def __init__(self):
        super().__init__("spot_executor_ros")
        self.debug = False
        self.background_thread = None
        self._current_plan_id = None
        self._plan_id_lock = threading.Lock()

        # Connectivity parameters
        self.declare_parameter("spot_ip", "")
        self.declare_parameter("bosdyn_client_username", "")
        self.declare_parameter("bosdyn_client_password", "")
        spot_ip = self.get_parameter("spot_ip").value
        bdai_username = self.get_parameter("bosdyn_client_username").value
        bdai_password = self.get_parameter("bosdyn_client_password").value

        # Follow Skill
        self.declare_parameter("follower_lookahead", 0.0)
        follower_lookahead = self.get_parameter("follower_lookahead").value
        assert follower_lookahead > 0

        self.declare_parameter("goal_tolerance", 0.0)
        goal_tolerance = self.get_parameter("goal_tolerance").value
        assert goal_tolerance > 0
        self.get_logger().info(f"{goal_tolerance=}")
        self.declare_parameter("follow_timeout_per_meter", 6.0)
        follow_timeout_per_meter = self.get_parameter(
            "follow_timeout_per_meter").value
        assert follow_timeout_per_meter > 0

        # Pick/Inspect Skill
        self.declare_parameter("semantic_model_path", "")
        self.declare_parameter("labelspace_path", "")
        self.declare_parameter("labelspace_grouping_path", "")
        self.declare_parameter("detector_model_path", "")
        detector_model_path = self.get_parameter("detector_model_path").value
        # semantic_model_path = self.get_parameter("semantic_model_path").value
        # labelspace_path = self.get_parameter("labelspace_path").value
        # semantic_name_to_id = load_inverse_semantic_id_map_from_label_space(
        #    labelspace_path
        # )
        # labelspace_grouping_path = self.get_parameter("labelspace_grouping_path").value
        # with open(labelspace_grouping_path, "r") as f:
        #    grouping_info = yaml.safe_load(f)
        # turn list of dictionaries into single dictionary
        # self.labelspace_map = {}
        # offset = int(grouping_info["offset"])
        # for group in grouping_info["groups"]:
        #    self.labelspace_map[group["name"]] = [g + offset for g in group["labels"]]

        self.declare_parameter("odom_frame", "")
        odom_frame = self.get_parameter("odom_frame").value
        assert odom_frame != ""
        self.odom_frame = odom_frame

        self.declare_parameter("body_frame", "")
        body_frame = self.get_parameter("body_frame").value
        assert body_frame != ""
        self.body_frame = body_frame

        self.declare_parameter("output_dir", "")
        output_dir = self.get_parameter("output_dir").value
        assert output_dir != ""

        # "#rrggbb" (or "color:#rrggbb") from topology.yaml; empty/"none" = defaults.
        self.declare_parameter("robot_color", "")
        robot_color = parse_robot_color(self.get_parameter("robot_color").value)

        # TODO(multy): quick way to log everything feedback print to log file
        self.declare_parameter("log_to_file", "")
        log_to_file = self.get_parameter("log_to_file").value

        self.feedback_collector = RosFeedbackCollector(
            self.odom_frame, output_dir, log_to_file, robot_color=robot_color
        )
        self.feedback_collector.register_publishers(self)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Robot Initialization
        self.declare_parameter("use_fake_spot_interface", False)
        use_fake_spot_interface = self.get_parameter("use_fake_spot_interface").value
        self.declare_parameter("spot_interface", "")
        self.declare_parameter("sim_rgb_topic", "")
        interface = resolve_spot_interface(
            self.get_parameter("spot_interface").value, use_fake_spot_interface
        )

        # mid-level planner parameters
        self.declare_parameter("mid_level_planner_type", "identity")
        self.declare_parameter("lookahead_distance", 50)
        self.declare_parameter("occupancy_inflation_radius", 0.5)
        self.declare_parameter("use_fake_path_plan", False)
        self.declare_parameter("use_cost_map", False)
        self.declare_parameter("cost_map_safe_distance", 0.5)
        self.declare_parameter("cost_map_nearest_obstacle_cost", 5.0)
        mid_level_planner_type = self.get_parameter("mid_level_planner_type").value
        lookahead_distance = self.get_parameter("lookahead_distance").value
        assert lookahead_distance > 0
        occupancy_inflation_radius = self.get_parameter(
            "occupancy_inflation_radius"
        ).value
        assert occupancy_inflation_radius > 0
        use_fake_path_plan = self.get_parameter("use_fake_path_plan").value
        use_cost_map = self.get_parameter("use_cost_map").value
        cost_map_safe_distance = self.get_parameter("cost_map_safe_distance").value
        cost_map_nearest_obstacle_cost = self.get_parameter(
            "cost_map_nearest_obstacle_cost"
        ).value
        self.get_logger().info(
            f"{mid_level_planner_type=}, {use_fake_path_plan=}, {use_cost_map=}"
        )

        # mid-level planner initialization
        match mid_level_planner_type:
            case "astar":
                self.occupancy_map = OccupancyMap(
                    self.feedback_collector,
                    inflate_radius_meters=occupancy_inflation_radius,
                    use_cost_map=use_cost_map,
                    safe_distance=cost_map_safe_distance,
                    nearest_obstacle_cost=cost_map_nearest_obstacle_cost,
                )
                self.occupancy_map_updater = OccupancyGridROSUpdater(
                    self,
                    self.body_frame,
                    self.odom_frame,
                    self.occupancy_map,
                    self.feedback_collector,
                    self.tf_buffer,
                )
                self.mid_level_planner = MidLevelPlanner(
                    self.occupancy_map,
                    self.feedback_collector,
                    lookahead_distance_grid=lookahead_distance,
                )
                self.get_logger().info("Using A* mid-level planner")
            case "identity":
                self.mid_level_planner = IdentityPlanner(self.feedback_collector)
            case _:
                raise ValueError(
                    f"Invalid mid-level planner type {mid_level_planner_type}"
                )

        if interface == "fake":
            self.declare_parameter("fake_spot_external_pose", False)
            external_pose = self.get_parameter("fake_spot_external_pose").value

            self.declare_parameter("use_fake_spot_pose", False)
            if self.get_parameter("use_fake_spot_pose").value:
                self.declare_parameter("fake_spot_x", np.inf)
                self.declare_parameter("fake_spot_y", np.inf)
                self.declare_parameter("fake_spot_z", np.inf)
                self.declare_parameter("fake_spot_yaw", np.inf)
                spot_x = self.get_parameter("fake_spot_x").value
                spot_y = self.get_parameter("fake_spot_y").value
                spot_z = self.get_parameter("fake_spot_z").value
                spot_yaw = self.get_parameter("fake_spot_yaw").value

                spot_init_pose2d = np.array([spot_x, spot_y, spot_z, spot_yaw])
                assert not any(np.isinf(spot_init_pose2d)), (
                    "Must set fake_spot_x, fake_spot_y, fake_spot_z, fake_spot_yaw"
                )
            else:
                spot_init_pose2d = np.array([0, 0, 0, 0])

            self.get_logger().info(str(spot_init_pose2d))
            self.get_logger().info("About to initialize fake spot")
            self.spot_interface = FakeSpot(
                username=bdai_username,
                password=bdai_password,
                init_pose=spot_init_pose2d,
                semantic_model_path=None,
            )

            self.spot_ros_interface = FakeSpotRos(
                self,
                self.spot_interface,
                odom_frame,
                body_frame,
                external_pose=external_pose,
            )

        elif interface == "sim":
            from dcist_sim_ros.sim_spot import SimSpot
            from dcist_sim_ros.sim_spot_ros import SimSpotRos

            # SimSpotRos provides get_pose_fn from TF; construct it first with
            # a placeholder sim_spot and wire the back-reference afterward.
            self.spot_ros_interface = SimSpotRos(
                self,
                None,
                odom_frame,
                body_frame,
                rgb_topic=self.get_parameter("sim_rgb_topic").value,
            )
            self.spot_interface = SimSpot(
                node=self,
                robot_name=body_frame.split("/")[0],
                get_pose_fn=self.spot_ros_interface.get_pose_fn,
            )
            self.spot_ros_interface.attach(self.spot_interface)

        elif interface == "real":
            assert spot_ip != ""
            assert bdai_username != ""
            assert bdai_password != ""
            self.get_logger().info("About to initialize Spot")
            self.get_logger().info(f"{bdai_username=}, {bdai_password=}, {spot_ip=}")
            self.spot_interface = Spot(
                username=bdai_username,
                password=bdai_password,
                ip=spot_ip,
            )

        else:
            raise ValueError(f"Unknown spot_interface: {interface}")

        self.get_logger().info("Initialized!")
        self.status_str = "Idle"

        # <spot_vision_frame> must get mapped to the TF frame corresponding
        # to Spot's vision odom estimate.
        special_tf_remaps = {"<spot_vision_frame>": odom_frame}

        def tf_lookup_fn(parent, child):
            if parent in special_tf_remaps:
                parent = special_tf_remaps[parent]
            if child in special_tf_remaps:
                child = special_tf_remaps[child]
            try:
                return get_tf_pose(self.tf_buffer, parent, child)
            except tf2_ros.TransformException as e:
                self.get_logger.warn(f"Failed to get transform: {e}")

        self.tf_lookup_fn = tf_lookup_fn  # TODO: use this to test transformation

        self.declare_parameter("detector_confidence", 0.25)
        self.declare_parameter("detector_class_synonyms", "")
        detector_confidence = self.get_parameter("detector_confidence").value
        detector_class_synonyms_str = self.get_parameter(
            "detector_class_synonyms"
        ).value
        detector_class_synonyms = (
            yaml.safe_load(detector_class_synonyms_str)
            if detector_class_synonyms_str
            else None
        )
        if detector_class_synonyms is not None and not isinstance(
            detector_class_synonyms, dict
        ):
            raise ValueError(
                "Parameter 'detector_class_synonyms' must be a YAML/JSON dict "
                "mapping canonical class names to prompt phrases, got "
                f"{type(detector_class_synonyms).__name__}: "
                f"{detector_class_synonyms!r}"
            )

        detector = YOLODetector(
            self.spot_interface,
            yolo_world_path=detector_model_path,
            conf=detector_confidence,
            class_synonyms=detector_class_synonyms,
        )

        self.spot_executor = se.SpotExecutor(
            self.spot_interface,
            detector,
            tf_lookup_fn,
            self.mid_level_planner,
            follower_lookahead,
            goal_tolerance,
            self.feedback_collector,
            use_fake_path_plan,
            follow_timeout_per_meter,
        )
        self.spot_executor.initialize_lease_manager(self.feedback_collector)

        self.action_sequence_sub = self.create_subscription(
            ActionSequenceMsg,
            "~/action_sequence_subscriber",
            self.process_action_sequence,
            10,
        )

        self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        # PR B8: volatile platform-state snapshot for the planner's dispatch
        # gate. Published on every 10th heartbeat tick (~1 Hz at the 0.1 s
        # heartbeat) — guards are dispatch-time state, not telemetry.
        self.runtime_guards_pub = self.create_publisher(
            RuntimeGuardsMsg, "~/runtime_guards", 1
        )
        self._guard_tick = 0
        heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
        timer_period_s = 0.1
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
        )

        # External pause/stop control: allow other nodes (e.g. goal_manager) to
        # pause/resume or hard-stop the current plan.
        self.pause_sub = self.create_subscription(
            Bool, "~/pause", self.handle_pause, 10
        )
        self.resume_sub = self.create_subscription(
            Bool, "~/resume", self.handle_resume, 10
        )
        self.stop_sub = self.create_subscription(Bool, "~/stop", self.handle_stop, 10)

        # Operator override (console "Take lease"): forcibly take the Spot lease,
        # e.g. from the tablet. Own group: it blocks on the robot for a moment.
        self.take_lease_srv = self.create_service(
            Trigger,
            "~/take_lease",
            self.handle_take_lease,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def hb_callback(self):
        msg = NodeInfoMsg()
        msg.nickname = "spot_executor"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = NodeInfoMsg.NOMINAL
        msg.notes = self.status_str
        self.heartbeat_pub.publish(msg)

        self._guard_tick += 1
        if self._guard_tick % 10 == 0:
            self.publish_runtime_guards()

    def publish_runtime_guards(self):
        """PR B8: best-effort guard snapshot; must never break the heartbeat."""
        from spot_executor.guards import extract_runtime_guards

        try:
            state = self.spot_interface.get_state()
        except Exception as e:  # noqa: BLE001 -- a state hiccup is not fatal
            self.get_logger().warning(f"runtime guards: get_state failed: {e}")
            return
        lease_owned = None
        owner = None
        lease_manager = getattr(self.spot_executor, "lease_manager", None)
        if lease_manager is not None:
            owner = getattr(lease_manager, "owner_name", None)
            if owner is not None:
                lease_owned = str(owner).startswith("understanding")
        guards = extract_runtime_guards(
            state, lease_owned=lease_owned, lease_owner=owner
        )
        msg = RuntimeGuardsMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.robot_name = self.get_namespace().strip("/")
        msg.battery_percent = float(guards["battery_percent"])
        msg.battery_known = bool(guards["battery_known"])
        msg.estop_pressed = bool(guards["estop_pressed"])
        msg.estop_known = bool(guards["estop_known"])
        msg.powered_on = bool(guards["powered_on"])
        msg.power_known = bool(guards["power_known"])
        msg.lease_owned = bool(guards["lease_owned"])
        msg.lease_known = bool(guards["lease_known"])
        msg.lease_owner = str(guards["lease_owner"])
        msg.faults = [str(f) for f in guards["faults"]]
        msg.notes = self.status_str
        self.runtime_guards_pub.publish(msg)

    def handle_take_lease(self, request, response):
        lease_manager = getattr(self.spot_executor, "lease_manager", None)
        if lease_manager is None:
            response.success = False
            response.message = "no lease manager (executor not initialized)"
            return response
        response.success, response.message = take_lease_report(lease_manager)
        log = self.get_logger().info if response.success else self.get_logger().error
        log(f"take_lease: {response.message}")
        return response

    def handle_pause(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received external pause request.")
            self.spot_executor.set_paused(True, self.feedback_collector)

    def handle_resume(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received external resume request.")
            self.spot_executor.set_paused(False, self.feedback_collector)

    def handle_stop(self, msg: Bool):
        # Best-effort hard stop of the current action sequence, if any.
        if self.background_thread is not None and self.background_thread.is_alive():
            self.get_logger().info(
                "Received external stop request; terminating current plan."
            )
            self.spot_executor.terminate_sequence(self.feedback_collector)

    def process_action_sequence(self, msg):
        # An exception escaping a subscription callback tears down the rclpy
        # executor and kills the node. Plan repair means many preemptions in a
        # row, so one bad plan must degrade rather than end the run.
        try:
            self._process_action_sequence(msg)
        except Exception:
            self.status_str = "Idle"
            with self._plan_id_lock:
                if self._current_plan_id == getattr(msg, "plan_id", None):
                    self._current_plan_id = None
            self.get_logger().error(
                "Failed to handle incoming plan "
                f"(plan_id={getattr(msg, 'plan_id', '<unknown>')}); node staying up:\n"
                f"{traceback.format_exc()}"
            )

    def _process_action_sequence(self, msg):
        # Plan-repair preemption: only accept a new plan when plan_id is different
        with self._plan_id_lock:
            if (
                self._current_plan_id is not None
                and msg.plan_id == self._current_plan_id
            ):
                self.get_logger().info(
                    f"Ignoring duplicate plan (plan_id={msg.plan_id}), already executing"
                )
                return
            if self.background_thread is not None and self.background_thread.is_alive():
                self.get_logger().info(
                    f"Preempting current plan (plan_id={self._current_plan_id}) with new plan (plan_id={msg.plan_id})"
                )
                self.spot_executor.terminate_sequence(self.feedback_collector)
            self._current_plan_id = msg.plan_id

        def process_sequence():
            try:
                self.status_str = "Processing action sequence"
                self.get_logger().info("Starting action sequence")
                sequence = from_msg(msg)

                # PR B5: every per-action result echoes the dispatched
                # sequence's identity. Set only after any preempted plan has
                # terminated, so its PREEMPTED report keeps its own plan_id.
                self.feedback_collector.current_plan_id = msg.plan_id
                self.feedback_collector.current_robot_name = msg.robot_name

                self.spot_executor.process_action_sequence(
                    sequence, self.feedback_collector
                )

                self.get_logger().info("Finished execution action sequence.")
            except Exception:
                # Same reasoning as above: report and let the node keep serving
                # plans instead of leaving a dead executor thread behind.
                self.get_logger().error(
                    f"Action sequence (plan_id={msg.plan_id}) aborted:\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                with self._plan_id_lock:
                    if self._current_plan_id == msg.plan_id:
                        self._current_plan_id = None
                self.status_str = "Idle"

        self.feedback_collector.break_out_of_waiting_loop = False
        self.feedback_collector.plan_valid = True
        self.background_thread = threading.Thread(target=process_sequence, daemon=False)
        self.background_thread.start()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SpotExecutorRos()

        ros_executor = MultiThreadedExecutor()
        ros_executor.add_node(node)

        try:
            ros_executor.spin()
        finally:
            ros_executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
