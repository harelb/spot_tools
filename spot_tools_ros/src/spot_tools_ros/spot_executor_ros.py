import logging
import os
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import spot_executor as se
import tf2_ros
import yaml
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface_ros.action_descriptions_ros import from_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from sensor_msgs.msg import Image
from shapely.geometry import Point
from spot_executor.fake_spot import FakeSpot
from spot_executor.spot import Spot
from spot_skills.detection_utils import YOLODetector
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from robot_executor_interface.mid_level_planner import (
    IdentityPlanner,
    MidLevelPlanner,
    OccupancyMap,
)
from spot_tools_ros.fake_spot_ros import FakeSpotRos
from spot_tools_ros.occupancy_grid_ros_updater import OccupancyGridROSUpdater
from spot_tools_ros.utils import get_tf_pose, waypoints_to_path

# Optional integrations: the trimmed mapping workspace ships without the rviz
# approval UI, the system monitor, heracles, or a numpy-2-compatible cv_bridge.
try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None
try:
    from heracles_ros_interfaces.srv import UpdateHoldingState
except ImportError:
    UpdateHoldingState = None
try:
    from nlu_interface_rviz.msg import (
        ManipulationApprovalRequest,
        ManipulationApprovalResponse,
    )
except ImportError:
    ManipulationApprovalRequest = ManipulationApprovalResponse = None
try:
    from ros_system_monitor_msgs.msg import NodeInfoMsg
except ImportError:
    NodeInfoMsg = None


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


def build_markers(pts, namespaces, frames, colors):
    ma = MarkerArray()
    for i, pt in enumerate(pts):
        m = pt_to_marker(pt, namespaces[i], i, colors[i], fid=frames[i])
        ma.markers.append(m)
    return ma


class RosFeedbackCollector:
    def __init__(self, odom_frame: str, output_dir: str, log_to_file_level):
        self.pick_confirmation_event = threading.Event()
        # self.pick_confirmation_response = False

        self.pick_confirmation_approved = False
        self.pick_confirmation_xy = [0, 0]
        self.pick_confirmation_image_index = 0

        self.break_out_of_waiting_loop = False
        self.plan_valid = True
        self.odom_frame = odom_frame

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

    def bounding_box_detection_feedback(
        self,
        detection_imgs,
        source_names,
        detection_index,
        centroid_x,
        centroid_y,
        semantic_class,
    ):
        if self.detection_img_pub is None:
            self.logger.warning(
                "No manipulation approval UI available; auto-approving detection"
            )
            xy = [centroid_x or 0, centroid_y or 0]
            return detection_index is not None, xy, detection_index or 0

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
        if CvBridge is None:
            return
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
        colors = [[0, 1, 1], [1, 0, 1]]
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
        colors = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
        frames = [self.odom_frame] * 3
        self.mlp_target_publisher.publish(
            build_markers(pts, namespaces, frames, colors)
        )

    def gaze_feedback(self, pose, gaze_point):
        pass

    def print(self, level, string):
        match level:
            case "DEBUG":
                log_fn = self.logger.debug
            case "INFO":
                log_fn = self.logger.info
            case "WARNING":
                log_fn = self.logger.warning
            case "ERROR":
                log_fn = self.logger.error
            case _:
                raise ValueError(f"Invalid log level {level}")
        log_fn(str(string))

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

    def register_publishers(self, node):
        self.logger = node.get_logger()

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

        self.detection_img_pub = None
        if ManipulationApprovalRequest is not None:
            self.detection_img_pub = node.create_publisher(
                ManipulationApprovalRequest,
                "~/manipulation_request",
                qos_profile=latching_qos,
            )
            node.create_subscription(
                ManipulationApprovalResponse,
                "~/pick_confirmation",
                self.pick_confirmation_callback,
                10,
            )

        self.lease_takeover_publisher = node.create_publisher(String, "~/takeover", 10)

        self.holding_client = None
        if UpdateHoldingState is not None:
            self.holding_client = node.create_client(
                UpdateHoldingState, "update_holding_state"
            )

        # TODO(aaron): Once we switch logging to python logger,
        # should move into init
        self.logger.info(f"Logging to: {self.output_dir}")
        if not os.path.exists(self.output_dir):
            self.logger.info(f"Making {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        log_fn = os.path.join(self.output_dir, "lease_log.txt")
        with open(log_fn, "w") as fo:
            fo.write("time,event\n")

    def set_robot_holding_state(self, is_holding: bool, object_id: str, timeout=5):
        if self.holding_client is None:
            self.logger.warning("UpdateHoldingState service unavailable in this build")
            return False
        req = UpdateHoldingState.Request()
        req.is_holding = is_holding
        req.id = object_id

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


class SpotExecutorRos(Node):
    def __init__(self):
        super().__init__("spot_executor_ros")
        self.debug = False
        self.background_thread = None

        # Connectivity parameters
        self.declare_parameter("spot_ip", "")
        self.declare_parameter("bosdyn_client_username", "")
        self.declare_parameter("bosdyn_client_password", "")
        spot_ip = self.get_parameter("spot_ip").value
        assert spot_ip != ""
        bdai_username = self.get_parameter("bosdyn_client_username").value
        assert bdai_username != ""
        bdai_password = self.get_parameter("bosdyn_client_password").value
        assert bdai_password != ""

        # Follow Skill
        self.declare_parameter("follower_lookahead", 0.0)
        follower_lookahead = self.get_parameter("follower_lookahead").value
        assert follower_lookahead > 0

        self.declare_parameter("goal_tolerance", 0.0)
        goal_tolerance = self.get_parameter("goal_tolerance").value
        assert goal_tolerance > 0
        self.get_logger().info(f"{goal_tolerance=}")

        # <= 0 keeps only the path_distance * 6 global timeout
        self.declare_parameter("follow_progress_timeout", 0.0)
        follow_progress_timeout = self.get_parameter("follow_progress_timeout").value
        if follow_progress_timeout <= 0:
            follow_progress_timeout = None

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

        # TODO(multy): quick way to log everything feedback print to log file
        self.declare_parameter("log_to_file", "")
        log_to_file = self.get_parameter("log_to_file").value

        self.feedback_collector = RosFeedbackCollector(
            self.odom_frame, output_dir, log_to_file
        )
        self.feedback_collector.register_publishers(self)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Robot Initialization
        self.declare_parameter("use_fake_spot_interface", False)
        use_fake_spot_interface = self.get_parameter("use_fake_spot_interface").value
        self.get_logger().info(
            f"{use_fake_spot_interface=} (override present: "
            f"{'use_fake_spot_interface' in self._parameter_overrides})"
        )

        # mid-level planner parameters
        self.declare_parameter("mid_level_planner_type", "identity")
        self.declare_parameter("lookahead_distance", 50)
        self.declare_parameter("occupancy_inflation_radius", 0.5)
        self.declare_parameter("use_fake_path_plan", False)
        self.declare_parameter("use_cost_map", False)
        self.declare_parameter("cost_map_safe_distance", 0.5)
        self.declare_parameter("cost_map_nearest_obstacle_cost", 5.0)
        self.declare_parameter("path_commitment_weight", 0.0)
        self.declare_parameter("path_commitment_band", 0.5)
        self.declare_parameter("allow_unknown_target", False)
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
                    path_commitment_weight=self.get_parameter(
                        "path_commitment_weight"
                    ).value,
                    path_commitment_band_m=self.get_parameter(
                        "path_commitment_band"
                    ).value,
                    allow_unknown_target=self.get_parameter(
                        "allow_unknown_target"
                    ).value,
                )
                self.get_logger().info("Using A* mid-level planner")
            case "identity":
                self.mid_level_planner = IdentityPlanner(self.feedback_collector)
            case _:
                raise ValueError(
                    f"Invalid mid-level planner type {mid_level_planner_type}"
                )

        if use_fake_spot_interface:
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

            self.declare_parameter("fake_spot_kinematic", False)
            self.get_logger().info(str(spot_init_pose2d))
            self.get_logger().info("About to initialize fake spot")
            self.spot_interface = FakeSpot(
                username=bdai_username,
                password=bdai_password,
                init_pose=spot_init_pose2d,
                semantic_model_path=None,
                kinematic=self.get_parameter("fake_spot_kinematic").value,
            )

            self.spot_ros_interface = FakeSpotRos(
                self,
                self.spot_interface,
                odom_frame,
                body_frame,
                external_pose=external_pose,
            )

        else:
            self.get_logger().info("About to initialize Spot")
            self.get_logger().info(f"{bdai_username=}, password set: {bool(bdai_password)}, {spot_ip=}")
            self.spot_interface = Spot(
                username=bdai_username,
                password=bdai_password,
                ip=spot_ip,
            )

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
                self.get_logger().warn(f"Failed to get transform: {e}")
                raise

        self.tf_lookup_fn = tf_lookup_fn  # TODO: use this to test transformation

        # No detector path -> navigation-only node; Pick fails cleanly instead of
        # pulling torch + weights in at startup.
        detector = None
        # 'none' as well as '': ros2 launch rejects `arg:=` with an empty value
        if detector_model_path and detector_model_path.lower() != "none":
            detector = YOLODetector(
                self.spot_interface,
                yolo_world_path=detector_model_path,
            )
        else:
            self.get_logger().warn("detector_model_path empty, Pick is disabled")

        self.spot_executor = se.SpotExecutor(
            self.spot_interface,
            detector,
            tf_lookup_fn,
            self.mid_level_planner,
            follower_lookahead,
            goal_tolerance,
            self.feedback_collector,
            use_fake_path_plan,
            follow_progress_timeout=follow_progress_timeout,
        )
        self.spot_executor.initialize_lease_manager(self.feedback_collector)

        # Optional GraphNav waypoint travel for Follow (graph_nav_enabled). The
        # GraphNav extension is not present in every spot_tools checkout, so keep
        # the ordinary executor usable when the optional module is absent.
        try:
            from spot_tools_ros.graph_nav_executor_ros import maybe_wrap_graph_nav
        except ModuleNotFoundError as exc:
            if exc.name != "spot_tools_ros.graph_nav_executor_ros":
                raise
            self.declare_parameter("graph_nav_enabled", False)
            if self.get_parameter("graph_nav_enabled").value:
                raise RuntimeError(
                    "graph_nav_enabled requires spot_tools_ros.graph_nav_executor_ros"
                ) from exc
            self.get_logger().warn(
                "GraphNav extension is unavailable; using the ordinary executor"
            )
        else:
            self.spot_executor = maybe_wrap_graph_nav(
                self, self.spot_executor, use_fake_spot=use_fake_spot_interface
            )

        # One JSON String per finished (or aborted) sequence: {plan_id, success, actions,
        # preempted, error}. Planners that dispatch over ROS wait on this to know when
        # the robot is done (open_set_navigation spot_stack, wait_for_completion_s).
        self.action_result_pub = self.create_publisher(String, "~/action_sequence_result", 10)
        self.action_sequence_sub = self.create_subscription(
            ActionSequenceMsg,
            "~/action_sequence_subscriber",
            self.process_action_sequence,
            10,
        )

        if NodeInfoMsg is not None:
            self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
            heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
            timer_period_s = 0.1
            self.timer = self.create_timer(
                timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
            )

    def hb_callback(self):
        msg = NodeInfoMsg()
        msg.nickname = "spot_executor"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = NodeInfoMsg.NOMINAL
        msg.notes = self.status_str
        self.heartbeat_pub.publish(msg)

    def process_action_sequence(self, msg):
        def process_sequence():
            import json

            self.status_str = "Processing action sequence"
            self.get_logger().info("Starting action sequence")
            sequence = from_msg(msg)
            result = {"plan_id": sequence.plan_id, "success": False, "actions": [], "preempted": False}
            try:
                outcomes = self.spot_executor.process_action_sequence(
                    sequence, self.feedback_collector
                ) or []
                result["actions"] = outcomes
                result["preempted"] = not self.spot_executor.keep_going
                result["success"] = (
                    len(outcomes) == len(sequence.actions)
                    and all(o["success"] for o in outcomes)
                    and not result["preempted"]
                )
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"
                self.get_logger().error(f"action sequence failed: {result['error']}")
            self.get_logger().info(f"Finished execution action sequence. success={result['success']}")
            self.status_str = "Idle"
            self.action_result_pub.publish(String(data=json.dumps(result)))

        if self.background_thread is not None and self.background_thread.is_alive():
            self.spot_executor.terminate_sequence(self.feedback_collector)

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
        # try_shutdown: a second shutdown (Ctrl-C during startup already shut the context
        # down) must not raise and bury the exception that actually stopped the node
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
