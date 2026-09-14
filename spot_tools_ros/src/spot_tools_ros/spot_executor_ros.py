import os
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import spot_executor as se
import tf2_ros
import yaml
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface_ros.action_descriptions_ros import from_msg
from robot_executor_msgs.msg import (
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
    def __init__(self, odom_frame: str, output_dir: str):
        self.pick_confirmation_event = threading.Event()
        # self.pick_confirmation_response = False

        self.pick_confirmation_approved = False
        self.pick_confirmation_xy = [0, 0]
        self.pick_confirmation_image_index = 0

        self.break_out_of_waiting_loop = False
        self.odom_frame = odom_frame

        # Executor -> planner return channel (PR B5): publisher + the
        # dispatched sequence's identity, set by process_action_sequence
        # before execution starts so every per-action result can echo them.
        self.action_result_pub = None
        self._clock = None
        self.current_plan_id = ""
        self.current_robot_name = ""
        self.held_object_id = None

        self.output_dir = output_dir

    def bounding_box_detection_feedback(
        self, detection_imgs, detection_index, centroid_x, centroid_y, semantic_class
    ):
        if self.detection_img_pub is None or self.manipulation_request_type is None:
            raise RuntimeError("manipulation interfaces are disabled")
        bridge = CvBridge()

        request_msg = self.manipulation_request_type()
        request_msg.images = [
            bridge.cv2_to_imgmsg(img, encoding="passthrough") for img in detection_imgs
        ]
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
        colors = [[0, 1, 1], [1, 0, 1]]
        frames = [self.odom_frame] * 2
        self.progress_point_pub.publish(build_markers(pts, namespaces, frames, colors))

    def path_follow_MLP_feedback(self, path, target_point_metric):
        self.mlp_path_publisher.publish(waypoints_to_path(self.odom_frame, path))
        target_point_metric_flattened = Point([p[0] for p in target_point_metric[:3]])

        pts = [target_point_metric_flattened]
        namespaces = ["projected target point"]
        colors = [[1, 0, 1]]
        frames = [self.odom_frame]
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

    def register_publishers(self, node, enable_manipulation_interfaces=False):
        self.logger = node.get_logger()
        self.detection_img_pub = None
        self.holding_client = None
        self.manipulation_request_type = None

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

        self.lease_takeover_publisher = node.create_publisher(String, "~/takeover", 10)

        # PR B5: per-action results back to the planner. Plain queue QoS —
        # consumers want the stream, not just the last value.
        self.action_result_pub = node.create_publisher(
            ActionResultMsg, "~/action_result", 10
        )
        self._clock = node.get_clock()

        if enable_manipulation_interfaces:
            # These interfaces belong to the deprecated NLU/manipulation
            # stack. Keep them lazy so this navigation-only deployment does
            # not require that stack merely to import or start the executor.
            from heracles_ros_interfaces.srv import UpdateHoldingState
            from nlu_interface_rviz.msg import (
                ManipulationApprovalRequest,
                ManipulationApprovalResponse,
            )

            self.manipulation_request_type = ManipulationApprovalRequest
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
            self.holding_client = node.create_client(
                UpdateHoldingState, "update_holding_state"
            )

        # TODO(aaron): Once we switch logging to python logger,
        # should move into init
        self.logger.info(f"Logging to: {self.output_dir}")
        if not os.path.exists(self.output_dir):
            self.logger.info(f"Making {self.output_dir}")
            os.makedirs(self.output_dir)
        log_fn = os.path.join(self.output_dir, "lease_log.txt")
        with open(log_fn, "w") as fo:
            fo.write("time,event\n")

    def set_robot_holding_state(self, is_holding: bool, object_id: str, timeout=5):
        # Identity comes from the completed real skill; the separate live
        # holding publisher confirms occupancy using current SDK state.
        self.held_object_id = object_id if is_holding else None
        if self.holding_client is None:
            return True
        from heracles_ros_interfaces.srv import UpdateHoldingState

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


def resolve_spot_interface(spot_interface: str, use_fake_spot_interface: bool) -> str:
    """Back-compat resolution: explicit spot_interface wins; the legacy
    use_fake_spot_interface flag maps to 'fake'."""
    if spot_interface not in ("", "real", "spot", "fake", "sim", "isaac"):
        raise ValueError(f"Invalid spot_interface: {spot_interface}")
    if spot_interface:
        return "real" if spot_interface == "spot" else spot_interface
    return "fake" if use_fake_spot_interface else "real"


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

        self.feedback_collector = RosFeedbackCollector(self.odom_frame, output_dir)
        self.declare_parameter("enable_manipulation_interfaces", False)
        self.feedback_collector.register_publishers(
            self,
            self.get_parameter("enable_manipulation_interfaces").value,
        )

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
        locked_backend = os.environ.get("SPOT_TOOLS_BACKEND_LOCK")
        if locked_backend and interface != locked_backend:
            raise RuntimeError(f"launch restricts hardware backend to {locked_backend!r}")

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

        elif interface == "isaac":
            from spot_executor.isaac_spot import IsaacSpot
            self.declare_parameter("isaac_endpoint", "http://127.0.0.1:9250")
            if use_fake_path_plan or use_fake_spot_interface:
                raise ValueError("Isaac parity requires the real executor and path planner")
            self.spot_interface = IsaacSpot(self.get_parameter("isaac_endpoint").value)
            # Sensor/TF ownership belongs to the Isaac bridge, never FakeSpotRos.
            self.spot_ros_interface = None

        elif interface == "real":
            assert spot_ip != ""
            assert bdai_username != ""
            assert bdai_password != ""
            self.get_logger().info("About to initialize Spot")
            # Never emit the Spot password to ROS logs.
            self.get_logger().info(f"{bdai_username=}, {spot_ip=}")
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
            load_on_demand=self.declare_parameter("detector_load_on_demand", False).value,
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
            self.declare_parameter("pick_image_source", "frontleft_fisheye_image").value,
        )
        if interface == "isaac":
            self.spot_interface.take_lease()
        self.spot_executor.initialize_lease_manager(self.feedback_collector)

        self.declare_parameter("enable_live_approval", False)
        if self.get_parameter("enable_live_approval").value:
            from spot_tools_ros.live_approval import LiveApproval
            self.live_approval = LiveApproval(self, self.feedback_collector)
            self.feedback_collector.bounding_box_detection_feedback = self.live_approval.pick
            self.feedback_collector.placement_feedback = self.live_approval.place
        verification_url = self.declare_parameter("placement_verification_url", "").value
        if verification_url:
            import json
            from urllib.request import Request, urlopen
            from spot_skills.placement_verification import PlacementVerifier
            episode = self.declare_parameter("placement_verification_episode", "").value
            stream = self.declare_parameter("placement_verification_stream", "").value
            if not episode or not stream:
                raise ValueError("Placement verification requires episode and calibrated live stream")
            def search_placement(body):
                request = Request(verification_url, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=45) as response:
                    payload = response.read(32*1024*1024+1)
                    if len(payload)>32*1024*1024:
                        raise RuntimeError("Placement evidence response exceeds memory limit")
                    return json.loads(payload)
            self.feedback_collector.placement_verifier = PlacementVerifier(
                self.spot_interface, search_placement, episode, stream,
                cancelled=lambda: self.feedback_collector.break_out_of_waiting_loop)
        self.live_cancel_sub = self.create_subscription(
            String, "~/live_cancel", self.cancel_live, 10
        )

        self.action_sequence_sub = self.create_subscription(
            ActionSequenceMsg,
            "~/action_sequence_subscriber",
            self.process_action_sequence,
            10,
        )

        self.manual_sequence_sub = self.create_subscription(
            ActionSequenceMsg, '~/manual_action_sequence',
            lambda msg: self.process_action_sequence(msg, mode='manual'), 10)

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
        # Liveness guards use elapsed wall time, including while simulation
        # clock advances slowly or is paused. Message stamps remain ROS time.
        from rclpy.clock import Clock, ClockType
        self.heartbeat_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group,
            clock=self.heartbeat_clock,
        )

        self.manual_server = None
        port = self.declare_parameter("manual_port", 0).value
        if port:
            from spot_tools_ros.manual_server import ManualServer
            self.manual_server = ManualServer(self, port,
                self.declare_parameter("manual_token_file", "").value,
                self.declare_parameter("run_id", "").value,
                self.declare_parameter("episode_id", "").value)

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
        lease_manager = getattr(self.spot_executor, "lease_manager", None)
        if lease_manager is not None:
            owner = getattr(lease_manager, "owner_name", None)
            if owner is not None:
                lease_owned = str(owner).startswith("understanding")
        guards = extract_runtime_guards(state, lease_owned=lease_owned)
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
        msg.faults = [str(f) for f in guards["faults"]]
        msg.notes = self.status_str
        self.runtime_guards_pub.publish(msg)
        from std_msgs.msg import String
        import json
        if not hasattr(self, 'holding_state_pub'):
            self.holding_state_pub = self.create_publisher(String, '~/holding_state', 1)
        context=self.manual_server.control if self.manual_server else None
        holding=bool(state.manipulator_state.is_gripper_holding_item)
        if not holding:self.feedback_collector.held_object_id=None
        self.holding_state_pub.publish(String(data=json.dumps(dict(
            known=state.HasField('manipulator_state'),is_holding=holding,
            object_id=self.feedback_collector.held_object_id if holding else None,
            plan_id=self.feedback_collector.current_plan_id,
            run_id=context.run_id if context else None,
            episode_id=context.episode_id if context else None,
            observed_at=time.time(),source='SDK state and completed skill identity'))))

    def cancel_live(self, msg):
        # Do not block the ROS callback waiting for a skill to finish.
        self.spot_executor.keep_going = False
        self.spot_executor.cancel_event.set()
        self.feedback_collector.break_out_of_waiting_loop = True
        try:
            from bosdyn.client.robot_command import RobotCommandBuilder
            self.spot_interface.command_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception as exc:
            self.get_logger().error(f"Could not send stop command: {exc}")
            return
        plan_id = msg.data or self.feedback_collector.current_plan_id
        worker = self.background_thread
        def acknowledge():
            if worker is not None:
                worker.join()
            response = ActionResultMsg()
            response.header.stamp = self.get_clock().now().to_msg()
            response.plan_id = plan_id
            response.robot_name = self.feedback_collector.current_robot_name
            response.status = "PREEMPTED"
            response.detail = "stop sent and execution worker terminated"
            self.feedback_collector.action_result_pub.publish(response)
        threading.Thread(target=acknowledge, daemon=True).start()

    def process_action_sequence(self, msg, mode="planned"):
        from contextlib import nullcontext
        ownership = self.manual_server.control.lock if self.manual_server else nullcontext()
        with ownership:
            if self.manual_server is not None and (self.manual_server.control.mode != mode or self.manual_server.control.compute_token):
                response = ActionResultMsg()
                response.plan_id, response.robot_name = msg.plan_id, msg.robot_name
                response.status = 'FAILED'
                response.detail = f'{self.manual_server.control.mode} control or a compute reservation owns motion; switch to {mode} and review again'
                self.feedback_collector.action_result_pub.publish(response)
                return
            if self.background_thread is not None and self.background_thread.is_alive():
                # Never preempt an active sequence by silently replacing it.
                response = ActionResultMsg()
                response.plan_id, response.robot_name = msg.plan_id, msg.robot_name
                response.status = 'FAILED'
                response.detail = 'Another action sequence is active; cancel and wait for acknowledgement'
                self.feedback_collector.action_result_pub.publish(response)
                return
            self.feedback_collector.break_out_of_waiting_loop = False
            self.spot_executor.cancel_event.clear()
            self.spot_executor.processing_action_sequence = True
            def process_sequence():
                self.status_str = 'Processing action sequence'
                self.feedback_collector.current_plan_id = msg.plan_id
                self.feedback_collector.current_robot_name = msg.robot_name
                try:
                    transport=getattr(self.spot_interface,'transport',None)
                    if transport is not None and hasattr(transport,'context'):
                        transport.context.value={'plan_id':msg.plan_id,'origin':mode}
                    sequence = from_msg(msg)
                    self.spot_executor.process_action_sequence(sequence, self.feedback_collector)
                except Exception as exc:
                    response = ActionResultMsg()
                    response.plan_id, response.robot_name = msg.plan_id, msg.robot_name
                    response.status, response.detail = 'FAILED', str(exc)
                    self.feedback_collector.action_result_pub.publish(response)
                finally:
                    if transport is not None and hasattr(transport,'context'):
                        transport.context.value={}
                    self.spot_executor.processing_action_sequence = False
                    self.status_str = 'Idle'
            self.background_thread = threading.Thread(target=process_sequence, daemon=True)
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
            if node.manual_server is not None:
                node.manual_server.close()
            node.spot_executor.keep_going = False
            node.feedback_collector.break_out_of_waiting_loop = True
            node.spot_executor.lease_manager.close()
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
