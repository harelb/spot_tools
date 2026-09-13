import logging
import math
import time
from contextlib import nullcontext

import bosdyn.client.util
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from bosdyn import geometry
from bosdyn.api import basic_command_pb2, image_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_stand,
)

from spot_executor.stitch_front_images import stitch, stitch_live, stitch_RGB


class Spot:
    def pick_camera_session(self):
        """Hardware cameras are already available; simulators may render on demand."""
        return nullcontext()

    def __init__(
        self,
        username="user",
        password="password",
        ip="192.168.80.3",
        take_lease=True,
        set_estop=False,
        verbose=False,
        semantic_model_path=None,
        debug=False,
        semantic_name_to_id=None,
        labelspace_map=None,
    ):
        self.is_fake = False
        self.verbose = verbose
        bosdyn.client.util.setup_logging(verbose)
        self.sdk = bosdyn.client.create_standard_sdk("understanding-spot")
        self.robot = self.sdk.create_robot(ip)

        self.debug = debug
        if debug:
            return

        self.id_client = self.robot.ensure_client("robot-id")
        self.id = self.id_client.get_id()
        self.robot.authenticate(username, password)

        self.state_client = self.robot.ensure_client("robot-state")
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)

        if self.robot.has_arm():
            self.manipulation_api_client = self.robot.ensure_client(
                ManipulationApiClient.default_service_name
            )
        else:
            logging.warning("Spot does not have an arm, manipulation disabled!")
            self.manipulation_api_client = None

        self.power_client = self.robot.ensure_client("power")
        self.estop_client = self.robot.ensure_client("estop")
        self.estop_keep_alive = None
        self.lease_client = self.robot.ensure_client("lease")
        # self.command_client = self.robot.ensure_client('lease')
        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self.lease_keep_alive = None
        if semantic_model_path is not None:
            self.ort_session = ort.InferenceSession(semantic_model_path)
        else:
            self.ort_session = None
        self.semantic_name_to_id = semantic_name_to_id
        self.labelspace_map = labelspace_map

        if set_estop:
            self.set_estop()

        if take_lease:
            self.take_lease()
        else:
            self.aquire_lease()

    def get_state(self):
        return self.state_client.get_robot_state()

    def get_pose(self):
        robot_state = self.get_state()
        transforms = robot_state.kinematic_state.transforms_snapshot

        assert str(transforms) != ""

        out_tform_body = get_se2_a_tform_b(
            transforms, VISION_FRAME_NAME, BODY_FRAME_NAME
        )

        return np.array([out_tform_body.x, out_tform_body.y, out_tform_body.angle])

    def set_twist(self, vx, vy, v_rot):
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=vx,
            v_y=vy,
            v_rot=v_rot,
        )
        # Send with a short timeout so Spot stops if messages stop arriving
        self.command_client.robot_command(
            command=cmd,
            end_time_secs=time.time() + 0.5,
        )

    def get_image(self, view="hand_color_image", show=False):
        self.robot.logger.info("Getting an image from: %s", view)
        image_responses = self.image_client.get_image_from_sources([view])

        if len(image_responses) != 1:
            print(f"Got invalid number of images: {len(image_responses)}")
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)
        if show:
            image.show()

        return image, img

    def pixel_format_string_to_enum(self, enum_string):
        return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

    def get_stitched_image_RGB(self, fl_img, fr_img, crop_image=False):
        return stitch_RGB(fl_img, fr_img, crop_image)

    def get_stitched_image(self, jpeg_quality_percent=50, crop_image=False):
        return stitch(self.robot, jpeg_quality_percent, crop_image)

    def get_live_stitched_image(self, jpeg_quality_percent=50):
        return stitch_live(self.robot, jpeg_quality_percent)

    def get_image_RGB(
        self, view="hand_color_image", pixel_format="PIXEL_FORMAT_RGB_U8"
    ):
        pixel_format = self.pixel_format_string_to_enum(pixel_format)
        image_request = image_pb2.ImageRequest(
            image_source_name=view, quality_percent=100, pixel_format=pixel_format
        )

        image_responses = self.image_client.get_image([image_request])

        if len(image_responses) != 1:
            print(f"Got invalid number of images: {len(image_responses)}")
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_bytes = 3
        else:
            num_bytes = 1

        dtype = np.uint8

        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
        else:
            img = cv2.imdecode(img, -1)

        return image, img

    def segment_image(
        self,
        image,
        model_path="data/models/efficientvit_seg_l2.onnx",
        rotate=0,
        class_name="bag",
        show=False,
    ):
        if self.ort_session is None:
            raise Exception("Cannot segment image if no ort_session is loaded!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (512, 512))  # Resize to match the model's input size

        # Rotate Image
        for i in range(rotate):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if show:
            plt.imshow(
                image, cmap="jet"
            )  # 'jet' is a colormap, you can choose other colormaps like 'gray'
            plt.colorbar()  # Add a colorbar to the side
            plt.title("Image Output")
            plt.show()

        # Normalize and convert to float
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to the format (batch_size, channels, height, width)
        image = np.transpose(image, (2, 0, 1))  # Change shape to (3, 512, 512)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Run the model on the input image
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: image})

        if show:
            plt.imshow(
                outputs[0], cmap="jet"
            )  # 'jet' is a colormap, you can choose other colormaps like 'gray'
            plt.colorbar()  # Add a colorbar to the side
            plt.title("Segmentation Output")
            plt.show()

        # Rotate Outputs
        for i in range(rotate):
            outputs[0] = cv2.rotate(outputs[0], cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Get the model's output
        return outputs[0]

    def set_estop(self, name="my_estop", timeout=9.0):
        self.estop_client.get_status()
        estop_endpoint = bosdyn.client.estop.EstopEndpoint(
            client=self.estop_client, name=name, estop_timeout=timeout
        )
        estop_endpoint.force_simple_setup()
        self.estop_keep_alive = bosdyn.client.estop.EstopKeepAlive(estop_endpoint)
        self.estop_client.get_status()

    def aquire_lease(self):
        self.lease = self.lease_client.acquire()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, warnings=False
        )
        self.lease_client.list_leases()

    def take_lease(self):
        self.lease_client.take()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, warnings=False
        )
        self.lease_client.list_leases()

    def power_on(self):
        """
        Power on robot.

        Args:
            self: (Robot) Interface to Spot robot.
        """
        self.robot.logger.info("Powering on robot...")
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."
        self.robot.logger.info("Robot powered on.")

    def safe_power_off(self):
        """
        Sit and power off robot.

        Args:
            self: (Robot) Interface to Spot robot.
        """

        self.robot.logger.info("Powering off robot...")
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.robot.logger.info("Robot safely powered off.")

    def ready_for_motion(self):
        """
        Check if the robot is able to move

        @return  True if robot is ready for motion command; false otherwise
        """
        if self.robot is None:
            return False

        # Establish time sync with the robot
        self.robot.time_sync.wait_for_sync()

        # Verify the robot is not estopped
        assert not self.robot.is_estopped(), (
            "Robot is estopped. "
            "Please use an external E-Stop client, "
            "such as the estop SDK example, to configure E-Stop."
        )

        # Acquire a lease to indicate that we want to control the robot
        # [yveys]: Should this say self.lease_client is not None?
        # Though this if statement is never entered in practice since the least_client should be
        # set properly in the init function
        if self.lease_client is None:
            try:
                self.take_lease()
            except Exception as ex:
                print(ex)
                self.aquire_lease()

        # Power the motor on
        if not self.robot.is_powered_on():
            self.robot.power_on(timeout_sec=20)

        return self.robot.is_powered_on()

    def stand(self):
        """
        Stand robot.

        Args:
            self: (Robot) Interface to Spot robot.
        """

        assert self.ready_for_motion(), "Robot not ready for motion."

        self.robot.logger.info("Commanding robot to stand...")
        self.robot.ensure_client(RobotCommandClient.default_service_name)

        blocking_stand(self.command_client, timeout_sec=10)
        self.robot.logger.info("Robot standing.")

    def sit(self):
        """
        Sit robot.

        Args:
            self: (Robot) Interface to Spot robot.
        """

        assert self.ready_for_motion(), "Robot not ready for motion."

        self.robot.logger.info("Commanding robot to sit...")
        command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        cmd = RobotCommandBuilder.synchro_sit_command()
        command_client.robot_command(cmd)
        self.robot.logger.info("Robot sitting.")

    def pitch_up(self):
        """
        Pitch robot up to look for door handle.

        Args:
            self: (Robot) Interface to Spot robot.
        """
        robot = self.robot
        robot.logger.info("Pitching robot up...")
        robot.ensure_client(RobotCommandClient.default_service_name)

        footprint_R_body = geometry.EulerZXY(0.0, 0.0, -1 * math.pi / 6.0)
        cmd = RobotCommandBuilder.synchro_stand_command(
            footprint_R_body=footprint_R_body
        )
        cmd_id = self.command_client.robot_command(cmd)
        timeout_sec = 10.0
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            response = self.command_client.robot_command_feedback(cmd_id)
            synchronized_feedback = response.feedback.synchronized_feedback
            status = (
                synchronized_feedback.mobility_command_feedback.stand_feedback.status
            )
            if status == basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING:
                robot.logger.info("Robot pitched.")
                return
            time.sleep(1.0)
        raise Exception("Failed to pitch robot.")
