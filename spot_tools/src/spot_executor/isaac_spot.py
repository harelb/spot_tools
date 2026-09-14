"""Isaac hardware transport underneath the unmodified Spot Tools skills.

Only the SDK calls used by those skills are exposed. Requests and feedback retain
Boston Dynamics protobuf types. No networking to a physical Spot is performed.
The server must advertise the explicit Isaac protocol before any command is sent.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from bosdyn.api import image_pb2, manipulation_api_pb2, robot_command_pb2, robot_state_pb2
from .spot import Spot

PROTOCOL = "spot-tools-isaac-v1"


class IsaacTransport:
    def __init__(self, url="http://127.0.0.1:9250", timeout=5.0):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Isaac hardware endpoint must use loopback HTTP")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("invalid Isaac endpoint")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = None
        self.context = threading.local()
        health = self.call("health")
        if health.get("protocol") != PROTOCOL or health.get("backend") != "isaac":
            raise ValueError("endpoint is not the Isaac hardware backend")
        self.session = health["session_id"]
        self.capabilities = tuple(health.get("capabilities", ()))

    def call(self, method, **arguments):
        data = json.dumps(dict(method=method, session_id=self.session, arguments=arguments,
                               trace_context=getattr(self.context,'value',{})),
                          allow_nan=False).encode()
        request = Request(self.url + "/rpc", data=data,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        if value.get("error"):
            raise RuntimeError(value["error"])
        if self.session is not None and value.get("session_id") != self.session:
            raise RuntimeError("Isaac episode changed; reconnect and review again")
        return value["result"]

    def proto(self, method, response_type, request=None, **arguments):
        if request is not None:
            arguments["request"] = base64.b64encode(request.SerializeToString()).decode()
        result = self.call(method, **arguments)
        return response_type.FromString(base64.b64decode(result["protobuf"], validate=True))


class _CommandClient:
    def __init__(self, transport):
        self.transport = transport

    def robot_command(self, command, end_time_secs=None, lease=None, **kwargs):
        if kwargs:
            raise NotImplementedError(f"unsupported command options: {sorted(kwargs)}")
        # Send a relative lifetime, independent of simulator wall/simulation clock.
        lifetime = None if end_time_secs is None else end_time_secs - time.time()
        if lifetime is not None and lifetime <= 0:
            raise ValueError("command has already expired")
        result = self.transport.call("robot_command", request=base64.b64encode(
            command.SerializeToString()).decode(), lifetime_s=lifetime)
        return int(result["command_id"])

    def robot_command_feedback(self, command_id, **kwargs):
        if kwargs:
            raise NotImplementedError("unsupported feedback options")
        return self.transport.proto("robot_command_feedback",
            robot_command_pb2.RobotCommandFeedbackResponse, command_id=int(command_id))

    def clear_behavior_fault(self, fault_id):
        return self.transport.call("clear_behavior_fault", fault_id=int(fault_id))


class _ManipulationClient:
    def __init__(self, transport):
        self.transport = transport

    def manipulation_api_command(self, manipulation_api_request):
        return self.transport.proto("manipulation_command",
            manipulation_api_pb2.ManipulationApiResponse, manipulation_api_request)

    def manipulation_api_feedback_command(self, manipulation_api_feedback_request):
        return self.transport.proto("manipulation_feedback",
            manipulation_api_pb2.ManipulationApiFeedbackResponse, manipulation_api_feedback_request)

    def grasp_override_command(self, request):
        # Backend may update carry policy, but must never fabricate a held object.
        return self.transport.proto("grasp_override",
            manipulation_api_pb2.ApiGraspOverrideResponse, request)


class _LeaseClient:
    def __init__(self, transport):
        self.transport = transport

    def take(self):
        return self.transport.call("lease_take")

    acquire = take

    def list_leases(self):
        result = self.transport.call("lease_state")
        return [SimpleNamespace(lease_owner=SimpleNamespace(client_name=result["owner"]))]


class IsaacSpot(Spot):
    """Real skill branch, with injected SDK clients instead of hardware clients."""
    backend = "isaac"

    def __init__(self, endpoint="http://127.0.0.1:9250", *, transport=None):
        # Do not call Spot.__init__: it authenticates to physical hardware.
        self.transport = transport if transport is not None else IsaacTransport(endpoint)
        self.is_fake = False
        self.ort_session = None
        self.semantic_name_to_id = self.labelspace_map = None
        self.command_client = _CommandClient(self.transport)
        self.manipulation_api_client = _ManipulationClient(self.transport)
        self.lease_client = _LeaseClient(self.transport)
        self.state_client = SimpleNamespace(get_robot_state=self.get_state)
        self.image_client = SimpleNamespace(list_image_sources=self.list_image_sources)
        self.robot = SimpleNamespace(
            ensure_client=self.ensure_client,
            time_sync=SimpleNamespace(wait_for_sync=self.wait_for_sync),
            logger=logging.getLogger("spot_executor.isaac"),
            has_arm=lambda: True,
        )

    def ensure_client(self, service):
        clients = {"robot-command": self.command_client, "robot-state": self.state_client,
                   "manipulation": self.manipulation_api_client, "lease": self.lease_client}
        if service not in clients:
            raise NotImplementedError(f"Isaac SDK service {service!r} is unsupported")
        return clients[service]

    @contextmanager
    def pick_camera_session(self):
        request_id = uuid.uuid4().hex
        self.transport.call('pick_cameras', enabled=True, request_id=request_id)
        try:
            deadline = time.monotonic() + 10
            while True:
                health = self.transport.call('health')
                if health.get('pick_camera_error'):
                    raise RuntimeError(health['pick_camera_error'])
                if health.get('pick_camera_ready'):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('pick cameras did not produce fresh observations')
                time.sleep(.05)
            yield
        finally:
            self.transport.call('pick_cameras', enabled=False, request_id=request_id)

    def wait_for_sync(self):
        return self.transport.call("clock")

    def get_state(self):
        return self.transport.proto("robot_state", robot_state_pb2.RobotState)

    def get_image_RGB(self, view="front_zed_color_image", pixel_format="PIXEL_FORMAT_RGB_U8"):
        if pixel_format != "PIXEL_FORMAT_RGB_U8":
            raise NotImplementedError("only calibrated RGB sources are supported")
        response = self.transport.proto("image", image_pb2.ImageResponse, source=view)
        frame = response.shot.image
        if response.source.name != view or not response.shot.frame_name_image_sensor:
            raise ValueError("hardware returned a different or uncalibrated image source")
        if frame.format != image_pb2.Image.FORMAT_RAW or frame.pixel_format != image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            raise ValueError("Isaac must return native raw RGB")
        rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.rows, frame.cols, 3)
        # Spot's get_image_RGB returns OpenCV BGR arrays to the existing detector.
        return response, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def list_image_sources(self):
        return [image_pb2.ImageSource.FromString(base64.b64decode(row['protobuf'],validate=True))
                for row in self.transport.call('image_sources')]

    def take_lease(self):
        return self.lease_client.take()

    aquire_lease = take_lease

    def set_estop(self, *args, **kwargs):
        return self.transport.call("stop", reason="operator")

    def power_on(self):
        return self.transport.call("power", enabled=True)

    def safe_power_off(self):
        return self.transport.call("power", enabled=False)
