"""Executor-side blocking approval with correlation and bounded lifetime."""

import threading
import time
from uuid import uuid4


class LiveApproval:
    def __init__(self, node, feedback):
        from robot_executor_msgs.msg import ManipulationRequest, ManipulationResponse

        self.feedback = feedback
        self.request_type = ManipulationRequest
        self.publisher = node.create_publisher(
            ManipulationRequest, "~/live_approval_request", 10
        )
        self.subscription = node.create_subscription(
            ManipulationResponse, "~/live_approval_response", self.receive, 10
        )
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.pending = None
        self.response = None

    def receive(self, msg):
        with self.lock:
            if self.pending != (msg.request_id, msg.plan_id) or self.event.is_set():
                return
            self.response = msg
            self.event.set()

    def ask(self, kind, object_class, images=(), index=0, x=0, y=0):
        import cv2
        from sensor_msgs.msg import CompressedImage

        request = self.request_type()
        request.request_id, request.plan_id = (
            str(uuid4()),
            self.feedback.current_plan_id,
        )
        request.kind, request.object_class = kind, object_class
        request.image_index = index or 0
        request.image_x, request.image_y = int(x or 0), int(y or 0)
        for image in images:
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("could not encode approval image")
            request.images.append(CompressedImage(format="jpeg", data=data.tobytes()))
        with self.lock:
            self.event.clear()
            self.response = None
            self.pending = request.request_id, request.plan_id
        self.publisher.publish(request)
        deadline = time.monotonic() + 60
        while not self.event.wait(0.1):
            if self.feedback.break_out_of_waiting_loop or time.monotonic() >= deadline:
                break
        with self.lock:
            response = self.response
            self.pending = None
        if (
            self.feedback.break_out_of_waiting_loop
            or response is None
            or not response.approve
        ):
            # A rejection is terminal, not an invitation to retry the same skill.
            self.feedback.break_out_of_waiting_loop = True
            raise RuntimeError("manipulation cancelled or approval expired")
        return response

    def pick(self, images, index, x, y, object_class):
        msg = self.ask("PICK", object_class, images, index, x, y)
        if msg.image_index >= len(images):
            raise RuntimeError("invalid approved image")
        h, w = images[msg.image_index].shape[:2]
        if not (0 <= msg.image_x < w and 0 <= msg.image_y < h):
            raise RuntimeError("approved pixel outside image")
        return True, [msg.image_x, msg.image_y], msg.image_index

    def place(self, object_class):
        msg = self.ask("PLACE", object_class)
        if not msg.has_target:
            raise RuntimeError("placement requires a selected surface point")
        return [msg.target.x, msg.target.y, msg.target.z]
