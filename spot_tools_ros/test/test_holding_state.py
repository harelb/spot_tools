"""Holding-state updates after pick and place (field issue: none reached the map).

The executor's call to the base station's UpdateHoldingState must survive a
slow relay (longer wait, retries), wait for the reply without busy-spinning,
log loudly when it finally fails, and never change the pick or place result.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from spot_tools_ros.service_utils import call_service_with_retry, wait_for_future


class FakeLogger:
    def __init__(self):
        self.records = []

    def _log(self, level, msg):
        self.records.append((level, str(msg)))

    def debug(self, msg):
        self._log("DEBUG", msg)

    def info(self, msg):
        self._log("INFO", msg)

    def warning(self, msg):
        self._log("WARNING", msg)

    def error(self, msg):
        self._log("ERROR", msg)

    def levels(self):
        return [level for level, _ in self.records]


class FakeFuture:
    """Just the parts of rclpy.task.Future the helper uses."""

    def __init__(self):
        self._lock = threading.Lock()
        self._done = False
        self._result = None
        self._exception = None
        self._callbacks = []

    def done(self):
        return self._done

    def result(self):
        return self._result

    def exception(self):
        return self._exception

    def add_done_callback(self, cb):
        with self._lock:
            if not self._done:
                self._callbacks.append(cb)
                return
        cb(self)

    def _finish(self, result=None, exception=None):
        with self._lock:
            self._result = result
            self._exception = exception
            self._done = True
            callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            cb(self)


class FakeClient:
    """Answers from another thread, as an executor would.

    available: list of wait_for_service results, one per attempt.
    replies: list of (delay_s or None, response); None never answers.
    """

    def __init__(self, available, replies):
        self.available = list(available)
        self.replies = list(replies)
        self.wait_timeouts = []
        self.calls = 0
        self.removed = []

    def wait_for_service(self, timeout_sec):
        self.wait_timeouts.append(timeout_sec)
        return self.available.pop(0) if self.available else True

    def call_async(self, request):
        self.calls += 1
        future = FakeFuture()
        delay, response = self.replies.pop(0)
        if delay is not None:
            timer = threading.Timer(delay, future._finish, kwargs={"result": response})
            timer.daemon = True
            timer.start()
        return future

    def remove_pending_request(self, future):
        self.removed.append(future)


OK = SimpleNamespace(success=True)


def test_wait_for_future_wakes_on_reply_without_polling():
    future = FakeFuture()
    threading.Timer(0.05, future._finish, kwargs={"result": OK}).start()
    assert wait_for_future(future, 2.0)
    assert future.result() is OK


def test_wait_for_future_already_done():
    future = FakeFuture()
    future._finish(result=OK)
    assert wait_for_future(future, 0.0)


def test_wait_for_future_times_out():
    assert not wait_for_future(FakeFuture(), 0.05)


def test_retries_when_service_not_yet_visible():
    client = FakeClient(available=[False, True], replies=[(0.01, OK)])
    logger = FakeLogger()
    response = call_service_with_retry(
        client, object(), logger, "hold O58", attempts=3, wait_for_service_s=3.0
    )
    assert response is OK
    assert client.wait_timeouts == [3.0, 3.0]
    assert logger.levels() == ["WARNING"]


def test_retries_after_lost_reply():
    client = FakeClient(available=[], replies=[(None, None), (0.01, OK)])
    logger = FakeLogger()
    response = call_service_with_retry(
        client, object(), logger, "hold O58", attempts=3, call_timeout_s=0.05
    )
    assert response is OK
    assert client.calls == 2
    assert len(client.removed) == 1


def test_final_failure_logged_at_error_with_description():
    client = FakeClient(available=[False, False, False], replies=[])
    logger = FakeLogger()
    response = call_service_with_retry(
        client,
        object(),
        logger,
        "UpdateHoldingState for robot 'hamilton' placed object 'O58'",
        attempts=3,
        wait_for_service_s=0.0,
    )
    assert response is None
    assert logger.levels() == ["WARNING", "WARNING", "WARNING", "ERROR"]
    assert "hamilton" in logger.records[-1][1]
    assert "O58" in logger.records[-1][1]


# --- SpotExecutor: the action result is the pick or place, not the map update


try:
    import spot_executor.spot_executor as se
except ImportError:  # spot_tools (bosdyn) not installed
    se = None

needs_spot_executor = pytest.mark.skipif(se is None, reason="spot_tools not installed")


class FakeFeedback:
    def __init__(self, holding_result=True, holding_raises=False):
        self.holding_result = holding_result
        self.holding_raises = holding_raises
        self.holding_calls = []
        self.messages = []

    def print(self, level, msg):
        self.messages.append((level, str(msg)))

    def set_robot_holding_state(self, is_holding, object_id, position=None):
        self.holding_calls.append((is_holding, object_id, position))
        if self.holding_raises:
            raise RuntimeError("relay down")
        return self.holding_result

    def levels(self):
        return [level for level, _ in self.messages]


class IdentityRotation:
    x = y = z = 0.0
    w = 1.0


def make_executor(robot_xy=(0.0, 0.0), threshold=1.0):
    spot = SimpleNamespace(
        is_fake=True, get_pose=lambda: np.array([robot_xy[0], robot_xy[1], 0.0])
    )
    return se.SpotExecutor(
        spot,
        detector=None,
        transform_lookup=lambda parent, child: (np.zeros(3), IdentityRotation()),
        planner=None,
        place_distance_warn_m=threshold,
    )


def place_command(xy=(0.0, 0.0)):
    return se.Place(
        frame="map",
        object_class="box",
        robot_point=np.zeros(3),
        object_point=np.array([xy[0], xy[1], 0.0]),
        object_id="o58",
    )


@pytest.mark.parametrize(
    "feedback",
    [FakeFeedback(holding_result=False), FakeFeedback(holding_raises=True)],
    ids=["update_rejected", "update_raises"],
)
@needs_spot_executor
def test_place_succeeds_when_holding_update_fails(feedback):
    executor = make_executor()
    assert executor.execute_place(place_command(), feedback) is True
    assert feedback.holding_calls[0][:2] == (False, "O58")
    assert "ERROR" in feedback.levels()


@needs_spot_executor
def test_pick_succeeds_when_holding_update_fails(monkeypatch):
    monkeypatch.setattr(se, "object_grasp", lambda *a, **k: True)
    executor = make_executor()
    feedback = FakeFeedback(holding_result=False)
    command = se.Pick(
        frame="map",
        object_class="box",
        robot_point=np.zeros(3),
        object_point=np.zeros(3),
        object_id="o58",
    )
    assert executor.execute_pick(command, feedback) is True
    assert feedback.holding_calls == [(True, "O58", None)]
    assert "ERROR" in feedback.levels()


@needs_spot_executor
def test_place_far_from_place_point_warns():
    executor = make_executor(robot_xy=(3.0, 4.0), threshold=1.0)
    feedback = FakeFeedback()
    assert executor.execute_place(place_command((0.0, 0.0)), feedback) is True
    warnings = [m for level, m in feedback.messages if level == "WARNING"]
    assert len(warnings) == 1
    assert "5.00 m" in warnings[0]


@needs_spot_executor
def test_place_near_place_point_does_not_warn():
    executor = make_executor(robot_xy=(0.3, 0.4), threshold=1.0)
    feedback = FakeFeedback()
    assert executor.execute_place(place_command((0.0, 0.0)), feedback) is True
    assert "WARNING" not in feedback.levels()


@needs_spot_executor
def test_place_without_tf_still_places():
    executor = make_executor()
    executor.transform_lookup = lambda parent, child: None
    feedback = FakeFeedback()
    assert executor.execute_place(place_command(), feedback) is True
    assert "WARNING" not in feedback.levels()


# --- The ROS feedback collector against a real rclpy service


def test_ros_holding_update_round_trip_while_default_group_is_busy():
    """The reply arrives even while a default-group callback blocks.

    A preempting plan blocks a default-group subscription callback until the
    old sequence ends; the old sequence may be waiting for its holding reply.
    """
    rclpy = pytest.importorskip("rclpy")
    ros = pytest.importorskip("spot_tools_ros.spot_executor_ros")
    from heracles_ros_interfaces.srv import UpdateHoldingState
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import Empty

    rclpy.init()
    try:
        server_node = rclpy.create_node("fake_state_updater")
        received = []

        def handle(request, response):
            received.append((request.is_holding, request.id))
            response.success = True
            return response

        server_node.create_service(
            UpdateHoldingState,
            "update_holding_state",
            handle,
            callback_group=ReentrantCallbackGroup(),
        )

        node = rclpy.create_node("spot_executor_under_test")
        collector = ros.RosFeedbackCollector("odom", "/tmp", "")
        collector.logger = node.get_logger()
        collector.robot_name = "hamilton"
        collector.holding_client = node.create_client(
            UpdateHoldingState,
            "update_holding_state",
            callback_group=ReentrantCallbackGroup(),
        )

        # Hold the node's default (mutually exclusive) group busy.
        release = threading.Event()
        blocking = threading.Event()

        def block(_msg):
            blocking.set()
            release.wait(10.0)

        node.create_subscription(Empty, "block", block, 1)
        pub = node.create_publisher(Empty, "block", 1)

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(server_node)
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        try:
            for _ in range(50):
                pub.publish(Empty())
                if blocking.wait(0.1):
                    break
            assert blocking.is_set()
            ok = collector.set_robot_holding_state(
                False, "O58", position=[1.0, 2.0], timeout=3.0
            )
            assert ok
            assert received == [(False, "O58")]
        finally:
            release.set()
            executor.shutdown()
            server_node.destroy_node()
            node.destroy_node()
    finally:
        rclpy.shutdown()
