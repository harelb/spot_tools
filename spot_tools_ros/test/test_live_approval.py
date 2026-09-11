from types import SimpleNamespace

import pytest

pytest.importorskip("robot_executor_msgs.msg")
from robot_executor_msgs.msg import ManipulationResponse
from spot_tools_ros.live_approval import LiveApproval


class Node:
    def create_publisher(self, *args):
        return SimpleNamespace(publish=self.publish)

    def create_subscription(self, *args):
        return object()

    def publish(self, msg):
        self.request = msg


def test_response_is_correlated_and_rejection_is_terminal():
    node = Node()
    feedback = SimpleNamespace(current_plan_id="plan", break_out_of_waiting_loop=False)
    gate = LiveApproval(node, feedback)
    gate.pending = ("request", "plan")
    gate.receive(ManipulationResponse(request_id="old", plan_id="plan", approve=True))
    assert not gate.event.is_set()

    def publish(msg):
        gate.receive(
            ManipulationResponse(
                request_id=msg.request_id, plan_id=msg.plan_id, approve=False
            )
        )

    gate.publisher.publish = publish
    with pytest.raises(RuntimeError, match="cancelled"):
        gate.ask("PLACE", "mug")
    assert feedback.break_out_of_waiting_loop
    assert gate.pending is None
