"""Check compiled-sequence transport in separate ROS processes, without dispatch.

Uses dedicated probe topics. An echo is transport receipt, never action success.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time


def receiver(expected, wire, ready):
    import rclpy
    from rclpy.node import Node
    from robot_executor_msgs.msg import ActionSequenceMsg, ActionResultMsg
    from rclpy.serialization import deserialize_message
    from robot_executor_interface_ros.action_descriptions_ros import from_msg
    rclpy.init()
    node = Node('isaac_control_probe_receiver')
    publisher = node.create_publisher(ActionResultMsg, '/isaac_transport_probe/result', 10)
    got = []
    def receive(msg):
        # CDR alignment padding is not semantic and need not serialize identically.
        assert msg == deserialize_message(wire, ActionSequenceMsg), 'compiled ROS fields changed'
        assert from_msg(msg).plan_id == expected['plan_id']
        got.append(msg.plan_id)
        reply = ActionResultMsg()
        reply.plan_id = msg.plan_id
        reply.robot_name = msg.robot_name
        reply.status = 'TRANSPORT_RECEIVED'
        reply.detail = 'No action executor or hardware was invoked'
        publisher.publish(reply)
    node.create_subscription(ActionSequenceMsg, '/isaac_transport_probe/sequence', receive, 10)
    ready.set()
    end = time.monotonic() + 15
    while time.monotonic() < end and not got:
        rclpy.spin_once(node, timeout_sec=.05)
    assert got, 'compiled sequence not received'
    time.sleep(.5)
    node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID') == '62'
    expected = json.loads(args.receipt.read_text())['compiled_sequence']
    import rclpy
    from rclpy.node import Node
    from robot_executor_msgs.msg import ActionSequenceMsg, ActionResultMsg
    from robot_executor_interface.sequence_wire import decode_sequence
    from robot_executor_interface_ros.action_descriptions_ros import to_msg_at
    rclpy.init()
    node = Node('isaac_control_probe_sender')
    publisher = node.create_publisher(ActionSequenceMsg, '/isaac_transport_probe/sequence', 10)
    replies = []
    node.create_subscription(ActionResultMsg, '/isaac_transport_probe/result', replies.append, 10)
    from rclpy.serialization import serialize_message
    message = to_msg_at(decode_sequence(expected), node.get_clock().now().to_msg())
    wire = serialize_message(message)
    context = multiprocessing.get_context('spawn')
    ready = context.Event()
    worker = context.Process(target=receiver, args=(expected, wire, ready))
    worker.start()
    try:
        assert ready.wait(10), 'receiver did not initialize'
        end = time.monotonic() + 10
        while not publisher.get_subscription_count() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        assert publisher.get_subscription_count(), 'probe discovery failed'
        publisher.publish(message)
        while not replies and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        assert replies and replies[0].plan_id == expected['plan_id']
        assert replies[0].status == 'TRANSPORT_RECEIVED'
        worker.join(5)
        assert worker.exitcode == 0
        result = dict(scope='compiled sequence + result transport only; no hardware execution',
                      rmw_implementation=rclpy.utilities.get_rmw_implementation_identifier(),
                      source_ros_serialization_sha256=hashlib.sha256(wire).hexdigest(),
                      all_ros_fields_equal=True,
                      plan_id=expected['plan_id'], action_count=len(expected['actions']), passed=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
