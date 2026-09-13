"""Opt-in front-ZED ROS reception test; no motion or Hydra integration.

Source the isolated ROS environment before running. Starts the production sensor
publisher, validates matching RGB/depth/calibration messages, and records loss.
"""
import argparse
import hashlib
import os
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

import numpy as np
from spot_tools_ros.isaac_sensors import SensorClient


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=25.)
    parser.add_argument('--wait', type=float, default=600.)
    args = parser.parse_args()
    if args.duration < 5:
        parser.error('duration must be at least five seconds')
    deadline = time.monotonic() + args.wait
    while True:
        try:
            client = SensorClient('http://127.0.0.1:9250')
            if client.next() is None:
                raise RuntimeError('camera not yet ready')
            break
        except (URLError, TimeoutError, RuntimeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    rclpy.init()
    node = Node('isaac_rgbd_acceptance')
    pending, complete = {}, []
    received = {kind: set() for kind in ('rgb', 'depth', 'info')}
    started = time.monotonic()

    def accept(kind, msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        assert stamp not in received[kind], 'duplicate source timestamp'
        received[kind].add(stamp)
        bucket = pending.setdefault(stamp, {})
        bucket[kind] = msg
        if len(bucket) == 3:
            rgb, depth, info = (bucket[k] for k in ('rgb', 'depth', 'info'))
            assert rgb.header.frame_id == depth.header.frame_id == info.header.frame_id == 'hamilton/mapping_optical'
            assert (rgb.width, rgb.height) == (depth.width, depth.height) == (info.width, info.height) == (1280, 720)
            assert rgb.encoding == 'rgb8' and depth.encoding == '32FC1'
            assert len(rgb.data) == rgb.height * rgb.width * 3
            d = np.frombuffer(depth.data, dtype='<f4')
            assert d.size == depth.height * depth.width and np.isfinite(d).all() and (d >= 0).all()
            assert info.k[0] > 0 and info.k[4] > 0
            assert not complete or stamp > complete[-1][0], 'nonchronological observation'
            complete.append((stamp, time.monotonic() - started))
            del pending[stamp]
        while len(pending) > 16:
            del pending[min(pending)]

    for kind, typ, suffix in [('rgb', Image, 'rgb'), ('depth', Image, 'depth'), ('info', CameraInfo, 'camera_info')]:
        node.create_subscription(typ, '/hamilton/mapping/' + suffix,
                                 lambda msg, key=kind: accept(key, msg), qos_profile_sensor_data)
    publisher_log = args.output.with_suffix('.publisher.log')
    with publisher_log.open('w') as log:
        publisher = subprocess.Popen([sys.executable, '-u', '-m', 'spot_tools_ros.isaac_sensors',
                                      '--duration', str(args.duration)], stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.monotonic() - started < args.duration + 8:
                rclpy.spin_once(node, timeout_sec=.05)
                if publisher.poll() is not None:
                    # Drain messages that arrived just before publisher shutdown.
                    drain = time.monotonic() + 1
                    while time.monotonic() < drain:
                        rclpy.spin_once(node, timeout_sec=.05)
                    break
            if publisher.poll() is None:
                raise TimeoutError('bounded sensor publisher did not finish')
            assert publisher.returncode == 0, publisher_log.read_text()
            union = set.union(*received.values())
            result = dict(scope='stationary live Isaac to ROS; no Hydra', session=client.transport.session,
                          source='front_zed_color_image', resolution=[1280, 720],
                          received={k: len(v) for k, v in received.items()}, aligned_observations=len(complete),
                          incomplete_observations=len(union) - len(complete),
                          observation_stamps_ns=[s for s, _ in complete],
                          receipt_elapsed_s=[t for _, t in complete])
            result['rmw_implementation'] = rclpy.utilities.get_rmw_implementation_identifier()
            result['transport_configs'] = {
                key: dict(path=os.environ[key], sha256=hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest())
                for key in ('ZENOH_SESSION_CONFIG_URI', 'ZENOH_ROUTER_CONFIG_URI', 'FASTRTPS_DEFAULT_PROFILES_FILE')
                if os.environ.get(key)
            }
            result['ros_domain_id'] = os.environ.get('ROS_DOMAIN_ID')
            result['passed'] = len(complete) >= 20 and len(complete) / max(1, len(union)) >= .98
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)
            assert result['passed'], 'unacceptable RGB-D transport loss; inspect receipt'
        finally:
            if publisher.poll() is None:
                publisher.terminate()
                try:
                    publisher.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    publisher.kill()
                    publisher.wait()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
