"""Bounded native RGB-D transport from Isaac into the robot's ROS sensor path.

No image encoding, disk frames, or generated display frames. A response contains
one immutable camera snapshot with its calibration and capture-time transform.
"""
from __future__ import annotations
import argparse
import json
import struct
import time
import base64
import hashlib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import urlencode

import numpy as np


def decode_rgbd(payload, session):
    if len(payload) < 4:
        raise ValueError('truncated RGB-D header')
    length = struct.unpack('<I', payload[:4])[0]
    if not 0 < length <= 16384 or 4+length > len(payload):
        raise ValueError('invalid RGB-D header size')
    meta = json.loads(payload[4:4+length])
    if meta['protocol'] != 'spot-tools-isaac-v1' or meta['session_id'] != session:
        raise ValueError('camera episode changed')
    height, width = int(meta['height']), int(meta['width'])
    if not (0 < height <= 2160 and 0 < width <= 3840):
        raise ValueError('invalid image resolution')
    if (meta['rgb_encoding'],meta['depth_encoding'],meta['depth_units']) != ('rgb8','32FC1','m'):
        raise ValueError('unsupported sensor encodings')
    rgb_size, depth_size = height*width*3, height*width*4
    if (meta['rgb_bytes'],meta['depth_bytes']) != (rgb_size,depth_size):
        raise ValueError('sensor payload sizes disagree with calibration')
    start = 4+length
    if len(payload) != start+rgb_size+depth_size:
        raise ValueError('truncated or extra sensor bytes')
    rgb = np.frombuffer(payload,dtype=np.uint8,count=rgb_size,offset=start).reshape(height,width,3)
    depth = np.frombuffer(payload,dtype='<f4',count=height*width,offset=start+rgb_size).reshape(height,width)
    K=np.asarray(meta['K'])
    if K.shape != (3,3) or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError('invalid camera intrinsics')
    if not np.isfinite(depth).all() or (depth<0).any():
        raise ValueError('nonfinite or negative geometric depth')
    pose=np.asarray([*meta['position'],*meta['orientation_wxyz']],dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>.001:
        raise ValueError('invalid optical transform')
    if int(meta['timestamp_ns']) < 0:
        raise ValueError('invalid capture timestamp')
    return meta,rgb,depth


class SensorClient:
    def __init__(self, endpoint, source='front_zed_color_image'):
        from spot_executor.isaac_spot import IsaacTransport
        self.transport=IsaacTransport(endpoint)
        if 'raw_rgbd' not in self.transport.capabilities:
            raise RuntimeError('Isaac does not advertise raw RGB-D')
        self.source=source
        self.index=-1
        self.stamp=-1

    def next(self):
        query=urlencode(dict(session=self.transport.session,source=self.source,after=self.index))
        with urlopen(self.transport.url+'/rgbd?'+query,timeout=2) as response:
            if response.status==204:
                return None
            length=int(response.headers['Content-Length'])
            if not 4 < length <= 64*1024*1024:
                raise ValueError('sensor response exceeds memory bound')
            result=decode_rgbd(response.read(length),self.transport.session)
        meta=result[0]
        if meta['source'] != self.source or meta['frame_index']<=self.index or meta['timestamp_ns']<=self.stamp:
            raise ValueError('stale, reordered or different-source observation')
        self.index,self.stamp=meta['frame_index'],meta['timestamp_ns']
        return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--endpoint',default='http://127.0.0.1:9250')
    parser.add_argument('--robot',default='hamilton')
    parser.add_argument('--source',default='front_zed_color_image',choices=['front_zed_color_image'])
    parser.add_argument('--duration',type=float,default=0)
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--occupancy-topic',default='/hamilton/occupancy_grid')
    args=parser.parse_args()
    import os
    if os.environ.get('ROS_DOMAIN_ID') != '62':
        raise RuntimeError('Isaac sensor launch requires isolated ROS domain 62')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from geometry_msgs.msg import TransformStamped
    from rosgraph_msgs.msg import Clock
    from tf2_ros import TransformBroadcaster
    from bosdyn.api import robot_state_pb2
    from std_msgs.msg import String
    from nav_msgs.msg import OccupancyGrid
    client=SensorClient(args.endpoint,args.source)
    rclpy.init();node=Node('isaac_spot_sensors',namespace=args.robot)
    sensor='mapping'
    prefix=f'/{args.robot}/{sensor}'
    color_pub=node.create_publisher(Image,prefix+'/rgb',qos_profile_sensor_data)
    depth_pub=node.create_publisher(Image,prefix+'/depth',qos_profile_sensor_data)
    info_pub=node.create_publisher(CameraInfo,prefix+'/camera_info',qos_profile_sensor_data)
    joints_pub=node.create_publisher(JointState,f'/{args.robot}/joint_states',qos_profile_sensor_data)
    clock_pub=node.create_publisher(Clock,'/clock',10)
    telemetry=node.create_publisher(String,f'/{args.robot}/isaac_sensor_metrics',10)
    tf=TransformBroadcaster(node)
    optical=f'{args.robot}/{sensor}_optical'
    def occupancy(msg):
        # Only a registered perception grid can authorize the hardware sweep.
        if msg.header.frame_id != 'vision':
            node.get_logger().warning('Ignoring occupancy grid not registered into vision')
            return
        p=msg.info.origin.position;q=msg.info.origin.orientation
        if abs(q.x)>1e-6 or abs(q.y)>1e-6:
            node.get_logger().warning('Ignoring nonplanar occupancy grid')
            return
        import math
        cells=np.asarray(msg.data,dtype=np.int8).tobytes()
        revision=hashlib.sha256(msg.header.frame_id.encode()+cells+str(msg.header.stamp).encode()).hexdigest()
        try:
            client.transport.call('observed_grid',frame='vision',revision=revision,
                width=msg.info.width,height=msg.info.height,resolution=msg.info.resolution,
                origin_xy_yaw=[p.x,p.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)],
                cells=base64.b64encode(cells).decode())
        except (RuntimeError,ValueError) as exc:
            node.get_logger().error(f'Observed collision grid refused: {exc}')
    node.create_subscription(OccupancyGrid,args.occupancy_topic,occupancy,1)
    def stamped(ns):
        from builtin_interfaces.msg import Time
        return Time(sec=ns//1000000000,nanosec=ns%1000000000)
    def transform(child,parent,p,q,stamp):
        m=TransformStamped();m.header.frame_id=parent;m.child_frame_id=child;m.header.stamp=stamp
        m.transform.translation.x,m.transform.translation.y,m.transform.translation.z=map(float,p)
        m.transform.rotation.w,m.transform.rotation.x,m.transform.rotation.y,m.transform.rotation.z=map(float,q)
        return m
    started=time.monotonic();published=0
    try:
        while rclpy.ok() and (not args.duration or time.monotonic()-started<args.duration):
            rclpy.spin_once(node,timeout_sec=0)
            try:
                frame=client.next()
            except HTTPError as exc:
                if exc.code == 409 and client.transport.call('health').get('paused'):
                    time.sleep(.1)
                    continue
                raise
            if frame is None:
                time.sleep(.005);continue
            meta,rgb,depth=frame;stamp=stamped(meta['timestamp_ns'])
            for pixels,encoding,pub in [(rgb,'rgb8',color_pub),(depth,'32FC1',depth_pub)]:
                msg=Image();msg.header.stamp=stamp;msg.header.frame_id=optical
                msg.height,msg.width=pixels.shape[:2];msg.encoding=encoding
                msg.step=msg.width*(3 if encoding=='rgb8' else 4);msg.is_bigendian=False
                msg.data=pixels.tobytes();pub.publish(msg)
            info=CameraInfo();info.header.stamp=stamp;info.header.frame_id=optical
            info.width,info.height=meta['width'],meta['height'];info.k=np.asarray(meta['K']).ravel().tolist()
            info.r=np.eye(3).ravel().tolist();info.p=np.column_stack([np.asarray(meta['K']),np.zeros(3)]).ravel().tolist()
            info.distortion_model='plumb_bob';info.d=[0.]*5;info_pub.publish(info)
            tf.sendTransform(transform(optical,'vision',meta['position'],meta['orientation_wxyz'],stamp))
            if sensor=='mapping':
                clock_pub.publish(Clock(clock=stamp))
                state=client.transport.proto('robot_state',robot_state_pb2.RobotState)
                state_stamp=state.kinematic_state.acquisition_timestamp.ToNanoseconds()
                transforms=[]
                for child,edge in state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.items():
                    if not edge.parent_frame_name:continue
                    pose=edge.parent_tform_child;p=pose.position;q=pose.rotation
                    parent=edge.parent_frame_name
                    transforms.append(transform(f'{args.robot}/{child}',parent if parent=='vision' else f'{args.robot}/{parent}',[p.x,p.y,p.z],[q.w,q.x,q.y,q.z],stamped(state_stamp)))
                tf.sendTransform(transforms)
                msg=JointState();msg.header.stamp=stamped(state_stamp)
                msg.name=[j.name for j in state.kinematic_state.joint_states]
                msg.position=[j.position.value for j in state.kinematic_state.joint_states]
                msg.velocity=[j.velocity.value for j in state.kinematic_state.joint_states];joints_pub.publish(msg)
            published+=1
            if args.status_file and published % 5 == 0:
                receipt=dict(ready=True, session_id=client.transport.session,
                    updated_at=time.time(), published=published, elapsed_s=time.monotonic()-started,
                    last_source_index=client.index, timestamp_ns=client.stamp,
                    rmw=rclpy.get_rmw_implementation_identifier(), source=args.source)
                temp=args.status_file.with_suffix('.tmp')
                temp.write_text(json.dumps(receipt));temp.replace(args.status_file)
            if published%30==0:
                telemetry.publish(String(data=json.dumps(dict(session_id=client.transport.session,
                    published=published,elapsed_s=time.monotonic()-started,last_source_index=client.index))))
    finally:
        node.get_logger().info(f'Published {published} synchronized RGB-D observations in {time.monotonic()-started:.3f}s')
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
