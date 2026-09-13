import time
from itertools import zip_longest

import numpy as np
import rclpy.time
import tf2_ros
import tf_transformations
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from tf_transformations import euler_from_quaternion, quaternion_from_euler


def waypoints_to_path(fixed_frame, waypoints):
    now = rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()

    frame_name = fixed_frame
    path_viz = Path()
    path_viz.header.stamp = now
    path_viz.header.frame_id = frame_name

    q = [0, 0, 0, 1]
    for w, wnext in zip_longest(waypoints, waypoints[1:]):
        p = PoseStamped()
        p.header.frame_id = frame_name
        p.header.stamp = now
        p.pose.position.x = w[0]
        p.pose.position.y = w[1]
        p.pose.position.z = 0.0

        if len(w) == 3:
            q = quaternion_from_euler(0, 0, float(w[2]))
        elif wnext is not None:
            d_next = wnext[:2] - w[:2]
            theta = np.arctan2(d_next[1], d_next[0])
            q = quaternion_from_euler(0, 0, theta)
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]

        path_viz.poses.append(p)
    return path_viz


def path_to_waypoints(path):
    # NOTE: Currently ignores the frame that the path is in
    waypoints = []
    for p in path.poses:
        quat = p.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        _, _, psi = euler_from_quaternion(q)
        waypoints.append([p.pose.position.x, p.pose.position.y, psi])

    return np.array(waypoints)


def pose_to_homo(pose, quat):
    """
    Input:
        - pose: list [x, y, z]
        - quat: ros2 geometry_msgs.msg.Quaternion
    """
    # Convert pose and quaternion to a 4x4 homogeneous transformation matrix
    trans = np.array(pose)
    rot_mat = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    homo_mat = np.eye(4)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat


def get_tf_pose(tf_buffer, parent_frame: str, child_frame: str, get_euler=False):
    """
    Looks up the transform from parent_frame to child_frame and returns [x, y, z, yaw].

    """

    try:
        now = rclpy.time.Time()
        tf_buffer.can_transform(
            parent_frame,
            child_frame,
            now,
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        transform = tf_buffer.lookup_transform(parent_frame, child_frame, now)

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to Euler angles
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        if get_euler:
            return np.array([translation.x, translation.y, translation.z]), (
                roll,
                pitch,
                yaw,
            )
        return np.array([translation.x, translation.y, translation.z]), rotation

    except tf2_ros.TransformException as e:
        print(f"Transform error: {e}")
        raise
