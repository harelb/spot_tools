import time
from functools import singledispatch

import numpy as np
import rclpy.time
from geometry_msgs.msg import Point
from robot_executor_msgs.msg import ActionMsg, ActionSequenceMsg
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)
from spot_tools_ros.utils import path_to_waypoints, waypoints_to_path


# get time message
def gtm():
    return rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()


@singledispatch
def to_msg(action):
    raise Exception(f"Unknown action type {type(action)}")


@singledispatch
def to_viz_msg(action, marker_ns, color=None):
    """Markers for ``action``.

    ``color`` (rgb/rgba floats in [0, 1], "#rrggbb", or a std_msgs ColorRGBA) tints
    every marker with the robot's color and adds a text label naming the action
    kind; ``None`` keeps the fixed per-action-type colors.
    """
    raise Exception(f"Unknown action type {type(action)}")


def to_color_rgba(color, alpha=None):
    """Normalize a robot color to ColorRGBA (None stays None).

    ``alpha`` overrides the input's alpha (spheres are drawn half transparent).
    """
    if color is None:
        return None
    if isinstance(color, ColorRGBA):
        rgba = (color.r, color.g, color.b, color.a)
    elif isinstance(color, str):
        h = color.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError(f"not a #rrggbb color: {color!r}")
        rgba = tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)
    else:
        rgba = tuple(float(c) for c in color)
        if len(rgba) == 3:
            rgba += (1.0,)
        if len(rgba) != 4:
            raise ValueError(f"expected an rgb or rgba tuple, got {color!r}")
    if alpha is not None:
        rgba = rgba[:3] + (float(alpha),)
    return ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])


def _label_marker(frame, marker_ns, marker_id, text, position, color):
    """View-facing text above ``position`` so the action kind survives recoloring."""
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = marker_id
    m.type = m.TEXT_VIEW_FACING
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.pose.position.x = float(position[0])
    m.pose.position.y = float(position[1])
    m.pose.position.z = float(position[2]) + 0.5
    m.scale.z = 0.4  # text height
    m.color = to_color_rgba(color, alpha=1.0)
    m.text = text
    return m


def _arrow_markers(action, marker_ns, color, target_point, default_rgb, label):
    """Robot point -> target arrow shared by Gaze/Pick/Place (plus label if colored)."""
    pt1 = Point()
    pt1.x = action.robot_point[0]
    pt1.y = action.robot_point[1]
    pt1.z = action.robot_point[2]
    pt2 = Point()
    pt2.x = target_point[0]
    pt2.y = target_point[1]
    pt2.z = target_point[2]

    m = Marker()
    m.header.frame_id = action.frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = 0
    m.type = m.ARROW
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.1  # shaft diameter
    m.scale.y = 0.2  # head diameter
    if color is None:
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = default_rgb
    else:
        m.color = to_color_rgba(color, alpha=1.0)
    m.points = [pt1, pt2]

    if color is None:
        return [m]
    return [m, _label_marker(action.frame, marker_ns, 1, label, target_point, color)]


@to_msg.register
def _(action: ActionSequence):
    msg = ActionSequenceMsg()
    msg.plan_id = action.plan_id
    msg.robot_name = action.robot_name
    msg.header.stamp = gtm()

    print([to_msg(a) for a in action.actions])
    msg.actions = [to_msg(a) for a in action.actions]
    return msg


@to_viz_msg.register
def _(action: ActionSequence, marker_ns, color=None):
    ma = MarkerArray()
    for ix, a in enumerate(action.actions):
        ma.markers += to_viz_msg(a, marker_ns + f"/{ix}", color=color)
    return ma


def from_msg(msg):
    actions = []
    for a in msg.actions:
        match a.action_type:
            case a.FOLLOW:
                actions.append(follow_from_msg(a))
            case a.PICK:
                actions.append(pick_from_msg(a))
            case a.PLACE:
                actions.append(place_from_msg(a))
            case a.GAZE:
                actions.append(gaze_from_msg(a))
            case _:
                raise Exception(f"Received invalid action type {a.action_type}")
    return ActionSequence(
        plan_id=msg.plan_id, robot_name=msg.robot_name, actions=actions
    )


def follow_from_msg(msg):
    return Follow(frame=msg.path.header.frame_id, path2d=path_to_waypoints(msg.path))


@to_msg.register
def _(action: Follow):
    msg = ActionMsg()
    msg.action_type = msg.FOLLOW
    msg.path = waypoints_to_path(action.frame, action.path2d)
    return msg


@to_viz_msg.register
def _(action: Follow, marker_ns, color=None):
    points = []
    for p in action.path2d:
        pt = Point()
        pt.x = p[0]
        pt.y = p[1]
        pt.z = 0.0
        points.append(pt)
    m = Marker()
    m.header.frame_id = action.frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = 0
    m.type = m.LINE_STRIP
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.2
    m.scale.y = 0.2
    m.color.a = 1.0
    m.color.r = 0.0
    m.color.g = 1.0
    m.color.b = 0.0
    if color is not None:
        m.color = to_color_rgba(color, alpha=1.0)
    m.points = points

    start = Marker()
    start.header.frame_id = action.frame
    start.header.stamp = gtm()
    start.ns = marker_ns
    start.id = 1
    start.type = m.SPHERE
    start.action = m.ADD
    start.pose.orientation.w = 1.0
    start.scale.x = 0.4
    start.scale.y = 0.4
    start.scale.z = 0.4
    start.color.a = 0.5
    start.color.r = 1.0
    start.color.g = 0.0
    start.color.b = 0.0
    if color is not None:
        start.color = to_color_rgba(color, alpha=0.5)
    start.pose.position.x = points[0].x
    start.pose.position.y = points[0].y
    start.pose.position.z = points[0].z

    end = Marker()
    end.header.frame_id = action.frame
    end.header.stamp = gtm()
    end.ns = marker_ns
    end.id = 2
    end.type = m.SPHERE
    end.action = m.ADD
    end.pose.orientation.w = 1.0
    end.scale.x = 0.4
    end.scale.y = 0.4
    end.scale.z = 0.4
    end.color.a = 0.5
    end.color.r = 0.0
    end.color.g = 0.0
    end.color.b = 1.0
    if color is not None:
        end.color = to_color_rgba(color, alpha=0.5)
    end.pose.position.x = points[-1].x
    end.pose.position.y = points[-1].y
    end.pose.position.z = points[-1].z

    return [m, start, end]


def gaze_from_msg(msg):
    robot_point = np.array([msg.robot_point.x, msg.robot_point.y, msg.robot_point.z])
    gaze_point = np.array([msg.gaze_point.x, msg.gaze_point.y, msg.gaze_point.z])
    stow_after = msg.place_frame == "STOW"
    object_id = msg.object_id

    return Gaze(
        frame=msg.gaze_frame,
        robot_point=robot_point,
        gaze_point=gaze_point,
        stow_after=stow_after,
        object_id=object_id,
    )


@to_msg.register
def _(action: Gaze):
    msg = ActionMsg()
    msg.action_type = msg.GAZE

    msg.robot_point.x = action.robot_point[0]
    msg.robot_point.y = action.robot_point[1]
    msg.robot_point.z = action.robot_point[2]

    msg.gaze_point.x = action.gaze_point[0]
    msg.gaze_point.y = action.gaze_point[1]
    msg.gaze_point.z = action.gaze_point[2]

    msg.place_frame = "STOW" if action.stow_after else "NO_STOW"

    msg.object_id = action.object_id

    return msg


@to_viz_msg.register
def _(action: Gaze, marker_ns, color=None):
    return _arrow_markers(
        action, marker_ns, color, action.gaze_point, (1.0, 0.0, 0.0), "gaze"
    )


def pick_from_msg(msg):
    robot_point = np.array([msg.robot_point.x, msg.robot_point.y, msg.robot_point.z])
    object_point = np.array(
        [msg.object_point.x, msg.object_point.y, msg.object_point.z]
    )

    return Pick(
        frame=msg.pick_frame,
        object_class=msg.object_class,
        robot_point=robot_point,
        object_point=object_point,
        object_id=msg.object_id,
    )


@to_msg.register
def _(action: Pick):
    msg = ActionMsg()
    msg.action_type = msg.PICK

    msg.robot_point.x = action.robot_point[0]
    msg.robot_point.y = action.robot_point[1]
    msg.robot_point.z = action.robot_point[2]

    msg.object_point.x = action.object_point[0]
    msg.object_point.y = action.object_point[1]
    msg.object_point.z = action.object_point[2]

    msg.object_class = action.object_class
    msg.object_id = action.object_id

    return msg


@to_viz_msg.register
def _(action: Pick, marker_ns, color=None):
    return _arrow_markers(
        action, marker_ns, color, action.object_point, (0.0, 1.0, 0.0), "pick"
    )


def place_from_msg(msg):
    robot_point = np.array([msg.robot_point.x, msg.robot_point.y, msg.robot_point.z])
    object_point = np.array(
        [msg.object_point.x, msg.object_point.y, msg.object_point.z]
    )

    return Place(
        frame=msg.place_frame,
        object_class=msg.object_class,
        robot_point=robot_point,
        object_point=object_point,
        object_id=msg.object_id,
    )


@to_msg.register
def _(action: Place):
    msg = ActionMsg()
    msg.action_type = msg.PLACE

    msg.robot_point.x = action.robot_point[0]
    msg.robot_point.y = action.robot_point[1]
    msg.robot_point.z = action.robot_point[2]

    msg.object_point.x = action.object_point[0]
    msg.object_point.y = action.object_point[1]
    msg.object_point.z = action.object_point[2]

    msg.object_id = action.object_id

    return msg


@to_viz_msg.register
def _(action: Place, marker_ns, color=None):
    return _arrow_markers(
        action, marker_ns, color, action.object_point, (0.0, 0.0, 1.0), "place"
    )
