"""Per-robot marker colors: to_viz_msg(color=...) and the robot_color param."""

import numpy as np
import pytest
from robot_executor_interface_ros.action_descriptions_ros import (
    to_color_rgba,
    to_viz_msg,
)
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)
from spot_tools_ros.spot_executor_ros import RosFeedbackCollector, parse_robot_color


def _seq():
    p = np.array([0.0, 0.0, 0.0])
    q = np.array([1.0, 2.0, 0.5])
    return ActionSequence(
        plan_id="p",
        robot_name="r",
        actions=[
            Follow(frame="map", path2d=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])),
            Gaze(
                frame="map", robot_point=p, gaze_point=q, object_id="o", stow_after=True
            ),
            Pick(
                frame="map",
                object_class="mug",
                robot_point=p,
                object_point=q,
                object_id="o",
            ),
            Place(
                frame="map",
                object_class="mug",
                robot_point=p,
                object_point=q,
                object_id="o",
            ),
        ],
    )


def _rgba(c):
    return (round(c.r, 4), round(c.g, 4), round(c.b, 4), round(c.a, 4))


def test_default_colors_unchanged():
    ma = to_viz_msg(_seq(), "r")
    assert [m.type for m in ma.markers] == [
        Marker.LINE_STRIP,
        Marker.SPHERE,
        Marker.SPHERE,
        Marker.ARROW,
        Marker.ARROW,
        Marker.ARROW,
    ]
    assert [_rgba(m.color) for m in ma.markers] == [
        (0, 1, 0, 1),  # path
        (1, 0, 0, 0.5),  # start
        (0, 0, 1, 0.5),  # end
        (1, 0, 0, 1),  # gaze
        (0, 1, 0, 1),  # pick
        (0, 0, 1, 1),  # place
    ]


@pytest.mark.parametrize(
    "color",
    [(0.2, 0.4, 0.6), (0.2, 0.4, 0.6, 1.0), ColorRGBA(r=0.2, g=0.4, b=0.6, a=1.0)],
)
def test_robot_color_tints_everything_and_labels_arrows(color):
    ma = to_viz_msg(_seq(), "r", color=color)
    shapes = [m for m in ma.markers if m.type != Marker.TEXT_VIEW_FACING]
    labels = [m for m in ma.markers if m.type == Marker.TEXT_VIEW_FACING]
    for m in shapes:
        alpha = 0.5 if m.type == Marker.SPHERE else 1.0
        assert _rgba(m.color) == (0.2, 0.4, 0.6, alpha)
    assert [m.text for m in labels] == ["gaze", "pick", "place"]
    # Labels share their arrow's namespace and do not collide on id.
    ids = {(m.ns, m.id) for m in ma.markers}
    assert len(ids) == len(ma.markers)


def test_to_color_rgba_hex_and_errors():
    assert _rgba(to_color_rgba("#ff0000", alpha=0.5)) == (1, 0, 0, 0.5)
    assert to_color_rgba(None) is None
    with pytest.raises(ValueError):
        to_color_rgba((1.0, 0.0))


def test_parse_robot_color_param():
    assert parse_robot_color("#ff0000") == (1.0, 0.0, 0.0)
    assert parse_robot_color("color:#00ff00") == (0.0, 1.0, 0.0)
    for v in ["", "none", "color:none", "color:#zzzzzz", None]:
        assert parse_robot_color(v) is None


def test_feedback_collector_colors(tmp_path):
    default = RosFeedbackCollector("odom", str(tmp_path))
    assert default.progress_color == [0, 1, 1]
    assert default.target_color == [1, 0, 1]
    tinted = RosFeedbackCollector("odom", str(tmp_path), robot_color=(1.0, 0.5, 0.0))
    assert tinted.progress_color == [1.0, 0.5, 0.0]
    assert tinted.target_color == pytest.approx([0.55, 0.275, 0.0])
