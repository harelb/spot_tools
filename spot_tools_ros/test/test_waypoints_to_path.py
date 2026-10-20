import subprocess
import sys

import numpy as np
import pytest

from spot_tools_ros.utils import waypoints_to_path


@pytest.mark.parametrize("n", [1, 2, 5])
def test_orientations_are_floats(n):
    path = waypoints_to_path("map", np.arange(2.0 * n).reshape(n, 2))
    assert len(path.poses) == n
    for p in path.poses:
        q = p.pose.orientation
        assert all(isinstance(v, float) for v in (q.x, q.y, q.z, q.w))


def test_one_waypoint_path_serializes():
    # The failure was a C-level abort, which would take pytest down with it,
    # so serialize in a child process and look at its exit status.
    code = (
        "import numpy as np\n"
        "from rclpy.serialization import serialize_message\n"
        "from spot_tools_ros.utils import waypoints_to_path\n"
        "serialize_message(waypoints_to_path('map', np.array([[1.0, 2.0]])))\n"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
