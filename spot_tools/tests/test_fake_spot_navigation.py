"""FakeSpot must support the calls the executor makes while following a path."""

from bosdyn.client.math_helpers import SE2Pose

from spot_executor.fake_spot import FakeSpot
from spot_skills.navigation_utils import navigate_to_absolute_pose


def test_get_state_returns_robot_state():
    spot = FakeSpot(init_pose=(0.0, 0.0, 0.0, 0.0))
    state = spot.get_state()
    assert state.HasField("manipulator_state")
    assert state.HasField("kinematic_state")


def test_navigate_to_absolute_pose_moves_fake_robot():
    # Used by follow_trajectory_continuous for every FOLLOW action; it calls
    # spot.get_state(), which FakeSpot used to raise on.
    # FakeSpot moves at most fake_speed * 0.1 m per command (it interpolates
    # instead of teleporting), so re-issue the goal the way the follower does.
    spot = FakeSpot(init_pose=(0.0, 0.0, -1.0, 0.0))
    goal = SE2Pose(x=3.0, y=-2.0, angle=0.5)
    navigate_to_absolute_pose(spot, goal)
    x, y, z, _ = spot.get_pose()
    first_step = (x**2 + y**2) ** 0.5
    assert 0.0 < first_step <= spot.fake_speed * 0.1 + 1e-9
    for _ in range(50):
        navigate_to_absolute_pose(spot, goal)
    x, y, z, yaw = spot.get_pose()
    assert abs(x - 3.0) < 1e-9 and abs(y + 2.0) < 1e-9 and z == -1.0
    assert abs(yaw - 0.5) < 1e-9
