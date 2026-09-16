"""Closed-loop ROS 2 navigation test with the fake robot: no hardware, no GPU.

Processes (all on an isolated ROS_DOMAIN_ID):
  static TF  hamilton/map -> hamilton/odom (non-identity, exercises the frame math)
  static TF  hamilton/body -> hamilton/base_link
  fake_occupancy_publisher  (L-shaped hallway in hamilton/map, optional sensing crop)
  spot_executor_node        (fake, kinematic Spot; A* mid-level planner)
  fake_path_publisher       (one straight high-level path through the wall of the L)

The harness watches TF and asserts the robot reaches the goal through free space.
"""

import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
ROBOT = "hamilton"
RES = 0.12
MAP_ORIGIN_XY = (-16.0, 0.0)  # fake_occupancy_publisher.publish_map
MAP_T_ODOM = (1.0, -2.0, 0.5)  # x, y, yaw of hamilton/odom in hamilton/map
# inside the L's vertical arm (map frame) and its horizontal arm, see create_L_shape_hallway_map
START_MAP = (-11.0, 4.0, np.pi / 2)
GOAL_MAP = (3.0, 13.5)


def se2(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


def map_to_odom(x, y, yaw):
    p = np.linalg.inv(se2(*MAP_T_ODOM)) @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1]), float(yaw - MAP_T_ODOM[2])


def hallway_grid():
    from spot_tools_ros.fake_occupancy_publisher import create_L_shape_hallway_map

    return create_L_shape_hallway_map(200, 200, RES)


def map_xy_to_cell(x, y):
    return int((y - MAP_ORIGIN_XY[1]) / RES), int((x - MAP_ORIGIN_XY[0]) / RES)


class Stack:
    def __init__(
        self,
        workdir,
        python=sys.executable,
        crop=-1.0,
        kinematic=True,
        extra_params=(),
        domain_id=77,
    ):
        self.workdir = pathlib.Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.crop = crop
        self.kinematic = kinematic
        self.extra_params = list(extra_params)
        self.procs = {}
        self.env = dict(os.environ)
        self.env["ROS_DOMAIN_ID"] = str(domain_id)
        self.env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        self.env["MPLBACKEND"] = "Agg"

    def spawn(self, name, cmd):
        log = open(self.workdir / f"{name}.log", "w")
        self.procs[name] = (
            subprocess.Popen(
                cmd,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ),
            log,
        )

    def start(self):
        x, y, yaw = MAP_T_ODOM
        self.spawn(
            "tf_map_odom",
            [
                "ros2",
                "run",
                "tf2_ros",
                "static_transform_publisher",
                "--frame-id",
                f"{ROBOT}/map",
                "--child-frame-id",
                f"{ROBOT}/odom",
                "--x",
                str(x),
                "--y",
                str(y),
                "--yaw",
                str(yaw),
            ],
        )
        self.spawn(
            "tf_body_base",
            [
                "ros2",
                "run",
                "tf2_ros",
                "static_transform_publisher",
                "--frame-id",
                f"{ROBOT}/body",
                "--child-frame-id",
                f"{ROBOT}/base_link",
            ],
        )
        self.spawn(
            "occupancy",
            [
                self.python,
                "-m",
                "spot_tools_ros.fake_occupancy_publisher",
                "--robot_name",
                ROBOT,
                "--scenario",
                "L_shape_hallway",
                "--crop_distance",
                str(self.crop),
                "--resolution",
                str(RES),
            ],
        )
        sx, sy, syaw = map_to_odom(*START_MAP)
        params = {
            "spot_ip": "fake",
            "bosdyn_client_username": "fake",
            "bosdyn_client_password": "fake",
            "follower_lookahead": 2.1,
            "goal_tolerance": 1.0,
            "odom_frame": f"{ROBOT}/odom",
            "body_frame": f"{ROBOT}/body",
            "output_dir": str(self.workdir / "executor_out"),
            "use_fake_spot_interface": True,
            "use_fake_spot_pose": True,
            "fake_spot_x": sx,
            "fake_spot_y": sy,
            "fake_spot_z": 0.0,
            "fake_spot_yaw": syaw,
            "fake_spot_kinematic": self.kinematic,
            "mid_level_planner_type": "astar",
            "lookahead_distance": 50,
            "occupancy_inflation_radius": 0.2,
            "use_cost_map": True,
            "cost_map_safe_distance": 1.0,
            "cost_map_nearest_obstacle_cost": 5.0,
        }
        args = [
            "--ros-args",
            "-r",
            f"__ns:=/{ROBOT}",
            "-r",
            "__node:=spot_executor_node",
            "-r",
            f"~/occupancy_grid:=/{ROBOT}/hydra/tsdf/occupancy",
            "-r",
            f"~/action_sequence_subscriber:=/{ROBOT}/omniplanner_node/compiled_plan_out",
        ]
        for k, v in params.items():
            args += ["-p", f"{k}:={str(v).lower() if isinstance(v, bool) else v}"]
        for p in self.extra_params:
            args += ["-p", p]
        self.spawn(
            "executor", [self.python, "-m", "spot_tools_ros.spot_executor_ros"] + args
        )

    def send_goal(self):
        cmd = [
            self.python,
            "-m",
            "spot_tools_ros.fake_path_publisher",
            str(GOAL_MAP[0]),
            str(GOAL_MAP[1]),
            "--map_frame",
            f"{ROBOT}/map",
            "--robot_name",
            ROBOT,
        ]
        subprocess.run(
            cmd,
            env=self.env,
            stdout=open(self.workdir / "path_publisher.log", "w"),
            stderr=subprocess.STDOUT,
            timeout=30,
        )

    def alive(self, name):
        return self.procs[name][0].poll() is None

    def stop(self):
        for name, (p, log) in self.procs.items():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for name, (p, log) in self.procs.items():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=5)
            log.close()

    def log(self, name):
        return (self.workdir / f"{name}.log").read_text()


def run(
    workdir,
    crop=-1.0,
    kinematic=True,
    timeout_s=240.0,
    extra_params=(),
    verbose=True,
    domain_id=77,
):
    """Returns a dict with reached/violations/paths/traceback so callers can assert on it."""
    import rclpy
    import tf2_ros
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile

    grid = hallway_grid()
    stack = Stack(
        workdir,
        crop=crop,
        kinematic=kinematic,
        extra_params=extra_params,
        domain_id=domain_id,
    )
    result = {
        "reached": False,
        "executor_done": False,
        "violations": [],
        "mlp_paths": 0,
        "traceback": "",
        "trajectory": [],
        "elapsed": None,
    }
    rclpy.init(args=None, domain_id=int(stack.env["ROS_DOMAIN_ID"]))
    node = Node("fake_nav_harness")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    latching = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

    def on_path(msg):
        result["mlp_paths"] += 1

    node.create_subscription(
        Path, f"/{ROBOT}/spot_executor_node/mlp_path_publisher", on_path, latching
    )

    def robot_in_map():
        tf = buf.lookup_transform(f"{ROBOT}/map", f"{ROBOT}/body", rclpy.time.Time())
        return np.array([tf.transform.translation.x, tf.transform.translation.y])

    try:
        stack.start()
        t0 = time.time()
        while time.time() - t0 < 60.0:  # wait for the fake robot's TF and the executor
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                robot_in_map()
                break
            except tf2_ros.TransformException:
                if not stack.alive("executor"):
                    raise RuntimeError("executor died during startup")
        else:
            raise RuntimeError("no hamilton/map -> hamilton/body TF within 60 s")
        time.sleep(
            2.0
        )  # let the occupancy subscription deliver a grid before the plan arrives
        stack.send_goal()
        t_goal = time.time()
        last_print = 0.0
        while time.time() - t_goal < timeout_s:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                p = robot_in_map()
            except tf2_ros.TransformException:
                continue
            result["trajectory"].append(p.copy())
            i, j = map_xy_to_cell(*p)
            if (
                not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1])
                or grid[i, j] != 0
            ):
                result["violations"].append(p.copy())
            if np.linalg.norm(p - GOAL_MAP) < 1.5 and not result["reached"]:
                result["reached"] = True
                result["elapsed"] = time.time() - t_goal
                t_reached = time.time()
            if result["reached"]:
                if "Finished `follow` command with return True" in stack.log(
                    "executor"
                ):
                    result["executor_done"] = True
                    break
                if time.time() - t_reached > 30.0:
                    break
            if not stack.alive("executor"):
                break
            if verbose and time.time() - last_print > 5.0:
                last_print = time.time()
                print(
                    f"t={time.time() - t_goal:5.1f}s robot(map)=({p[0]:.2f}, {p[1]:.2f}) mlp_paths={result['mlp_paths']}",
                    flush=True,
                )
    finally:
        # only tracebacks raised while the stack was running count; shutdown races do not
        log = stack.log("executor") if "executor" in stack.procs else ""
        stack.stop()
        node.destroy_node()
        rclpy.shutdown()
        # ignore numpy's "compiled using NumPy 1.x" import-shim noise from Jazzy's cv_bridge
        blocks = log.split("Traceback (most recent call last):")[1:]
        real = [
            b
            for b in blocks
            if "NumPy 1.x" not in b[:1500] and "cv_bridge" not in b[:1500]
        ]
        if real:
            result["traceback"] = ("Traceback (most recent call last):" + real[-1])[
                :3000
            ]
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=float, default=-1.0)
    ap.add_argument(
        "--teleport", action="store_true", help="use the original teleporting FakeSpot"
    )
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--workdir", default="/tmp/fake_nav_harness")
    ap.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        help="extra executor param name:=value",
    )
    ap.add_argument("--domain-id", type=int, default=77)
    a = ap.parse_args()
    r = run(
        a.workdir,
        crop=a.crop,
        kinematic=not a.teleport,
        timeout_s=a.timeout,
        extra_params=a.param,
        domain_id=a.domain_id,
    )
    traj = np.array(r["trajectory"]) if r["trajectory"] else np.zeros((0, 2))
    print(
        f"reached={r['reached']} executor_done={r['executor_done']} elapsed={r['elapsed']} "
        f"mlp_paths={r['mlp_paths']} samples={len(traj)} violations={len(r['violations'])}"
    )
    if r["traceback"]:
        print(r["traceback"])
    ok = (
        r["reached"]
        and r["executor_done"]
        and not r["violations"]
        and not r["traceback"]
    )
    sys.exit(0 if ok else 1)
