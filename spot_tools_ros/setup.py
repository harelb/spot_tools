import pathlib
import shutil

from setuptools import find_packages, setup

package_name = "spot_tools_ros"
# resolve is required for symlink installs
curr_path = pathlib.Path(__file__).resolve().parent

# copies all base meshes to directories containing unique colors
template_path = curr_path / "meshes" / "templates"
output_path = curr_path / "meshes" / "visual"
materials = output_path.rglob("spot.mtl")
for material in materials:
    shutil.copytree(template_path, material.parent, dirs_exist_ok=True)


def get_share_info(top_level, pattern, dest=None):
    dest = pathlib.Path("share") / package_name if dest is None else pathlib.Path(dest)
    files = [x.relative_to(curr_path) for x in (curr_path / top_level).rglob(pattern)]
    parent_map = {}
    for x in files:
        key = str(dest / x.parent)
        parent_map[key] = parent_map.get(key, []) + [str(x)]
    return [(k, v) for k, v in parent_map.items()]


launch_files = get_share_info("launch", "*.launch.yaml")
config_files = get_share_info("config", "*.yaml")
config_files_csv = get_share_info("config", "*.csv")
rviz_files = get_share_info("rviz", "*.rviz")
urdf_files = get_share_info("urdf", "*.urdf") + get_share_info("urdf", "*.xacro")
mesh_files = (
    get_share_info("meshes/visual", "*.obj")
    + get_share_info("meshes", "*.mtl")
    + get_share_info("meshes/collision", "*.obj")
)
data_files = (
    [
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    + launch_files
    + config_files
    + config_files_csv
    + rviz_files
    + urdf_files
    + mesh_files
)

setup(
    name=package_name,
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aaron",
    maintainer_email="aaronray@mit.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "spot_executor_node = spot_tools_ros.spot_executor_ros:main",
            "isaac_spot_executor = spot_tools_ros.isaac_executor:main",
            "isaac_spot_sensors = spot_tools_ros.isaac_sensors:main",
            "spot_twist_node = spot_tools_ros.spot_twist_node:main",
            "spot_sensor_node = spot_tools_ros.spot_sensors:main",
            "fake_occupancy_publisher = spot_tools_ros.fake_occupancy_publisher:main",
            "fake_path_publisher = spot_tools_ros.fake_path_publisher:main",
        ],
    },
)
