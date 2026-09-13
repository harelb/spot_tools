"""Launch the real Spot Tools executor with an enforced Isaac hardware boundary."""
import os


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '62':
        raise RuntimeError('Isaac executor requires isolated ROS domain 62')
    os.environ['SPOT_TOOLS_BACKEND_LOCK'] = 'isaac'
    from spot_tools_ros.spot_executor_ros import main as execute
    execute()


if __name__=='__main__':main()
