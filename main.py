import pose
import websocket
import argparse
import asyncio


class Controller:
    """
    This class controls the ``pose`` class and the ``websocket`` class, aims to calculate the trajectory angles
    and then publish them to the ros node via websocket ``rosbridge``.

    Attributes
    ----------
    pose: class
        The instance of the class ``pose``.
    websocket: class
        The instance of the class ``websocket``.
    positions: list
        The angles calculated by the ``pose`` class.
    velocities: list
        The velocities, we temporarily don't need to calculate them. Defaults to zero.

    Methods
    -------
    run(self):
        The main function of the controller.
    """
    def __init__(self, pose, websocket):
        self.pose = pose
        self.websocket = websocket
        self.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    async def run(self):
        await self.websocket.connect_websocket()
        await self.websocket.advertise_topic()
        while True:
            success, self.positions = self.pose.main()
            await self.websocket.publish_trajectory_point(self.positions, self.velocities)
            await asyncio.sleep(self.websocket.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish JointTrajectoryPoint data via ROS2 Bridge WebSocket")
    parser.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="ROS2 Bridge server IP address")
    parser.add_argument("-p", "--port", type=int, default=9090, help="ROS2 Bridge server port")
    parser.add_argument("-d", "--interval", type=float, default=0.01, help="The interval between publishing data")
    parser.add_argument("-c", "--camera", type=int, default=0, help="The ID of the camera. e.g. 0 for /dev/video0.")
    args = parser.parse_args()

    # MVC framework
    model = pose.PoseDetection(args.camera)
    view = websocket.TrajectoryPublisher(args.ip, args.port, args.interval)
    controller = Controller(model, view)

    asyncio.get_event_loop().run_until_complete(controller.run())
