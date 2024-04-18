import argparse
import asyncio
from bcolors import Bcolors
import math
import json
import sys
import time
import websockets


class TrajectoryPublisher:
    def __init__(self, server_ip: str, server_port: int, interval: float = 0.01):
        self.server_ip = server_ip
        self.server_port = server_port
        self.interval = interval
        self.websocket = None
        self.start_time_ns = time.time_ns()
        self.positions_left = [math.radians(x) for x in [170, 10, 100, 150, 30]]
        self.velocities_left = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.positions_right = [math.radians(x) for x in [10, 170, 80, 30, 30]]
        self.velocities_right = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.advertise_joint_trajectory_msg = {
            "op": "advertise",
            "topic": "/joint_trajectory",
            "type": "trajectory_msgs/JointTrajectory"
        }
        self.advertise_joint_trajectory_left_msg = {
            "op": "advertise",
            "topic": "/joint_trajectory_point_left",
            "type": "trajectory_msgs/JointTrajectoryPoint"
        }
        self.advertise_joint_trajectory_right_msg = {
            "op": "advertise",
            "topic": "/joint_trajectory_point_right",
            "type": "trajectory_msgs/JointTrajectoryPoint"
        }

    def create_pub_trajectory_msg(self, is_left: bool, positions: list, velocities: list,
                                  frame_id: str = "arm") -> dict:
        """
        This function creates a trajectory message in python dictionary format.

        Parameters
        ----------
        is_left: bool
            True if this is the data for left arm, false otherwise.
        positions: list
            The trajectory positions of the arm.
        velocities: list
            The trajectory velocities of the arm.
        frame_id: str
            The name of the data

        Returns
        -------
        message: dict
            The corresponding trajectory_msgs/JointTrajectory message in dictionary format.
        """
        # Get current time
        nsecs = int(time.time_ns())
        # Duration time from start
        dur_nsecs = nsecs - self.start_time_ns

        secs = nsecs // int(1e9)
        nsecs = nsecs % int(1e9)
        dur_secs = dur_nsecs // int(1e9)
        dur_nsecs = dur_nsecs % int(1e9)

        message = {
            "op": "publish",
            "topic": "/joint_trajectory_left" if is_left else "/joint_trajectory_right",
            "msg": {
                # Ref: https://docs.ros.org/en/noetic/api/std_msgs/html/msg/Header.html
                "header": {
                    # Two-integer timestamp that is expressed as:
                    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
                    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
                    # time-handling sugar is provided by the client library
                    "stamp": {
                        "secs": secs,
                        "nsecs": nsecs
                    },
                    "frame_id": frame_id
                },
                "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5"],  # Replace with actual joint names
                # Ref: https://docs.ros.org/en/noetic/api/trajectory_msgs/html/msg/JointTrajectoryPoint.html
                "points": [
                    {
                        "positions": positions,
                        "velocities": velocities,
                        # "accelerations": [],
                        # "effort": [],
                        "time_from_start": {
                            "secs": dur_secs,
                            "nsecs": dur_nsecs
                        }
                    }
                ]
            }
        }
        return message

    def create_pub_trajectory_point_msg(self, is_left: bool, positions: list, velocities: list) -> dict:
        """
        This function creates a trajectory point message in python dictionary format.

        Parameters
        ----------
        is_left: bool
            True if this is the data for left arm, false otherwise.
        positions: list
            The trajectory positions of the arm.
        velocities: list
            The trajectory velocities of the arm.

        Returns
        -------
        message: dict
            The corresponding trajectory_msgs/JointTrajectoryPoint message in dictionary format.
        """
        # Get current time
        nsecs = int(time.time_ns())
        # Duration time from start
        dur_nsecs = nsecs - self.start_time_ns

        dur_secs = dur_nsecs // int(1e9)
        dur_nsecs = dur_nsecs % int(1e9)

        message = {
            "op": "publish",
            "topic": "/joint_trajectory_point_left" if is_left else "/joint_trajectory_point_right",
            "msg": {
                "positions": positions,
                "velocities": velocities,
                # "accelerations": [],
                # "effort": [],
                "time_from_start": {
                    "secs": dur_secs,
                    "nsecs": dur_nsecs
                }
            }
        }
        return message

    async def connect_websocket(self):
        uri = f"ws://{self.server_ip}:{self.server_port}"
        self.websocket = await websockets.connect(uri)

    async def advertise_topic(self):
        await self.websocket.send(json.dumps(self.advertise_joint_trajectory_left_msg))
        await self.websocket.send(json.dumps(self.advertise_joint_trajectory_right_msg))
        # print("Sent advertise message successfully.")

    async def publish_trajectory_point(self, is_left: bool, pos: list, vel: list):
        # Create and publish a JointTrajectoryPoint message
        # Ref: https://docs.ros.org/en/noetic/api/trajectory_msgs/html/msg/JointTrajectory.html
        message = self.create_pub_trajectory_point_msg(is_left=is_left, positions=pos, velocities=vel)
        await self.websocket.send(json.dumps(message))
        # print("Sent message", json.dumps(message, indent=4))

    async def run(self):
        await self.connect_websocket()
        await self.advertise_topic()
        while True:
            # Get positions and velocities from mediapipe in controller.
            await self.publish_trajectory_point(is_left=True, pos=self.positions_left, vel=self.velocities_left)
            await self.publish_trajectory_point(is_left=False, pos=self.positions_right, vel=self.velocities_right)
            await asyncio.sleep(self.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish JointTrajectoryPoint data via ROS2 Bridge WebSocket")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="ROS2 Bridge server IP address")
    parser.add_argument("--port", type=int, default=9090, help="ROS2 Bridge server port")
    parser.add_argument("--interval", type=float, default=0.01, help="The interval between publishing data")
    args = parser.parse_args()

    trajectory_publisher = TrajectoryPublisher(args.ip, args.port, args.interval)
    try:
        asyncio.get_event_loop().run_until_complete(trajectory_publisher.run())
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"{Bcolors.FAIL}{e}{Bcolors.ENDC}", file=sys.stderr)
        print(f"{Bcolors.FAIL}Connection of ROS bridge is closed unexpectedly{Bcolors.ENDC}", file=sys.stderr)
        quit(1)
    except ConnectionRefusedError as e:
        print(f"{Bcolors.FAIL}{e}{Bcolors.ENDC}", file=sys.stderr)
        print(f"{Bcolors.FAIL}Error: Connection is refused, please check the connection of ROS bridge{Bcolors.ENDC}",
              file=sys.stderr)
        quit(1)
