import argparse
import asyncio
import json
import time
import websockets


def create_pub_trajectory_msg(start_time_ns: int, positions: list, velocities: list,
                              frame_id: str = "arm") -> dict:
    """
    This function creates a trajectory message in python dictionary format.

    Parameters
    ----------
    start_time_ns: int
        The start time of the first publish time using time.time_ns().
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
    dur_nsecs = nsecs - start_time_ns

    secs = nsecs // int(1e9)
    nsecs = nsecs % int(1e9)
    dur_secs = dur_nsecs // int(1e9)
    dur_nsecs = dur_nsecs % int(1e9)

    message = {
        "op": "publish",
        "topic": "/joint_trajectory",
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


async def publish_joint_trajectory_point(server_ip: str, server_port: int, interval: float = 0.01):
    uri = f"ws://{server_ip}:{server_port}"
    async with websockets.connect(uri) as websocket:
        # Advertise the topic first
        advertise_msg = {
            "op": "advertise",
            "topic": "/joint_trajectory",
            "type": "trajectory_msgs/JointTrajectory"
        }
        await websocket.send(json.dumps(advertise_msg))
        print("sent advertise message successfully.")

        # Start in current time in nanoseconds
        start_time_ns = time.time_ns()
        while True:
            # Create and publish a JointTrajectoryPoint message
            # Ref: https://docs.ros.org/en/noetic/api/trajectory_msgs/html/msg/JointTrajectory.html
            message = create_pub_trajectory_msg(start_time_ns,
                                                positions=[0.0, 0.0, 0.0, 0.0, 0.0],
                                                velocities=[0.0, 0.0, 0.0, 0.0, 0.0])
            await websocket.send(json.dumps(message))
            print("sent message", json.dumps(message, indent=4))

            await asyncio.sleep(interval)  # Wait for the specified interval before sending the next message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish JointTrajectoryPoint data via ROS2 Bridge WebSocket")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="ROS2 Bridge server IP address")
    parser.add_argument("--port", type=int, default=9090, help="ROS2 Bridge server port")
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(publish_joint_trajectory_point(args.ip, args.port))
