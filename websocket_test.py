import argparse
import asyncio
import websockets
import json


async def ros2_bridge_client_subscription(server_ip: str, server_port: int):
    uri = f"ws://{server_ip}:{server_port}"
    async with websockets.connect(uri) as websocket:
        # Subscribe to a ROS2 topic via the bridge
        await websocket.send(json.dumps({
            "op": "subscribe",
            "topic": "/client_count"
        }))

        while True:
            try:
                message = await websocket.recv()
                print("Received message:", message)
                # Process the received message as needed
            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed normally")
                break
            except websockets.exceptions.ConnectionClosedError:
                print("Connection closed unexpectedly")
                break
            except Exception as e:
                print("An error occurred:", e)
                break


async def publish_joint_trajectory_point(server_ip: str, server_port: int):
    uri = f"ws://{server_ip}:{server_port}"
    async with websockets.connect(uri) as websocket:
        # Create and publish a JointTrajectoryPoint message
        message = {
            "op": "publish",
            "topic": "/joint_trajectory",
            "msg": {
                "header": {
                    "stamp": {
                        "sec": 0,
                        "nanosec": 0
                    },
                    "frame_id": "world"
                },
                "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5"],  # Replace with actual joint names
                "points": [
                    {
                        "positions": [1.0, 2.0, 3.0, 4.0, 5.0],  # Example positions for the joints
                        "velocities": [0.0, 0.0, 0.0, 0.0, 0.0],  # Example velocities for the joints
                        "time_from_start": {"sec": 1, "nanosec": 0}  # Example duration
                    }
                ]
            }
        }

        await websocket.send(json.dumps(message))
        print("Published JointTrajectoryPoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish JointTrajectoryPoint data via ROS2 Bridge WebSocket")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="ROS2 Bridge server IP address")
    parser.add_argument("--port", type=int, default=9090, help="ROS2 Bridge server port")
    args = parser.parse_args()

    uri = "ws://127.0.0.1:9090"  # Replace with actual IP and port
    # asyncio.get_event_loop().run_until_complete(ros2_bridge_client_subscription(args.ip, args.port))
    asyncio.get_event_loop().run_until_complete(publish_joint_trajectory_point(args.ip, args.port))
