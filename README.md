# Mediapipe Robot Arm

## Project Architecture

![robot_arm_architecture.drawio](./img/robot_arm_architecture.drawio.svg)



## Note

We don't consider the confidence of every point. This program will always publish the angles.



## Reference

- Left Arm: https://github.com/otischung/robot_arm_esp32/tree/left-arm-J5-master
- Right Arm: https://github.com/otischung/robot_arm_esp32/tree/right-arm-J5-master
- Docker Image: https://github.com/otischung/pros_AI_image
- Robot Control: https://github.com/otischung/robot_control



## Prerequisites and Installations

Below are the devices we need.

- Arduino ESP32 *2
- Arm-control board, e.g., Raspberry Pi, Jetson Orin Nano
- Camera (You can also use the camera on your laptop)
- Computing device



1. Follow the link for the left and right arms shown above to upload Arduino C++ code to ESP32.

2. Pull the docker image into the arm-control board by the following command:

   ```bash
   docker pull ghcr.io/otischung/pros_ai_image:latest
   ```

3. Download the "Robot Control" Python code, which is shown above, into the arm-control board.

4. Download this code to your computing device.

5. Create a Python virtual environment, source into it, and run `pip install -r requirements.txt`.



## Usage

1. In the arm-control board, run the robot control shell script to create the container.

2. Run `colcon build --symlink-install` to build the project "robot_control".

3. Open the ROS bridge server, arm writer, and arm keyboard inside the container.

   ```bash
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml
   ros2 run robot_control arm_writer
   ros2 run robot_control arm_keyboard
   ```

4. In the computing device, source the virtual environment, and run `python main.py`.

5. To exit the code, select the video window and then click `esc`.



- You can set the parameters in `main.py`.
  - `-i <IP>` specifies the IP address of the ROS bridge server.
  - `-p <Port>` specifies the port number of the ROS bridge server.
  - `-d <interval>` specifies the interval between publishing data, the default value is 0.01 seconds.
  - `-c <camera>` specifies the ID of the camera. e.g., 0 for `/dev/video0`.
- You can also run `python pose.py` to run the mediapipe only.



## Calibration

Use the equation

![eq](https://latex.codecogs.com/svg.image?y=mx&plus;k)

to do the calibration.

- *x* is the degree from mediapipe
- *y* is the degree for the robot arm



### Left

#### J2

![j1](https://latex.codecogs.com/svg.image?%5Cbegin%7Bcases%7D0=0m&plus;k%5C%5C90=65m&plus;k%5Cend%7Bcases%7D%5CRightarrow%5Cbegin%7Bcases%7Dm=18/13%5C%5Ck=0%5Cend%7Bcases%7D)

#### J3

![j3](https://latex.codecogs.com/svg.image?%5Cbegin%7Bcases%7D45=30m&plus;k%5C%5C180=110m&plus;k%5Cend%7Bcases%7D%5CRightarrow%5Cbegin%7Bcases%7Dm=27/16%5C%5Ck=-45/8%5Cend%7Bcases%7D)

#### J4

![j4](https://latex.codecogs.com/svg.image?%5Cbegin%7Bcases%7D60=70m&plus;k%5C%5C180=150m&plus;k%5Cend%7Bcases%7D%5CRightarrow%5Cbegin%7Bcases%7Dm=3/2%5C%5Ck=-45%5Cend%7Bcases%7D)

### Right

#### J2

![j2r](https://latex.codecogs.com/svg.image?%5Cbegin%7Bcases%7D180=0m&plus;k%5C%5C80=65m&plus;k%5Cend%7Bcases%7D%5CRightarrow%5Cbegin%7Bcases%7Dm=-20/13%5C%5Ck=180%5Cend%7Bcases%7D)

#### J3

![j3r](https://latex.codecogs.com/svg.image?%5Cbegin%7Bcases%7D0=110m&plus;k%5C%5C135=30m&plus;k%5Cend%7Bcases%7D%5CRightarrow%5Cbegin%7Bcases%7Dm=-27/16%5C%5Ck=1485/8%5Cend%7Bcases%7D)

#### J4

![j4r](https://latex.codecogs.com/svg.image?%5Cbegin%7Bcases%7D120=85m&plus;k%5C%5C0=150m&plus;k%5Cend%7Bcases%7D%5CRightarrow%5Cbegin%7Bcases%7Dm=-24/13%5C%5Ck=3600/13%5Cend%7Bcases%7D)



