# Mediapipe Robot Arm

## Note

We don't consider the confidence of every point. This program will always publish the angles.



## Reference

- Left Arm: https://github.com/otischung/robot_arm_esp32/tree/left-arm-J5-master
- Right Arm: https://github.com/otischung/robot_arm_esp32/tree/right-arm-J5-master

## Calibration

Use the equation

![eq](https://latex.codecogs.com/svg.image?y=mx&plus;k)

to do calibration.

- *x* is the degree from mediapipe
- *y* is the degree for robot arm



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



