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

![j1](https://latex.codecogs.com/svg.image?\begin{cases}0=0m&plus;k\\90=65m&plus;k\end{cases}\Rightarrow\begin{cases}m=18/13\\k=0\end{cases})

#### J3

![j3](https://latex.codecogs.com/svg.image?\begin{cases}45=30m&plus;k\\180=110m&plus;k\end{cases}\Rightarrow\begin{cases}m=27/16\\k=-45/8\end{cases})

#### J4

![j4](https://latex.codecogs.com/svg.image?\begin{cases}60=70m&plus;k\\180=150m&plus;k\end{cases}\Rightarrow\begin{cases}m=3/2\\k=-45\end{cases})

### Right

#### J2

![j2r](https://latex.codecogs.com/svg.image?\begin{cases}180=0m&plus;k\\80=65m&plus;k\end{cases}\Rightarrow\begin{cases}m=-20/13\\k=180\end{cases})

#### J3

![j3r](https://latex.codecogs.com/svg.image?\begin{cases}0=110m&plus;k\\135=30m&plus;k\end{cases}\Rightarrow\begin{cases}m=-27/16\\k=1485/8\end{cases})

#### J4

![j4r](https://latex.codecogs.com/svg.image?\begin{cases}120=85m&plus;k\\0=150m&plus;k\end{cases}\Rightarrow\begin{cases}m=-24/13\\k=3600/13\end{cases})



