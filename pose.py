import absl.logging
import cv2
import math
import mediapipe as mp
import numpy as np
import os
import sys
import time
from bcolors import Bcolors
from enum import Enum
from typing import NamedTuple


def enum_to_dict_reverse(enum_cls: Enum) -> dict:
    """
    Function to convert enum to dictionary reversely.
    For example, if we have the enum class
    class PoseLandmark(enum.IntEnum):
        NOSE = 0
        LEFT_EYE_INNER = 1
    , then the output will be
    {0: NOSE, 1: LEFT_EYE_INNER}

    Parameters
    ----------
    enum_cls: Enum
        The given enumerate.

    Returns
    -------
    out: dict
        The reversed dictionary.
    """
    return {member.value: member.name for member in enum_cls}


pose_name = enum_to_dict_reverse(mp.solutions.pose.PoseLandmark)


def clear_screen():
    """
    Clear the entire terminal. This function supports multiple platforms including Windows and Linux.
    """
    if sys.platform == "win32":
        os.system("cls")
    elif sys.platform == "linux" or sys.platform == "darwin":
        os.system("clear")


def print_landmark(fps_cnt: int, fps: float, joint_list: list, left_positions: list, left_confidence_bool: list,
                   right_positions: list, right_confidence_bool: list):
    """
    This function prints the detected 33 landmarks.

    Parameters
    ----------
    fps_cnt: int
        The count of frames.
    fps: float
        The frames per second.
    joint_list: list
        The list of landmarks
    left_positions: list
        The list of trajectory angles for left hand.
    left_confidence_bool: list
        The list of the left confidence scores for each trajectory.
    right_positions: list
        The list of trajectory angles for right hand.
    right_confidence_bool: list
        The list of the right confidence scores for each trajectory.
    """
    print(f"FPS Count: {fps_cnt} at FPS: {fps: .02f}Hz")
    print("------------------------------------------------------------------")
    print("##:\tLANDMARK NAME    \t[ x      y      z      visibility]")
    for i in range(len(joint_list)):
        # Check visibility
        if joint_list[i][3] < 0.5:
            print(Bcolors.FAIL, end='')
        elif joint_list[i][3] < 0.8:
            print(Bcolors.WARNING, end='')

        # The code `print(f"{string: <x}")` means right padding white space to length x.
        print(f"{i:02d}:\t{pose_name[i]: <17}\t{joint_list[i]}")

        # Check visibility
        if joint_list[i][3] < 0.8:
            print(Bcolors.ENDC, end='')
    print("Left trajectory angles:\t\t[", end='')
    for i in range(len(left_positions)):
        if not left_confidence_bool[i]:
            print(f"{Bcolors.WARNING}{left_positions[i]}{Bcolors.ENDC}", end='')
        else:
            print(f"{left_positions[i]}", end='')
        if i == len(left_positions) - 1:
            print(']')
        else:
            print('\t', end='')
    print("Left trajectory angles (deg):\t[", end='')
    for i in range(len(left_positions)):
        if not left_confidence_bool[i]:
            print(f"{Bcolors.WARNING}{round(math.degrees(left_positions[i]), 3)}{Bcolors.ENDC}", end='')
        else:
            print(f"{round(math.degrees(left_positions[i]), 3)}", end='')
        if i == len(left_positions) - 1:
            print(']')
        else:
            print('\t', end='')
    print("Right trajectory angles:\t[", end='')
    for i in range(len(right_positions)):
        if not right_confidence_bool[i]:
            print(f"{Bcolors.WARNING}{right_positions[i]}{Bcolors.ENDC}", end='')
        else:
            print(f"{right_positions[i]}", end='')
        if i == len(right_positions) - 1:
            print(']')
        else:
            print('\t', end='')
    print("Right trajectory angles (deg):\t[", end='')
    for i in range(len(right_positions)):
        if not right_confidence_bool[i]:
            print(f"{Bcolors.WARNING}{round(math.degrees(right_positions[i]), 3)}{Bcolors.ENDC}", end='')
        else:
            print(f"{round(math.degrees(right_positions[i]), 3)}", end='')
        if i == len(right_positions) - 1:
            print(']')
        else:
            print('\t', end='')


def dotproduct(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the dot product between two vectors v1 and v2.

    Parameters
    ----------
    v1: np.ndarray
        The first given vector.
    v2: np.ndarray
        The second given vector.

    Returns
    -------
    out: float
        The dot product of the two vectors.
    """
    return np.dot(v1, v2)


def length(v: np.ndarray) -> float:
    """
    Calculates the vector norm (length).

    Parameters
    ----------
    v: np.ndarray
        The given vector.

    Returns
    -------
    out: float
        The norm of the vector.
    """
    return math.sqrt(dotproduct(v, v))


def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the angle between vector 1 and vector 2. The unit is in radian.

    Parameters
    ----------
    v1: np.ndarray
        The vector 1.
    v2: np.ndarray
        The vector 2.

    Returns
    -------
    out: float
        The angle in radian [0, pi].
    """
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def landmark2list(landmark: NamedTuple) -> list:
    """
    Make pose_landmarks become 2D array.

    Parameters
    ----------
    landmark: NamedTuple
        The original format from mediapipe.solutions.pose.Pose.process
        Ref: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

        result.pose_landmarks [x, y, z, visibility]
        A list of pose landmarks. Each landmark consists of the following:

        x and y:
            Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
        z:
            Represents the landmark depth with the depth at the midpoint of hips being the origin, and the
            smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the
            same scale as x.
        visibility:
            A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not
            occluded) in the image.

    Returns
    -------
    joint_list: list
        The 2D array of type [
            np.array[x, y, z, visibility (v)],
            np.array[x2, y2, z2, v2],
            ...
        ]
    """
    joint_list = []
    # For each x, y, z, visibility.
    # NOTE: We use pose_world_landmarks here (2024.04.02)
    for data_point in landmark.pose_world_landmarks.landmark:
        # Make x, y, z, visibility to become 1D array.
        point_list = []
        point_list.append(round(float(data_point.x), 3))
        point_list.append(round(float(data_point.y), 3))
        point_list.append(round(float(data_point.z), 3))
        point_list.append(round(float(data_point.visibility), 3))
        # Append this 1D array to the 2D array.
        joint_list.append(np.array(point_list))
    return joint_list


def get_landmark_loc(landmark_list: list, idx: int) -> np.ndarray:
    """
    Get the location of the landmark by distinct index, ignoring visibility.

    Parameters
    ----------
    landmark_list: list
        The pose_landmarks [[x, y, z, visibility (v)], [x2, y2, z2, v2], ...]
        from landmark2list.
    idx: int
        The index of the landmark.
        Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

        [0, 0, x, x]         6 5 4     1 2 3        [1, 0, x, x]
        RIGHT             8         0         7             LEFT
                                 10---9

          20    22                                    21   19
         /   \ /        12--------------------11        \ /   \
        18---16        /  \                  /  \        15---17
               \----14/    \                /    \13----/
                            \              /
                             \            /
                              24--------23
                             /           \
                            /             \
                           /               \
                          /                 \
                        26                   25
                          \                 /
                           \               /
                            \             /
                             \           /
                             28         27
                            /  \       /  \
        [1, 0, x, x]       30---32    29---31       [1, 1, x, x]
        0 - nose
        1 - left eye (inner)
        2 - left eye
        3 - left eye (outer)
        4 - right eye (inner)
        5 - right eye
        6 - right eye (outer)
        7 - left ear
        8 - right ear
        9 - mouth (left)
        10 - mouth (right)
        11 - left shoulder
        12 - right shoulder
        13 - left elbow
        14 - right elbow
        15 - left wrist
        16 - right wrist
        17 - left pinky
        18 - right pinky
        19 - left index
        20 - right index
        21 - left thumb
        22 - right thumb
        23 - left hip
        24 - right hip
        25 - left knee
        26 - right knee
        27 - left ankle
        28 - right ankle
        29 - left heel
        30 - right heel
        31 - left foot index
        32 - right foot index

    Returns
    -------
    out: np.ndarray
        The 1D array in the format [x, y, z]
    """
    return np.array([landmark_list[idx][0], landmark_list[idx][1], landmark_list[idx][2]])


def vector_cal_idx(landmark_list: list, tail: int, head: int) -> np.ndarray:
    """
    Calculate the vector starts from tail to head given index.
    tail  ----->  head

    Parameters
    ----------
    landmark_list: list
        The pose_landmarks [[x, y, z, visibility (v)], [x2, y2, z2, v2], ...]
        from landmark2list.
    tail: int
        The index of the tail.
    head: int
        The index of the head.

    Returns
    -------
    out: np.ndarray
        The 3D np array [x, y, z] shows the vector starts from tail to head.
    """
    return get_landmark_loc(landmark_list, head) - get_landmark_loc(landmark_list, tail)


class PoseDetection:
    """
    This is the class to handle pose detection using mediapipe.

    Attributes
    ----------
    camera_id: int
        The ID of the camera. e.g., The ID of /dev/video0 is 0.
    video_name: str
        The filename of the saved video.
    txt_name: str
        The filename of the output locations of landmarks.
    fps_cnt: int
        The total frames read from camera.
    prev_time: float
        The time format defined by time.time().
        This is used to store the previous time stamp.
    cap: cv2.VideoCapture
        The cv2 video capturing object.
    pose:
        The mediapipe pose object.
    width: int
        The width of the frame.
    height: int
        The height of the frame.
    fourcc: cv2.VideoWriter_fourcc
    out: cv2.VideoWriter
        The cv2 video writer object.

    Methods
    -------
    read_image_and_process(self) -> tuple[bool, None | np.ndarray, ...]:
        Read a single frame from camera, process the image frame, and then return the resulting landmarks.
    draw_landmarks(self, img, landmarks):
        Use the image and landmarks from the previous step ``read_image_and_process``
        to draw landmarks on the given image.
    calculation(self, joint_list: list):
        Calculate the angle of the trajectory from the given joint list.
    calculation_old(self, joint_list: list):
        Calculate the angle of the trajectory from the given joint list.
        This method is provided by the senior member.
    main(self) -> tuple[bool, None | list]:
        This function provides the procedures to calculate the trajectory angles from the "single frame".
    run(self):
        This function provides an infinity loop to run the main function and thus calculates angles continuously.
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.video_name = "result.mp4"
        self.txt_name = 'landmarks.txt'
        self.fps_cnt = int(0)
        self.prev_time = time.time()

        # mediapipe
        # self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap = cv2.VideoCapture(self.camera_id)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.CAP_PROP_FPS, 60)

        # absl is the log method for mediapipe, we initialize it here.
        absl.logging.set_verbosity(absl.logging.ERROR)
        absl.logging.get_absl_handler().python_handler.stream = sys.stdout

        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                           static_image_mode=False, model_complexity=1)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # save realtime video
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(self.video_name, self.fourcc, 15.0, (self.width, self.height))

    def read_image_and_process(self) -> tuple[bool, None | np.ndarray, ...]:
        """
        Read a single frame from camera, process the image frame, and then return the resulting landmarks.
        If reading the frame successfully, the fps counter is incremented by 1.

        Returns
        -------
        out: tuple[bool, None | np.ndarray, None | NamedTuple]
            1. bool: True if the image is read successfully, False if not.
            2. np.ndarray: The corresponding image.
            3. NamedTuple: The result of the pose detection.
        """
        ret, img = self.cap.read()
        if not ret:
            return False, None, None

        self.fps_cnt += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return True, img, self.pose.process(imgRGB)

    def draw_landmarks(self, img, landmarks):
        """
        Use the image and landmarks from the previous step ``read_image_and_process``
        to draw landmarks on the given image.

        Parameters
        ----------
        img: np.ndarray
            The given image.
        landmarks: NamedTuple
            The result landmarks provided by mediapipe.
        """
        # Draw the landmarks into the video.
        mp.solutions.drawing_utils.draw_landmarks(img, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        self.out.write(img)
        cv2.imshow('Robot Arm', img)

        key = cv2.waitKey(1)
        if key == 27:  # esc
            cv2.destroyAllWindows()
            exit(0)

    def calculation(self, joint_list: list) -> tuple[list, list, list, list]:
        """
        Calculate the angle of the trajectory from the given joint list.

        Parameters
        ----------
        joint_list: list
            The given joint list.

        Returns
        -------
        out: tuple[list, list, list, list]
            The trajectory angles in radian and the confidence for each trajectory.
            [left_angle, left_confidence, right_angle, right_confidence]
        confidence_bool: list[bool, bool, bool, bool, bool]
            This Python list is used to record whether the visibility of all corresponding trajectory points'
            mediapipe landmarks needing computation is greater than 0.8.
            If the condition is met, then it is marked as True.
        """
        confidence_bool = [False, False, False, False, False]
        right_confidence_bool = [False, False, False, False, False]

        # The meaning of the index is written in the doc string of the function
        # ``get_landmark_loc``
        # 左右肩
        shoulder = vector_cal_idx(joint_list, 11, 12)
        # 肩膀 -> 手肘 (上臂)
        larm = vector_cal_idx(joint_list, 11, 13)
        # 手肘 -> 手腕 (前臂)
        lforearm = vector_cal_idx(joint_list, 13, 15)
        # 食指
        lindex = vector_cal_idx(joint_list, 15, 19)
        # 小指
        lpinky = vector_cal_idx(joint_list, 15, 17)
        # 手肘->食指
        lelbow_index = vector_cal_idx(joint_list, 13, 19)
        # 手肘->拇指
        lelbow_thumb = vector_cal_idx(joint_list, 13, 21)
        # 肩膀 -> 手肘 (上臂)
        rarm = vector_cal_idx(joint_list, 12, 14)
        # 手肘 -> 手腕 (前臂)
        rforearm = vector_cal_idx(joint_list, 14, 16)
        # 食指
        rindex = vector_cal_idx(joint_list, 16, 20)
        # 小指
        rpinky = vector_cal_idx(joint_list, 16, 18)
        # 手肘->食指
        relbow_index = vector_cal_idx(joint_list, 14, 20)
        # 手肘->拇指
        relbow_thumb = vector_cal_idx(joint_list, 14, 22)
        # 左肩到左臀
        lshoulder2hip = vector_cal_idx(joint_list, 11, 23)

        # 身體的法向量，方向向前
        lbody_norm = np.cross(shoulder, lshoulder2hip)
        # print(f"lbody_norm: {lbody_norm}")

        # 身體中軸，方向向下
        shoulder_center = (get_landmark_loc(joint_list, 11) + get_landmark_loc(joint_list, 12)) / 2
        hip_center = (get_landmark_loc(joint_list, 23) + get_landmark_loc(joint_list, 24)) / 2
        body_central = hip_center - shoulder_center
        # print(f"body_central: {body_central}")

        # 身體中軸的法向量，方向向左
        body_norm_norm = np.cross(lbody_norm, body_central)
        # print(f"body_norm_norm: {body_norm_norm}")

        # J1 = arm - proj_{body_norm_norm}(arm)
        j1_vec = larm - (np.dot(larm, body_norm_norm) / (length(body_norm_norm) ** 2) * body_norm_norm)
        j1r_vec = rarm - (np.dot(rarm, -body_norm_norm) / (length(body_norm_norm) ** 2) * -body_norm_norm)
        # Setting zero degree to point downward.
        j1 = angle(j1_vec, np.array([0, 1, 0]))
        j1r = angle(j1r_vec, np.array([0, 1, 0]))
        # print(f"j1_vec: {j1_vec}")
        # print(f"j1: {j1}")

        j2 = angle(j1_vec, larm)
        j2r = angle(j1r_vec, rarm)
        # print(f"j2: {j2}")
        confidence_bool[0] = all(val > 0.8 for val in [joint_list[row][3] for row in [11, 12, 13, 23, 24]])
        confidence_bool[1] = confidence_bool[0]
        right_confidence_bool[0] = all(val > 0.8 for val in [joint_list[row][3] for row in [11, 12, 14, 23, 24]])
        right_confidence_bool[1] = right_confidence_bool[0]

        # TODO: Complete J3
        arm_norm = np.cross(-larm, lforearm)
        hand_norm = np.cross(lpinky, lindex)
        rarm_norm = np.cross(-rarm, rforearm)
        rhand_norm = np.cross(rpinky, rindex)
        j3 = angle(arm_norm, hand_norm)
        j3r = angle(rarm_norm, rhand_norm)
        # print(f"j3: {j3}")
        confidence_bool[2] = all(val > 0.8 for val in [joint_list[row][3] for row in [11, 13, 15, 17, 19]])
        right_confidence_bool[2] = all(val > 0.8 for val in [joint_list[row][3] for row in [12, 14, 16, 18, 20]])

        j4 = angle(-larm, lforearm)
        j4r = angle(-rarm, rforearm)
        # print(f"-larm: {-larm}")
        # print(f"lforarm: {lforearm}")
        # print(f"j4: {j4}")
        confidence_bool[3] = all(val > 0.8 for val in [joint_list[row][3] for row in [11, 13, 15]])
        right_confidence_bool[3] = all(val > 0.8 for val in [joint_list[row][3] for row in [12, 14, 16]])

        # NOTE: J5 is not correct because mediapipe can't detect hand well.
        j5 = angle(lpinky, lindex)
        j5r = angle(rpinky, rindex)
        # print(f"j5: {j5}")
        confidence_bool[4] = all(val > 0.8 for val in [joint_list[row][3] for row in [15, 17, 19]])
        right_confidence_bool[4] = all(val > 0.8 for val in [joint_list[row][3] for row in [16, 18, 20]])

        result = [round(j1, 3), round(j2, 3), round(j3, 3), round(j4, 3), round(j5, 3)]
        resultr = [round(j1r, 3), round(j2r, 3), round(j3r, 3), round(j4r, 3), round(j5r, 3)]

        # # Calibration for Unity
        # result = [a + b for (a, b) in
        #           zip(result, [0, 0, math.radians(-90), 0, 0])]
        # result = [round(i, 3) for i in result]

        # Calibration for real robot arm
        # Use y = mx + k, and then trial and error.
        result[1] = math.radians(-15 / 16 * math.degrees(result[1]) + 120)
        result[2] = math.radians(-23 / 12 * math.degrees(result[2]) + 940 / 3)
        result = [round(i, 3) for i in result]
        resultr[1] = math.radians(-15 / 16 * math.degrees(resultr[1]) + 120)
        resultr[2] = math.radians(-23 / 12 * math.degrees(resultr[2]) + 940 / 3)
        resultr = [round(i, 3) for i in resultr]

        return result, confidence_bool, resultr, right_confidence_bool

    def calculate_old(self, joint_list: list):
        """
        Calculate the angle of the trajectory from the given joint list.
        This method is provided by the senior member, and is shown in ``sample_code.py``.

        Parameters
        ----------
        joint_list: list
            The given joint list.

        Returns
        -------
        out: list
            The trajectory angles in radian.
        """
        angle1, a1_last, a1_f = 0, 0, 0
        angle2, a2_last, a2_f = 0, 0, 0
        angle3, a3_last, a3_f = 0, 0, 0
        angle4, a4_last, a4_f = 0, 0, 0
        angle5, a5_last, a5_f = 0, 0, 0
        angle6, a6_last, a6_f = 0, 0, 0

        send_1 = 0
        send_2 = 0
        send_3 = 0
        send_4 = 0
        send_5 = 0

        # 11: lshoulder / 12: rshoulder / 13: elbow / 15: wrist / 21: thumb / 17: pinky / 19: index / 23: hip
        # 肩膀->手肘
        arm = (
            joint_list[13][0] - joint_list[12][0],
            joint_list[13][1] - joint_list[12][1],
            joint_list[13][2] - joint_list[12][2]
        )
        # 手肘->手腕
        forearm = (
            joint_list[15][0] - joint_list[13][0],
            joint_list[15][1] - joint_list[13][1],
            joint_list[15][2] - joint_list[13][2]
        )
        # 左右肩
        shoulder = (
            joint_list[11][0] - joint_list[12][0],
            joint_list[11][1] - joint_list[12][1],
            joint_list[11][2] - joint_list[12][2]
        )
        # 食指
        index = (
            joint_list[19][0] - joint_list[15][0],
            joint_list[19][1] - joint_list[15][1],
            joint_list[19][2] - joint_list[15][2]
        )
        # 小指
        pinky = (
            joint_list[17][0] - joint_list[15][0],
            joint_list[17][1] - joint_list[15][1],
            joint_list[17][2] - joint_list[15][2]
        )
        # 手肘->食指
        elbow_index = (
            joint_list[19][0] - joint_list[13][0],
            joint_list[19][1] - joint_list[13][1],
            joint_list[19][2] - joint_list[13][2]
        )
        # 手肘->拇指
        elbow_thumb = (
            joint_list[21][0] - joint_list[13][0],
            joint_list[21][1] - joint_list[13][1],
            joint_list[21][2] - joint_list[13][2]
        )
        # 肩膀->骨盆
        hip_shou = (
            joint_list[12][0] - joint_list[23][0],
            joint_list[12][1] - joint_list[23][1],
            joint_list[12][2] - joint_list[23][2]
        )

        #### calculate angle
        if joint_list[12][3] > 0.8 and joint_list[13][3] > 0.8 and joint_list[15][3] > 0.8:
            index_pinky = np.cross(index, pinky)
            arm_fore = np.cross((-arm[0], -arm[1], -arm[2]), forearm)

            # J1角度
            J1 = round(math.degrees(angle((arm[0], 0, arm[2]), (1, 0, 0))), 3)
            # J1方向
            dir_a1 = dotproduct((0, 0, arm[2]), (0, 0, 1))
            if dir_a1 != 0:
                dir_a1 /= abs(dir_a1)
                angle1 = J1 * dir_a1
            else:
                angle1 = J1 * dir_a1
            # J1: -165~+165
            if angle1 > 163:
                angle1 = 163
            elif angle1 < -163:
                angle1 = -163
            else:
                pass
            if abs(abs(angle1) - abs(a1_last)) <= 1:
                pass
            else:
                a1_f = angle1
                send_1 += 1
            a1_last = angle1

            # J2角度: -125 ~ +85->70
            J2 = round(math.degrees(angle((shoulder[0], shoulder[1], 0), (arm[0], arm[1], 0))), 1)
            # J2方向
            if joint_list[13][1] > joint_list[12][1]:
                angle2 = -J2
            elif joint_list[13][1] <= joint_list[12][1]:
                angle2 = J2

            if angle2 > 70:
                angle2 = 70
            elif angle2 < -60:
                angle2 = -60
            else:
                pass
            if abs(abs(angle2) - abs(a2_last)) <= 1:
                send_2 = 0
            else:
                a2_f = angle2
                send_2 += 1
            a2_last = angle2

            # J3角度: -55 ~ +185
            J3 = round(math.degrees(angle((arm[0], arm[1], 0), (forearm[0], forearm[1], 0))), 1)
            # J3方向
            if joint_list[15][1] < joint_list[13][1]:
                angle3 = J3 + 90
            elif joint_list[15][1] >= joint_list[13][1]:
                angle3 = -J3 + 90

            if angle3 > 180:
                angle3 = 180
            elif angle3 < -45:
                angle3 = -45
            else:
                pass
            if abs(abs(angle3) - abs(a3_last)) <= 1:
                send_3 = 0
            else:
                a3_f = angle3
                send_3 += 1
            a3_last = angle3

            # J4角度: -190 ~ +190
            J4 = round(math.degrees(angle(index_pinky, arm_fore)), 1)
            # J4方向
            if J4 == 0:
                angle4 = -J4
            elif J4 == 180:
                angle4 = J4
            else:
                cross_hand_elb = np.cross(index_pinky, arm_fore)
                dir_a4 = dotproduct(cross_hand_elb, forearm)
                dir_a4 /= abs(dir_a4)
                angle4 = J4 * dir_a4

            if angle4 > 90:
                angle4 = 90
            elif angle4 < -90:
                angle4 = -90
            else:
                pass
            if abs(abs(angle4) - abs(a4_last)) <= 1:
                pass
            else:
                a4_f = angle4
                send_4 += 1
            a4_last = angle4

            # J5角度: -115 ~ +115
            J5 = round(math.degrees(angle(forearm, index)), 1)
            # J5方向
            if joint_list[19][1] >= joint_list[15][1]:
                angle5 = -J5
            elif joint_list[19][1] < joint_list[15][1]:
                angle5 = J5

            if angle5 > 110:
                angle5 = 110
            elif angle5 < -110:
                angle5 = -110
            else:
                pass
            if abs(abs(angle5) - abs(a5_last)) <= 1:
                pass
            else:
                a5_f = angle5
                send_5 += 1
            a5_last = angle5

            result = [math.radians(deg) for deg in [a1_f, a2_f, a3_f, a4_f, a5_f]]
            # Calibration for Unity
            result = [a + b for (a, b) in
                      zip(result, [math.radians(90), math.radians(90), 0, math.radians(90), math.radians(90)])]
            print(f"Result: {result}")

            return result

    def main(self) -> tuple[bool, None | list, None | list, None | list, None | list]:
        """
        This function provides the procedures to calculate the trajectory angles from the "single frame".

        Returns
        -------
        out: tuple[bool, None | list, None | list, None | list, None | list]
            1. True if the function should keep track of the pose, False otherwise.
            2. The list of the left positions.
            3. The list of the left confidence.
            4. The list of the right positions.
            5. The list of the right confidence.
        """
        success, img, result = self.read_image_and_process()
        if not success:
            return False, None
        self.draw_landmarks(img, result.pose_landmarks)
        clear_screen()

        # If mediapipe detects landmarks successfully.
        if result.pose_landmarks:
            # Make pose_landmarks become 2D array.
            joint_list = landmark2list(result)
            positions, confidence_bool, right_positions, right_confidence_bool = self.calculation(joint_list)

            # Calculate time
            cur_time = time.time()
            fps = 1 / (cur_time - self.prev_time)
            print_landmark(self.fps_cnt, fps, joint_list, positions, confidence_bool, right_positions,
                           right_confidence_bool)
            self.prev_time = cur_time

            return True, positions, confidence_bool, right_positions, right_confidence_bool
        else:
            # Calculate time
            cur_time = time.time()
            fps = 1 / (cur_time - self.prev_time)
            print(f"FPS Count: {self.fps_cnt} at FPS: {fps: .02f}Hz")
            print(f"{Bcolors.FAIL}Error, pose detection failed.{Bcolors.ENDC}", file=sys.stderr)
            self.prev_time = cur_time

            return False, None, None, None, None

    def run(self):
        """
        This function provides an infinity loop to run the main function and thus calculates angles continuously.
        """
        while True:
            cont = self.main()
            if not cont:
                break
        self.cap.release()
        self.out.release()


if __name__ == '__main__':
    pose_detection = PoseDetection(0)
    pose_detection.run()
