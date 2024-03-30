import cv2
import math
import mediapipe as mp
import numpy as np
import os
import sys
import time
from typing import NamedTuple


def clear_screen():
    """
    Clear the entire terminal. This function supports multiple platforms including Windows and Linux.
    """
    if sys.platform == "win32":
        os.system("cls")
    elif sys.platform == "linux" or sys.platform == "darwin":
        os.system("clear")


def print_landmark(fps_cnt: int, fps: float, joint_list: list):
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
    """
    print(f"FPS Count: {fps_cnt} at FPS: {fps: .02f}Hz")
    for i in range(len(joint_list)):
        print(f"{i:02d}: {joint_list[i]}")


### angle
def dotproduct(v1, v2):
    return np.dot(v1, v2)


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
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
    for data_point in landmark.pose_landmarks.landmark:
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
    main(self) -> tuple[bool, None | list]:
        This function provides the procedures to calculate the trajectory angles from the "single frame".
    run(self):
        This function provides an infinity loop to run the main function and thus calculates angles continuously.
    """
    def __init__(self):
        self.video_name = "result.mp4"
        self.txt_name = 'landmarks.txt'
        self.fps_cnt = int(0)
        self.prev_time = time.time()

        # mediapipe
        # self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.CAP_PROP_FPS, 60)

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

    def calculation(self, joint_list: list):
        """
        Calculate the angle of the trajectory from the given joint list.

        Parameters
        ----------
        joint_list: list
            The given joint list.

        Returns
        -------
        out: list
            The trajectory angles.
        """
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

        ##### J1 #####
        # 身體的法向量
        lbody_norm = np.cross(shoulder, lshoulder2hip)
        # print(lbody_norm)

        # 身體中軸
        shoulder_center = (get_landmark_loc(joint_list, 11) + get_landmark_loc(joint_list, 12)) / 2
        hip_center = (get_landmark_loc(joint_list, 23) + get_landmark_loc(joint_list, 24)) / 2
        body_central = hip_center - shoulder_center
        # print(body_central)

        # 身體中軸的法向量
        body_norm_norm = np.cross(lbody_norm, body_central)
        # print(body_norm_norm)

        return [1.0, 1.0, 1.0, 1.0, 1.0]

    def main(self) -> tuple[bool, None | list]:
        """
        This function provides the procedures to calculate the trajectory angles from the "single frame".

        Returns
        -------
        out: tuple[bool, None | list]
            1. True if the function should keep track of the pose, False otherwise.
            2. The list of the positions.
        """
        success, img, result = self.read_image_and_process()
        if not success:
            return False, None
        clear_screen()

        # If mediapipe detects landmarks successfully.
        if result.pose_landmarks:
            # # Draw the landmarks into the video.
            # mp.solutions.drawing_utils.draw_landmarks(img, result.pose_landmarks,
            #                                           mp.solutions.pose.POSE_CONNECTIONS)
            self.draw_landmarks(img, result.pose_landmarks)
            # Make pose_landmarks become 2D array.
            joint_list = landmark2list(result)
            positions = self.calculation(joint_list)

            # Calculate time
            cur_time = time.time()
            fps = 1 / (cur_time - self.prev_time)
            print_landmark(self.fps_cnt, fps, joint_list)
            self.prev_time = cur_time

            return True, positions
        else:
            # Calculate time
            cur_time = time.time()
            fps = 1 / (cur_time - self.prev_time)
            print(f"FPS Count: {self.fps_cnt} at FPS: {fps: .02f}Hz")
            print("Error, pose detection failed.", file=sys.stderr)
            self.prev_time = cur_time

            return False, None

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
    pose_detection = PoseDetection()
    pose_detection.run()
