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


def main():
    # absl.logging.set_verbosity(absl.logging.ERROR)
    # absl.logging.get_absl_handler().python_handler.stream = sys.stdout

    video_name = "result.mp4"
    txt_name = 'landmarks.txt'
    fps_cnt = 0
    prev_time = time.time()

    ## mediapipe
    # cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 60)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False,
                       model_complexity=1)

    mpDraw = mp.solutions.drawing_utils

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 影像寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 影像高度

    ## save realtime video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_name, fourcc, 15.0, (width, height))
    while True:
        ret, img = cap.read()
        if not ret:
            break

        fps_cnt += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        clear_screen()

        # If mediapipe detects landmarks successfully.
        if result.pose_landmarks:
            # Draw the landmarks into the video.
            mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
            # Make pose_landmarks become 2D array.
            joint_list = landmark2list(result)

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
            print(lbody_norm)
            # 身體中軸
            shoulder_center = (get_landmark_loc(joint_list, 11) + get_landmark_loc(joint_list, 12)) / 2
            hip_center = (get_landmark_loc(joint_list, 23) + get_landmark_loc(joint_list, 24)) / 2
            body_central = hip_center - shoulder_center
            print(body_central)
            # 身體中軸的法向量
            body_norm_norm = np.cross(lbody_norm, body_central)
            print(body_norm_norm)

            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            print_landmark(fps_cnt, fps, joint_list)
            prev_time = cur_time
        else:
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            print(f"FPS Count: {fps_cnt} at FPS: {fps: .02f}Hz")
            print("Error, pose detection failed.", file=sys.stderr)
            prev_time = cur_time

        out.write(img)
        cv2.imshow('Robot Arm', img)

        key = cv2.waitKey(1)
        if key == 27:  # esc
            cv2.destroyAllWindows()
            break

    cap.release()
    out.release()


if __name__ == '__main__':
    main()
