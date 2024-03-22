import cv2
import math
import mediapipe as mp
import numpy as np
import os
import sys
import time


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
    clear_screen()
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


def main():
    # absl.logging.set_verbosity(absl.logging.ERROR)
    # absl.logging.get_absl_handler().python_handler.stream = sys.stdout

    video_name = "result.mp4"
    txt_name = 'landmarks.txt'
    fps_cnt = 0
    prev_time = time.time()

    ## mediapipe
    # cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(2)
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
        """
        Ref: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        result.pose_landmarks [x, y, z, v]
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
        """

        # If mediapipe detects landmarks successfully.
        if result.pose_landmarks:
            # Draw the landmarks into the video.
            mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
            # Make pose_landmarks become 2D array.
            joint_list = []
            # For each x, y, z, visibility.
            for data_point in result.pose_landmarks.landmark:
                # Make x, y, z, visibility to become 1D array.
                point_list = []
                point_list.append(round(float(data_point.x), 3))
                point_list.append(round(float(data_point.y), 3))
                point_list.append(round(float(data_point.z), 3))
                point_list.append(round(float(data_point.visibility), 3))
                # Append this 1D array to the 2D array.
                joint_list.append(point_list)

            # 左右肩
            shoulder = (
                joint_list[11][0] - joint_list[12][0],
                joint_list[11][1] - joint_list[12][1],
                joint_list[11][2] - joint_list[12][2]
            )
            # 肩膀 -> 手肘 (上臂)
            larm = (
                joint_list[13][0] - joint_list[11][0],
                joint_list[13][1] - joint_list[11][1],
                joint_list[13][2] - joint_list[11][2]
            )
            # 手肘 -> 手腕 (前臂)
            lforearm = (
                joint_list[15][0] - joint_list[13][0],
                joint_list[15][1] - joint_list[13][1],
                joint_list[15][2] - joint_list[13][2]
            )
            # 食指
            lindex = (
                joint_list[19][0] - joint_list[15][0],
                joint_list[19][1] - joint_list[15][1],
                joint_list[19][2] - joint_list[15][2]
            )
            # 小指
            lpinky = (
                joint_list[17][0] - joint_list[15][0],
                joint_list[17][1] - joint_list[15][1],
                joint_list[17][2] - joint_list[15][2]
            )
            # 手肘->食指
            lelbow_index = (
                joint_list[19][0] - joint_list[13][0],
                joint_list[19][1] - joint_list[13][1],
                joint_list[19][2] - joint_list[13][2]
            )
            # 手肘->拇指
            lelbow_thumb = (
                joint_list[21][0] - joint_list[13][0],
                joint_list[21][1] - joint_list[13][1],
                joint_list[21][2] - joint_list[13][2]
            )
            # 肩膀 -> 手肘 (上臂)
            rarm = (
                joint_list[14][0] - joint_list[12][0],
                joint_list[14][1] - joint_list[12][1],
                joint_list[14][2] - joint_list[12][2]
            )
            # 手肘 -> 手腕 (前臂)
            rforearm = (
                joint_list[16][0] - joint_list[14][0],
                joint_list[16][1] - joint_list[14][1],
                joint_list[16][2] - joint_list[14][2]
            )
            # 食指
            rindex = (
                joint_list[20][0] - joint_list[16][0],
                joint_list[20][1] - joint_list[16][1],
                joint_list[20][2] - joint_list[16][2]
            )
            # 小指
            rpinky = (
                joint_list[18][0] - joint_list[16][0],
                joint_list[18][1] - joint_list[16][1],
                joint_list[18][2] - joint_list[16][2]
            )
            # 手肘->食指
            relbow_index = (
                joint_list[20][0] - joint_list[14][0],
                joint_list[20][1] - joint_list[14][1],
                joint_list[20][2] - joint_list[14][2]
            )
            # 手肘->拇指
            relbow_thumb = (
                joint_list[22][0] - joint_list[14][0],
                joint_list[22][1] - joint_list[14][1],
                joint_list[22][2] - joint_list[14][2]
            )

            #### calculate angle
            if joint_list[12][3] > 0.8 and joint_list[13][3] > 0.8 and joint_list[15][3] > 0.8:
                pass
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            print_landmark(fps_cnt, fps, joint_list)
            prev_time = cur_time
        else:
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            print_landmark(fps_cnt, fps, joint_list)
            clear_screen()
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
