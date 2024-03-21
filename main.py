import cv2
import mediapipe as mp
import os
import sys
import numpy as np
import math
import absl.logging


def clear_screen():
    print(sys.platform)
    if sys.platform == "win32":
        os.system("cls")
    elif sys.platform == "linux" or sys.platform == "darwin":
        os.system("clear")


def print_landmark(joint_list: list):
    clear_screen()
    for i in range(len(joint_list)):
        print(f"{i:02d}: {joint_list[i]}")


### angle
def dotproduct(v1, v2):
    return np.dot(v1, v2)


def cross(v1, v2):
    return np.cross(v1, v2)


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def main():
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.get_absl_handler().python_handler.stream = sys.stdout

    video_name = 'video.avi'
    txt_name = 'video.txt'

    angle1, a1_last, a1_f = 0, 0, 0
    angle2, a2_last, a2_f = 0, 0, 0
    angle3, a3_last, a3_f = 0, 0, 0
    angle4, a4_last, a4_f = 0, 0, 0
    angle5, a5_last, a5_f = 0, 0, 0
    angle6, a6_last, a6_f = 0, 0, 0

    ## mediapipe
    # cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(4)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False,
                       model_complexity=1)

    mpDraw = mp.solutions.drawing_utils
    poseLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 0), thickness=3)
    poseConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=5)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 影像寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 影像高度

    ## save realtime video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_name, fourcc, 15.0, (width, height))
    while True:
        ret, img = cap.read()
        if not ret:
            break

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

        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
            joint = ''
            joint_list = []
            for data_point in result.pose_landmarks.landmark:
                point_list = []
                point_list.append(round(float(data_point.x), 3))
                point_list.append(round(float(data_point.y), 3))
                point_list.append(round(float(data_point.z), 3))
                point_list.append(round(float(data_point.visibility), 3))
                joint_list.append(point_list)
            send_1 = 0
            send_2 = 0
            send_3 = 0
            send_4 = 0
            send_5 = 0
            # print(f"\r{joint_list[11]}, {joint_list[13]}", end=" ")
            print_landmark(joint_list)

            # 11: lshoulder / 12: rshoulder / 13: elbow / 15: wrist / 21: thumb / 17: pinky / 19: index / 23: hip
            # 肩膀->手肘
            arm = (joint_list[13][0] - joint_list[11][0], joint_list[13][1] - joint_list[11][1],
                   joint_list[13][2] - joint_list[11][2])
            # 手肘->手腕
            forearm = (joint_list[15][0] - joint_list[13][0], joint_list[15][1] - joint_list[13][1],
                       joint_list[15][2] - joint_list[13][2])
            # 左右肩
            shoulder = (joint_list[11][0] - joint_list[12][0], joint_list[11][1] - joint_list[12][1],
                        joint_list[11][2] - joint_list[12][2])
            # 食指
            index = (joint_list[19][0] - joint_list[15][0], joint_list[19][1] - joint_list[15][1],
                     joint_list[19][2] - joint_list[15][2])
            # 小指
            pinky = (joint_list[17][0] - joint_list[15][0], joint_list[17][1] - joint_list[15][1],
                     joint_list[17][2] - joint_list[15][2])
            # 手肘->食指
            elbow_index = (joint_list[19][0] - joint_list[13][0], joint_list[19][1] - joint_list[13][1],
                           joint_list[19][2] - joint_list[13][2])
            # 手肘->拇指
            elbow_thumb = (joint_list[21][0] - joint_list[13][0], joint_list[21][1] - joint_list[13][1],
                           joint_list[21][2] - joint_list[13][2])
            # 肩膀->骨盆
            hip_shou = (joint_list[12][0] - joint_list[23][0], joint_list[12][1] - joint_list[23][1],
                        joint_list[12][2] - joint_list[23][2])

            #### calculate angle
            if joint_list[12][3] > 0.8 and joint_list[13][3] > 0.8 and joint_list[15][3] > 0.8:
                index_pinky = cross(index, pinky)
                arm_fore = cross((-arm[0], -arm[1], -arm[2]), forearm)

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
                    cross_hand_elb = cross(index_pinky, arm_fore)
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

                # J6
                angle6 = 0
                a6_f = angle6
                a6_last = angle6

            send_ = str(a1_f) + ';' + str(a2_f) + ';' + str(a3_f) + ';' + str(a4_f) + ';' + str(a5_f) + ';' + str(a6_f)
            if send_2 and send_3 and send_5 > 0:
                # print('-----send:', send_)
                send_switch = 0
                # message, address = receive_socket.recvfrom(1024)
                # msg = message.decode().split(";")
                # jRot = [msg[0], msg[1], msg[2], msg[3], msg[4], msg[5]]
                # print("receive: ", jRot)
                with open(txt_name, 'a') as file:
                    file.write(send_)
                    file.write('\n')

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
