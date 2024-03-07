#!/usr/bin/env python

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import rclpy
import curses
import threading
from rclpy.node import Node

from std_msgs.msg import String


class VideoSubscriber(Node):
    def __init__(self, stdscr):
        super().__init__('video_subscriber')
        self.subscription = self.create_subscription(
            String,
            "/out/compressed",
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.stdscr = stdscr
        curses.noecho()
        curses.raw()
        self.stdscr.keypad(False)

        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(e)

    def display_video(self):
        # cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        # while not rclpy.is_shutdown():
        #     if self.cv_image is not None:
        #         cv2.imshow('Video', self.cv_image)
        #     cv2.waitKey(1)
        while True:
            cv2.imshow('Robot Arm', self.cv_image)
            key = cv2.waitKey(1)
            if key == 27:  # esc
                cv2.destroyAllWindows()
                break


# ... Rest of your code, e.g. initializing rclpy and running the node
def main(args=None):
    rclpy.init(args=args)
    stdscr = curses.initscr()
    node = VideoSubscriber(stdscr)

    # Spin the node in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.display_video()
    finally:
        curses.endwin()
        node.get_logger().info(f'Quit keyboard!')
        rclpy.shutdown()
        spin_thread.join()  # Ensure the spin thread is cleanly stopped


if __name__ == '__main__':
    main()
