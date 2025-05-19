#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from coco_interfaces.msg import PoseResult
import os
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
from ultralytics import YOLO

DICT_GESTURES = {
    1: "Brazo derecho arriba",
    2: "Brazo izquierdo arriba",
    3: "Ambos brazos arriba",
    4: "Ambos brazos abajo",
    5: "Brazo derecho hacia delante",
    6: "Brazo izquierdo hacia delante",
    7: "Ambos brazos hacia delante",
    8: "Símbolo X con los brazos",
    9: "Tocando nariz con muneca derecha",
    10: "Tocando nariz con muneca izquierda",
    11: "Tocando ojo izquierdo con muneca izquierda",
    12: "Tocando ojo izquierdo con muneca derecha", 
    13: "Tocando ojo derecho con muneca derecha", 
    14: "Tocando ojo derecho con muneca izquierda", 
    15: "Tocando oreja derecha con mano derecha", 
    16: "Tocando oreja derecha con mano izquierda", 
    17: "Tocando oreja derecha con ambas manos", 
    18: "Tocando oreja izquierda con mano izquierda", 
    19: "Tocando oreja izquierda con mano derecha", 
    20: "Tocando oreja izquierda con ambas manos", 
    21: "Tocando hombro izquierdo con mano izquierda", 
    22: "Tocando hombro izquierdo con mano derecha",
    23: "Tocando hombro derecho con mano derecha",
    24: "Tocando hombro derecho con mano izquierda",
    25: "Tocando codo izquierdo con mano derecha",
    26: "Tocando codo derecho con mano izquierda",
}


class CocoPoseDetector(Node):
    def __init__(self):
        super().__init__('coco_pose_detector')
        
        self.pose_result_publisher = self.create_publisher(
            PoseResult, '/pose_result', 10)
        
        self.bridge = CvBridge()
        self.frame = None

        self.create_subscription(
            Int16, '/current_challenge', self.handle_new_challenge, 10)
        self.create_subscription(
            Image, '/image_raw', self.handle_camera_image, 10)
        
        pkg_share_dir = get_package_share_directory('coco_dice')
        model_path = os.path.join(pkg_share_dir, 'models', 'yolov8s-pose.pt')
        self.model = YOLO(model_path)
                
        self.current_challenge = None
        self.game_active = False
        
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        
        self.create_timer(1/30, self.detect_poses)
        
        self.get_logger().info('Coco Pose Detector started successfully')
    
    def handle_new_challenge(self, msg):
        self.current_challenge = msg.data
        self.get_logger().info(f'New challenge received: {self.current_challenge}')
    
    def handle_camera_image(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 

        cv2.imshow("Camera Feed", self.frame)
        cv2.waitKey(1)
    
    def is_above(self, point1, point2, threshold=30):
        return point2[1] - point1[1] > threshold

    def is_below(self, point1, point2, threshold=15):
        return 0 < abs(point1[1] - point2[1]) <= threshold

    def is_at_same_height(self, point1, point2, threshold=30):
        return 0 < abs(point1[1] - point2[1]) < threshold

    def is_in_horizontal_range(self, point1, point2, threshold=30):
        return 0 < abs(point1[0] - point2[0]) < threshold

    def is_right_of(self, point1, point2, threshold=30):
        return point2[0] - point1[0] > threshold

    def is_left_of(self, point1, point2, threshold=30):
        return point1[0] - point2[0] > threshold

    def is_near(self, point1, point2, threshold=50):
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        return distance < threshold
    
    def detect_pose_actions(self, keypoints):
        thershold_bt_shoulders = (keypoints[self.LEFT_SHOULDER][0] - keypoints[self.RIGHT_SHOULDER][0])//3
        thershold_vertical = (keypoints[self.NOSE][1] - keypoints[self.LEFT_EYE][1])*2
        thershold_touch_eye = (keypoints[self.LEFT_EYE][0] - keypoints[self.RIGHT_EYE][0])//2
        thershold_touch_elbow = (keypoints[self.NOSE][1] - keypoints[self.LEFT_EYE][1])*4
        
        if self.current_challenge == 1:
            if (self.is_above(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER]) and 
                self.is_above(keypoints[self.RIGHT_ELBOW], keypoints[self.RIGHT_SHOULDER]) and
                keypoints[self.RIGHT_WRIST][1] != 0 and keypoints[self.RIGHT_ELBOW][1] != 0):
                print(DICT_GESTURES.get(1))
                return True
        
        elif self.current_challenge == 2:
            if (self.is_above(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER]) and 
                self.is_above(keypoints[self.LEFT_ELBOW], keypoints[self.LEFT_SHOULDER]) and
                keypoints[self.LEFT_WRIST][1] != 0 and keypoints[self.LEFT_ELBOW][1] != 0):
                print(DICT_GESTURES.get(2))
                return True
        
        elif self.current_challenge == 3:
            arm_left_up = (self.is_above(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER]) and 
                            self.is_above(keypoints[self.LEFT_ELBOW], keypoints[self.LEFT_SHOULDER]) and
                            keypoints[self.LEFT_WRIST][1] != 0 and keypoints[self.LEFT_ELBOW][1] != 0)
            arm_right_up = (self.is_above(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER]) and 
                            self.is_above(keypoints[self.RIGHT_ELBOW], keypoints[self.RIGHT_SHOULDER]) and
                            keypoints[self.RIGHT_WRIST][1] != 0 and keypoints[self.RIGHT_ELBOW][1] != 0)
            if arm_left_up and arm_right_up:
                print(DICT_GESTURES.get(3))
                return True
        
        elif self.current_challenge == 4:
            if (self.is_below(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_HIP]) and 
                self.is_below(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_HIP])):
                print(DICT_GESTURES.get(4))
                return True
        
        elif self.current_challenge == 5:
            if (self.is_at_same_height(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], 100) and 
                self.is_in_horizontal_range(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], 100)):
                print(DICT_GESTURES.get(5))
                return True

        elif self.current_challenge == 6:
            if (self.is_at_same_height(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], 100) and 
                self.is_in_horizontal_range(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], 100)):
                print(DICT_GESTURES.get(6))
                return True

        elif self.current_challenge == 7:
            arm_left_forward = (self.is_at_same_height(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], 100) and 
                                self.is_in_horizontal_range(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], 100))
            arm_right_forward = (self.is_at_same_height(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], 100) and 
                                self.is_in_horizontal_range(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], 100))
            if arm_left_forward and arm_right_forward:
                print(DICT_GESTURES.get(7))
                return True
        
        elif self.current_challenge == 8:
            if (self.is_above(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER]) and 
                self.is_above(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER]) and
                self.is_left_of(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], thershold_bt_shoulders) and 
                self.is_right_of(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], thershold_bt_shoulders)):
                print(DICT_GESTURES.get(8))
                return True
        
        elif self.current_challenge == 9:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.NOSE], thershold_vertical):
                print(DICT_GESTURES.get(9))
                return True
        
        elif self.current_challenge == 10:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.NOSE], thershold_vertical):
                print(DICT_GESTURES.get(10))
                return True
        
        elif self.current_challenge == 11:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_EYE], thershold_touch_eye):
                print(DICT_GESTURES.get(11))
                return True

        elif self.current_challenge == 12:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_EYE], thershold_touch_eye):
                print(DICT_GESTURES.get(12))
                return True
        
        elif self.current_challenge == 13:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_EYE], thershold_touch_eye):
                print(DICT_GESTURES.get(13))
                return True

        elif self.current_challenge == 14:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_EYE], thershold_touch_eye):
                print(DICT_GESTURES.get(14))
                return True
        
        elif self.current_challenge == 15:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_EAR], thershold_vertical):
                print(DICT_GESTURES.get(15))
                return True

        elif self.current_challenge == 16:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_EAR], thershold_vertical):
                print(DICT_GESTURES.get(16))
                return True
        
        elif self.current_challenge == 17:
            wrist_right = self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_EAR], thershold_vertical)
            wrist_left = self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_EAR], thershold_vertical)
            if wrist_right and wrist_left:
                print(DICT_GESTURES.get(17))
                return True
        
        elif self.current_challenge == 18:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_EAR], thershold_vertical):
                print(DICT_GESTURES.get(18))
                return True

        elif self.current_challenge == 19:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_EAR], thershold_vertical):
                print(DICT_GESTURES.get(19))
                return True

        elif self.current_challenge == 20:
            wrist_left = self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_EAR], thershold_vertical)
            wrist_right = self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_EAR], thershold_vertical)
            if wrist_left and wrist_right:
                print(DICT_GESTURES.get(20))
                return True
        
        elif self.current_challenge == 21:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], thershold_vertical):
                print(DICT_GESTURES.get(21))
                return True
        
        elif self.current_challenge == 22:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_SHOULDER], thershold_vertical):
                print(DICT_GESTURES.get(22))
                return True
        
        elif self.current_challenge == 23:
            if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], thershold_vertical):
                print(DICT_GESTURES.get(23))
                return True
        
        elif self.current_challenge == 24:
            if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_SHOULDER], thershold_vertical):
                print(DICT_GESTURES.get(24))
                return True
        
        elif self.current_challenge == 25:
            if (self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_ELBOW], thershold_touch_elbow) and
                keypoints[self.LEFT_ELBOW][1] != 0 and keypoints[self.RIGHT_WRIST][1] != 0):
                print(DICT_GESTURES.get(25))
                return True
        
        elif self.current_challenge == 26:
            if (self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_ELBOW], thershold_touch_elbow) and
                keypoints[self.RIGHT_ELBOW][1] != 0 and keypoints[self.LEFT_WRIST][1] != 0):
                print(DICT_GESTURES.get(26))
                return True
        
        return False
    
    def detect_poses(self):
        """Detect poses from camera feed and publish results"""
        if self.current_challenge is None:
            return
        
        self.get_logger().info("Running YOLO model")
        results = self.model(self.frame)
        self.get_logger().info("YOLO model finished")
        
        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            person_keypoints = []
            for j, (x, y) in enumerate(keypoints.xy[0].cpu().numpy()):
                person_keypoints.append((float(x), float(y)))

            result_msg = PoseResult()
            result_msg.challenge = self.current_challenge
            result_msg.detected_poses = self.detect_pose_actions(person_keypoints)
            result_msg.timestamp = self.get_clock().now().to_msg()
            self.get_logger().info(f"Challenge: {self.current_challenge}")
            
            self.pose_result_publisher.publish(result_msg)
    
    def destroy_node(self):
        """Cleanup resources when node is destroyed"""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CocoPoseDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
