#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from coco_interfaces.msg import PoseResult
import os
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
from ultralytics import YOLO
import time

class CocoPoseDetector(Node):
    def __init__(self):
        super().__init__('coco_pose_detector')
        
        # Publishers
        self.pose_result_publisher = self.create_publisher(
            PoseResult, '/pose_result', 10)
        
        self.bridge = CvBridge()
        self.frame = None

        # Subscribers
        self.create_subscription(
            String, '/current_challenge', self.handle_new_challenge, 10)
        self.create_subscription(
            Image, '/image_raw', self.handle_camera_image, 10)
        
        # Initialize YOLO model
        pkg_share_dir = get_package_share_directory('coco_dice')
        model_path = os.path.join(pkg_share_dir, 'models', 'yolov8s-pose.pt')
        self.model = YOLO(model_path)
                
        # Game state
        self.current_challenge = None
        self.game_active = False
        
        # Detection parameters (based on your YOLO script)
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
        
        # Set timer for pose detection (30fps)
        self.create_timer(1/30, self.detect_poses)
        
        self.get_logger().info('Coco Pose Detector started successfully')
    
    def handle_new_challenge(self, msg):
        """Handle a new challenge from the game manager"""
        self.current_challenge = msg.data
        self.get_logger().info(f'New challenge received: {self.current_challenge}')
    
    def handle_camera_image(self, msg):
        """Handle camera image feed and save last frame"""
        # Convert ROS Image message to OpenCV format
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
        """Detect poses based on keypoints"""
        poses = []

        # Calculate dynamic thresholds based on body proportions
        thershold_bt_shoulders = (keypoints[self.LEFT_SHOULDER][0] - keypoints[self.RIGHT_SHOULDER][0])//3
        thershold_vertical = (keypoints[self.NOSE][1] - keypoints[self.LEFT_EYE][1])*2
        thershold_touch_eye = (keypoints[self.LEFT_EYE][0] - keypoints[self.RIGHT_EYE][0])//2
        thershold_touch_elbow = (keypoints[self.NOSE][1] - keypoints[self.LEFT_EYE][1])*4
        
        # Right arm up
        if (self.is_above(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER]) and 
            self.is_above(keypoints[self.RIGHT_ELBOW], keypoints[self.RIGHT_SHOULDER]) and
            keypoints[self.RIGHT_WRIST][1] != 0 and keypoints[self.RIGHT_ELBOW][1] != 0):
            poses.append("Brazo derecho arriba")
        
        # Left arm up
        if (self.is_above(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER]) and 
            self.is_above(keypoints[self.LEFT_ELBOW], keypoints[self.LEFT_SHOULDER]) and
            keypoints[self.LEFT_WRIST][1] != 0 and keypoints[self.LEFT_ELBOW][1] != 0):
            poses.append("Brazo izquierdo arriba")
        
        # Both arms up
        if "Brazo derecho arriba" in poses and "Brazo izquierdo arriba" in poses:
            poses.append("Ambos brazos arriba")
        
        # Both arms down
        if (self.is_below(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_HIP]) and 
            self.is_below(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_HIP])):
            poses.append("Ambos brazos abajo")
        
        # Right arm forward
        if (self.is_at_same_height(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], 100) and 
            self.is_in_horizontal_range(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], 100)):
            poses.append("Brazo derecho hacia delante")

        # Left arm forward
        if (self.is_at_same_height(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], 100) and 
            self.is_in_horizontal_range(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], 100)):
            poses.append("Brazo izquierdo hacia delante")

        # Both arms forward
        if "Brazo derecho hacia delante" in poses and "Brazo izquierdo hacia delante" in poses:
            poses.append("Ambos brazos hacia delante")
        
        # X symbol with arms
        if (self.is_above(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER]) and 
            self.is_above(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER]) and
            self.is_left_of(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], thershold_bt_shoulders) and 
            self.is_right_of(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], thershold_bt_shoulders)):
            poses.append("Símbolo X con los brazos")
        
        # Touch nose
        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.NOSE], thershold_vertical):
            poses.append("Tocando nariz con muneca derecha")
        
        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.NOSE], thershold_vertical):
            poses.append("Tocando nariz con muneca izquierda")
        
        # Touch eyes
        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_EYE], thershold_touch_eye):
            poses.append("Tocando ojo izquierdo con muneca izquierda")

        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_EYE], thershold_touch_eye):
            poses.append("Tocando ojo izquierdo con muneca derecha")
        
        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_EYE], thershold_touch_eye):
            poses.append("Tocando ojo derecho con muneca derecha")

        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_EYE], thershold_touch_eye):
            poses.append("Tocando ojo derecho con muneca izquierda")
        
        # Touch ears
        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_EAR], thershold_vertical):
            poses.append("Tocando oreja derecha con mano derecha")

        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_EAR], thershold_vertical):
            poses.append("Tocando oreja derecha con mano izquierda")

        if ("Tocando oreja derecha con mano derecha" in poses and 
            "Tocando oreja derecha con mano izquierda" in poses):
            poses.append("Tocando oreja derecha con ambas manos")
        
        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_EAR], thershold_vertical):
            poses.append("Tocando oreja izquierda con mano izquierda")

        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_EAR], thershold_vertical):
            poses.append("Tocando oreja izquierda con mano derecha")

        if ("Tocando oreja izquierda con mano izquierda" in poses and 
            "Tocando oreja izquierda con mano derecha" in poses):
            poses.append("Tocando oreja izquierda con ambas manos")
        
        # Touch shoulders
        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.LEFT_SHOULDER], thershold_vertical):
            poses.append("Tocando hombro izquierdo con mano izquierda")
        
        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_SHOULDER], thershold_vertical):
            poses.append("Tocando hombro izquierdo con mano derecha")
        
        if self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.RIGHT_SHOULDER], thershold_vertical):
            poses.append("Tocando hombro derecho con mano derecha")
        
        if self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_SHOULDER], thershold_vertical):
            poses.append("Tocando hombro derecho con mano izquierda")
        
        # Touch elbows
        if (self.is_near(keypoints[self.RIGHT_WRIST], keypoints[self.LEFT_ELBOW], thershold_touch_elbow) and
            keypoints[self.LEFT_ELBOW][1] != 0 and keypoints[self.RIGHT_WRIST][1] != 0):
            poses.append("Tocando codo izquierdo con mano derecha")
        
        if (self.is_near(keypoints[self.LEFT_WRIST], keypoints[self.RIGHT_ELBOW], thershold_touch_elbow) and
            keypoints[self.RIGHT_ELBOW][1] != 0 and keypoints[self.LEFT_WRIST][1] != 0):
            poses.append("Tocando codo derecho con mano izquierda")
        
        return poses
    
    def detect_poses(self):
        """Detect poses from camera feed and publish results"""
        if self.current_challenge is None:
            return
        
        # Run YOLO model on the frame
        self.get_logger().info("Running YOLO model")
        results = self.model(self.frame)
        self.get_logger().info("YOLO model finished")
        
        # Extract keypoints
        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            person_keypoints = []
            for j, (x, y) in enumerate(keypoints.xy[0].cpu().numpy()):
                person_keypoints.append((float(x), float(y)))
            
            # Detect poses
            detected_poses = self.detect_pose_actions(person_keypoints)
            
            # Send the result
            result_msg = PoseResult()
            result_msg.challenge = self.current_challenge
            result_msg.detected_poses = detected_poses
            result_msg.timestamp = self.get_clock().now().to_msg()
            self.get_logger().info(f"Challenge: {self.current_challenge}")
            self.get_logger().info(f"Detected poses: {detected_poses}")
            
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
