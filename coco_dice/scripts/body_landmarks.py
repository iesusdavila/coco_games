#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from coco_interfaces.msg import PoseLandmarks
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory

class BodyPointsDetectorNode(Node):
    def __init__(self):
        super().__init__('body_points_detector_node')
        self.bridge = CvBridge()
        pkg_share_dir = get_package_share_directory('coco_dice')
        model_path = os.path.join(pkg_share_dir, 'models', 'yolov8s-pose.pt')
        self.model = YOLO(model_path)
        
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(PoseLandmarks, 'pose_landmarks', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame)

        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            landmarks_msg = PoseLandmarks()
            landmarks_msg.header = msg.header
            for j, (x, y) in enumerate(keypoints.xy[0].cpu().numpy()):
                point = Point()
                point.x = float(x)
                point.y = float(y)
                landmarks_msg.landmarks.append(point)

            self.publisher.publish(landmarks_msg)        

def main(args=None):
    rclpy.init(args=args)
    node = BodyPointsDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()