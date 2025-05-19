#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from coco_interfaces.msg import PoseResult
import time

class CocoGameManager(Node):
    def __init__(self):
        super().__init__('coco_game_manager')
        
        self.next_challenge_publisher = self.create_publisher(
            Bool, '/next_challenge', 10)
        self.feedback_publisher = self.create_publisher(
            String, '/game_feedback', 10)
        
        self.create_subscription(
            PoseResult, '/pose_result', self.handle_pose_result, 10)
        self.create_subscription(
            Bool, '/audio_playing', self.handle_audio_status, 10)
        
        self.current_challenge = None
        self.score = 0
        self.audio_playing = False
        self.detection_ongoing = False
        self.challenge_timeout = None
        self.waiting_for_pose = False
        self.correct_pose_timer = None
        self.correct_pose_start_time = None
        self.correct_pose_duration = 1.0  
        
        self.challenge_timer = self.create_timer(0.5, self.check_challenge_timeout)
        
        self.get_logger().info('Coco Game Manager started successfully')
        self.get_logger().info("Game started")
    
    def handle_audio_status(self, msg):
        self.audio_playing = msg.data
        
        if not self.audio_playing and not self.detection_ongoing:
            self.start_detection()
    
    def start_detection(self):
        self.detection_ongoing = True
        self.waiting_for_pose = True
        self.challenge_timeout = time.time() + 10.0  
        self.get_logger().info("Detection phase started")
    
    def check_challenge_timeout(self):
        if not self.waiting_for_pose or self.challenge_timeout is None:
            return
            
        if time.time() > self.challenge_timeout:
            self.get_logger().info("Challenge timed out")
            self.handle_failed_challenge("¡Tiempo agotado! Vamos a intentar con otro desafío.")
    
    def handle_pose_result(self, msg):
        self.get_logger().info(f"LEYENDO DATA DE POSE: {msg.detected_poses}")
        if not self.waiting_for_pose or self.audio_playing:
            return

        self.get_logger().info(f"Pose result received: {msg}")
        
        self.current_challenge = msg.challenge
        detected_poses = msg.detected_poses
        
        if detected_poses:
            if self.correct_pose_start_time is None:
                self.correct_pose_start_time = time.time()
                self.get_logger().info(f"Correct pose detected, starting timer")
            elif time.time() - self.correct_pose_start_time >= self.correct_pose_duration:
                self.handle_successful_challenge()
        else:
            self.correct_pose_start_time = None
    
    def handle_successful_challenge(self):
        self.waiting_for_pose = False
        self.detection_ongoing = False
        self.correct_pose_start_time = None
        self.score += 1
        
        self.get_logger().info(f"Challenge completed successfully! Score: {self.score}")
        
        feedback_msg = String()
        feedback_msg.data = f"¡Muy bien! Has completado el desafío. Tu puntuación es {self.score}."
        self.feedback_publisher.publish(feedback_msg)
        
        time.sleep(2)
        
        next_challenge_msg = Bool()
        next_challenge_msg.data = True
        self.next_challenge_publisher.publish(next_challenge_msg)
    
    def handle_failed_challenge(self, feedback_text):
        self.waiting_for_pose = False
        self.detection_ongoing = False
        self.correct_pose_start_time = None
        
        self.get_logger().info(f"Challenge failed: {feedback_text}")
        
        feedback_msg = String()
        feedback_msg.data = feedback_text
        self.feedback_publisher.publish(feedback_msg)
        
        time.sleep(2)
        
        next_challenge_msg = Bool()
        next_challenge_msg.data = True
        self.next_challenge_publisher.publish(next_challenge_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CocoGameManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
