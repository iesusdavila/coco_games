#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int16
import random
import yaml
import os
from piper import PiperVoice
from ament_index_python.packages import get_package_share_directory
import tempfile
import wave
from playsound import playsound
import threading
from collections import deque

class CocoSpeakerNode(Node):
    def __init__(self):
        super().__init__('coco_speaker_node')
        
        pkg_share_dir_tts = get_package_share_directory('coco_chat')
        pkg_share_dir = get_package_share_directory('coco_dice')
        self.tts_model_path = os.path.join(pkg_share_dir_tts, 'models', 'TTS', 'es_MX-claude-high.onnx')
        self.tts_config_path = os.path.join(pkg_share_dir_tts, 'models', 'TTS', 'es_MX-claude-high.onnx.json')
        self.challenges_path = os.path.join(pkg_share_dir, 'config', 'challenges.yaml')
        
        self.voice = None
        self.init_tts()
        
        self.current_challenge_publisher = self.create_publisher(
            Int16, '/current_challenge', 10)
        self.audio_playing_publisher = self.create_publisher(
            Bool, '/audio_playing', 10)
        
        self.create_subscription(
            Bool, '/next_challenge', self.handle_next_challenge, 10)
        
        self.challenges = []
        self.load_challenges()
        self.recent_challenges = deque(maxlen=5) 
        self.current_challenge = None
        self.speaking_lock = threading.Lock()
        
        self.get_logger().info('Coco Speaker Node started successfully')

        self.get_logger().info("Game started, selecting first challenge")
        self.select_and_speak_challenge()
    
    def init_tts(self):
        try:
            self.voice = PiperVoice.load(
                model_path=self.tts_model_path,
                config_path=self.tts_config_path,
                use_cuda=True
            )
            self.get_logger().info("TTS engine initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize TTS engine: {str(e)}")
    
    def load_challenges(self):
        try:
            with open(self.challenges_path, 'r', encoding='utf-8') as file:
                challenge_data = yaml.safe_load(file)
                self.challenges = challenge_data.get('challenges', [])
                self.get_logger().info(f"Loaded {len(self.challenges)} challenges")
        except Exception as e:
            self.get_logger().error(f"Failed to load challenges: {str(e)}")
    
    def select_challenge(self):
        available_challenges = [c for c in self.challenges if c not in self.recent_challenges]
        
        if not available_challenges:
            return random.choice(self.challenges)
        
        return random.choice(available_challenges)
    
    def handle_next_challenge(self, msg):
        if msg.data:
            self.get_logger().info("Next challenge requested")
            self.select_and_speak_challenge()
    
    def select_and_speak_challenge(self):
        challenge = self.select_challenge()
        self.current_challenge = challenge
        
        self.recent_challenges.append(challenge)
        
        challenge_msg = Int16()
        challenge_id = challenge["id"]
        challenge_pose = challenge["pose"]
        challenge_msg.data = challenge_id
        self.get_logger().info(f"Selected challenge: {challenge_pose}")
        self.current_challenge_publisher.publish(challenge_msg)
        
        threading.Thread(target=self.speak_text, args=(challenge['text'],)).start()
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        if self.voice is None:
            self.get_logger().error("TTS engine not initialized")
            return
            
        with self.speaking_lock:
            audio_status_msg = Bool()
            audio_status_msg.data = True
            self.audio_playing_publisher.publish(audio_status_msg)
            
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
                    with wave.open(fp.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.voice.config.sample_rate)
                        self.voice.synthesize(text, wav_file)
                    
                    playsound(fp.name)
            except Exception as e:
                self.get_logger().error(f"Error generating or playing speech: {str(e)}")
            finally:
                audio_status_msg.data = False
                self.audio_playing_publisher.publish(audio_status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CocoSpeakerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
