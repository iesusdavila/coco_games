#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    pose_detector_node = Node(
        package='coco_dice',
        executable='pose_detector',
        name='pose_detector',
        output='screen'
    )

    body_landmarks = Node(
        package='coco_dice',
        executable='body_landmarks.py',
        name='body_landmarks',
        output='screen'
    )

    speaker_node = Node(
        package='coco_dice',
        executable='speaker_node.py',
        name='speaker_node',
        output='screen'
    )

    usb_cam_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam_node_exe",
        output="screen"
    )

    game_manager_node = Node(
        package='coco_dice',
        executable='game_manager',
        name='game_manager',
        output='screen'
    )

    return LaunchDescription([
        usb_cam_node,
        body_landmarks,
        RegisterEventHandler(
            OnProcessStart(
                target_action=body_landmarks,
                on_start=[
                    TimerAction(
                        period=3.0,
                        actions=[pose_detector_node]
                    )
                ]
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=pose_detector_node,
                on_start=[
                    TimerAction(
                        period=3.0,
                        actions=[speaker_node]
                    )
                ]
            )
        ),
        game_manager_node,
    ])
