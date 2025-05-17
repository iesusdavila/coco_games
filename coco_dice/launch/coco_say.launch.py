#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    # Nodo pose_detector (se lanza inmediatamente)
    pose_detector_node = Node(
        package='coco_dice',
        executable='pose_detector.py',
        name='pose_detector',
        output='screen'
    )

    # Nodo speaker_node (se lanzará después con delay)
    speaker_node = Node(
        package='coco_dice',
        executable='speaker_node.py',
        name='speaker_node',
        output='screen'
    )

    # Nodo usb_cam (se lanza sin condición)
    usb_cam_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam_node_exe",
        output="screen"
    )

    # Nodo game_manager (se lanza sin condición)
    game_manager_node = Node(
        package='coco_dice',
        executable='game_manager.py',
        name='game_manager',
        output='screen'
    )

    return LaunchDescription([
        usb_cam_node,
        pose_detector_node,
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
