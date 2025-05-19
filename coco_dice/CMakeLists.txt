cmake_minimum_required(VERSION 3.8)
project(coco_dice)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)

# Instalar el script Python
install(PROGRAMS
  scripts/game_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/body_landmarks.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/pose_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/speaker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Instalación común ==========
install(DIRECTORY
  config
  launch
  models
  scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  rclcpp_action
  control_msgs
  sensor_msgs
  coco_interfaces
  std_msgs
)

ament_package()
