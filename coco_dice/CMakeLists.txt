cmake_minimum_required(VERSION 3.8)
project(coco_dice)

# ========== Configuración para C++ ==========
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(coco_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

# Configuración del ejecutable C++
add_executable(game_manager src/game_manager.cpp)
ament_target_dependencies(game_manager 
  rclcpp 
  coco_interfaces 
  std_msgs
)

add_executable(pose_detector src/pose_detector.cpp)
ament_target_dependencies(pose_detector 
  rclcpp 
  coco_interfaces 
  std_msgs
)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)

install(PROGRAMS
  scripts/body_landmarks.py
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
  src
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS 
  game_manager
  pose_detector
  DESTINATION lib/${PROJECT_NAME}
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
