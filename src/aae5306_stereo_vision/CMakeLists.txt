cmake_minimum_required(VERSION 3.0.2)
project(aae5306_stereo_vision)

## Find catkin and required packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
  tf2_ros
  visualization_msgs
  message_generation
)

## Generate messages
add_message_files(
  FILES
  FeatureStats.msg
  MatchStats.msg
  DepthStats.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

## Catkin package configuration
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    image_transport
    message_filters
    tf2_ros
    visualization_msgs
    message_runtime
)

## Install Python scripts
catkin_install_python(PROGRAMS
  feature_detection_node.py
  feature_matching_node.py
  depth_recovery_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

## Install rviz config
install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)
