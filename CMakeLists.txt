cmake_minimum_required(VERSION 3.0.2)
project(ransac_comparison)

add_compile_options(-std=c++14)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp  
  rospy
  std_msgs
  pcl_ros
  tf_conversions
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES unld_object_detection
 CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(pointcloud_generator
  src/pointcloud_generator.cpp
)

target_link_libraries(pointcloud_generator
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(ransac1
  src/ransac1.cpp
)

target_link_libraries(ransac1
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(ransac2
  src/ransac2.cpp
)

target_link_libraries(ransac2
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)