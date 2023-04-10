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

find_package(OpenMP REQUIRED)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES unld_object_detection
 CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

add_compile_options(-pg -fprofile-arcs -ftest-coverage)
set(catkin_LIBRARIES ${catkin_LIBRARIES} -pg -fprofile-arcs -ftest-coverage)

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

add_executable(pointcloud_visualizer
  src/pointcloud_visualizer.cpp
)

target_link_libraries(pointcloud_visualizer
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ransac1
  src/ransac1.cpp
)

target_link_libraries(ransac1
  pcl_visualization
  ${OpenMP_LIBS}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_dependencies(ransac1 ${catkin_EXPORTED_TARGETS})

add_executable(ransac2
  src/ransac2.cpp
)

target_link_libraries(ransac2
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(ransac3
  src/ransac3.cpp
)

target_link_libraries(ransac3
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(error
  src/error.cpp
)

target_link_libraries(error
  pcl_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)