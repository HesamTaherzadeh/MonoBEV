cmake_minimum_required(VERSION 3.8)
project(bev)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(CUDA REQUIRED)
find_library(CUDNN_LIBRARY cudnn HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(bev_interface REQUIRED)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${onnxruntime_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Declare the executable
add_executable(${PROJECT_NAME}_node src/bev_node.cpp src/BEVTransformer.cpp )

target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBS}
  onnxruntime
  ${CUDA_LIBRARIES}
  ${CUDNN_LIBRARY}
)

# Add dependencies for the executable
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  cv_bridge
  image_transport
  sensor_msgs
  tf2_ros
  nav_msgs
  std_msgs
  pcl_conversions
  bev_interface
)

# Link OpenCV libraries
# Install the executable
install(TARGETS ${PROJECT_NAME}_node
DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY cfg/
  DESTINATION share/${PROJECT_NAME}/cfg
)


# Export dependencies
ament_export_include_directories(include)
ament_export_dependencies(
  cv_bridge
  image_transport
  rclcpp
  sensor_msgs
  tf2_ros
  nav_msgs
  std_msgs
  onnxruntime
  OpenCV
  rosidl_default_runtime
)

ament_package()
