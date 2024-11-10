cmake_minimum_required(VERSION 3.10)
project(depth_to_bev)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

find_package(onnxruntime REQUIRED)

add_executable(depth_to_bev main.cpp src/BEVTransformer.cpp)

target_link_libraries(depth_to_bev ${OpenCV_LIBS} onnxruntime)
