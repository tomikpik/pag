cmake_minimum_required(VERSION 3.9)
project(hw3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp main_skeleton.cpp nechci.cpp)
add_executable(hw3 ${SOURCE_FILES})