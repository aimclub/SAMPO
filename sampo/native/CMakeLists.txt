cmake_minimum_required(VERSION 3.24)
project(native)

set(CMAKE_CXX_STANDARD 11)

add_executable(native main.cpp)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(native ${PYTHON_LIBRARIES})
