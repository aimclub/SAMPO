cmake_minimum_required(VERSION 3.24)

set(CMAKE_C_COMPILER "C:\\msys64\\mingw64\\bin\\gcc")
set(CMAKE_CXX_COMPILER "C:\\msys64\\mingw64\\bin\\g++")

project(native CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(native native.cpp native.h dtime.cpp dtime.h python_deserializer.cpp python_deserializer.h workgraph.h contractor.h basic_types.h pycodec.cpp pycodec.h)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(native ${PYTHON_LIBRARIES})
