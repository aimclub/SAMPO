cmake_minimum_required(VERSION 3.24)

#set(CMAKE_C_COMPILER "C:\\msys64\\mingw64\\bin\\gcc")
#set(CMAKE_CXX_COMPILER "C:\\msys64\\mingw64\\bin\\g++")
#set(CMAKE_C_COMPILER "gcc")
#set(CMAKE_CXX_COMPILER "g++")

project(native CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(native
        native.cpp native.h                                # main files
        dtime.cpp dtime.h                                  # Time implementation
        python_deserializer.cpp python_deserializer.h      # custom Python datastruct handler
        workgraph.h contractor.h basic_types.h             # custom Python types definition
        pycodec.h                                          # basic type transcoder
        chromosome_evaluator.cpp chromosome_evaluator.h)   # algorithms

find_package(PythonLibs REQUIRED)

# this now works only for Windows, for other OS this path should have another construction
set(NUMPY_INCLUDE_DIR "${PYTHON_INCLUDE_DIRS}/../Lib/site-packages/numpy/core/include")
#set(NUMPY_INCLUDE_DIR "/home/quarter/.local/lib/python3.10/site-packages/numpy/core/include")

message(${PYTHON_INCLUDE_DIRS})
message(${NUMPY_INCLUDE_DIR})

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(native ${PYTHON_LIBRARIES})
