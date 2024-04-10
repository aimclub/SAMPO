cmake_policy(SET CMP0091 NEW)
cmake_minimum_required(VERSION 3.24)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis." OFF)

option(ENABLE_TEST_EXECUTABLE "Enables test-debug executable assemble" OFF)


get_filename_component(CUR_PROJ ${CMAKE_CURRENT_SOURCE_DIR} NAME)

sampo_add_headers_and_sources(${CUR_PROJ} include/native src)
Python3_add_library(${CUR_PROJ} MODULE ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")

sampo_common_targets_options(${CUR_PROJ})
target_include_directories(
    ${CUR_PROJ}
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${PYTHON_LIBRARIES}
)

message("${CMAKE_CURRENT_LIST_DIR}/../../../venv/Lib/site-packages/numpy/core/include")

## Setup DLL support ##
if (WIN32)
    set(DLLOADER_SRC
            modules/native/include/native/timeEstimatorLibrary/Windows/DLLoader.h
    )

    include_directories(
            include/native/timeEstimatorLibrary/Windows/
    )

    # this now works only for Windows, for other OS this path should have another construction
    set(NUMPY_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../venv/Lib/site-packages/numpy/core/include")

endif(WIN32)

if (UNIX)
    set(DLLOADER_SRC
            include/native/timeEstimatorLibrary/Unix/DLLoader.h
    )

    include_directories(
            include/native/timeEstimatorLibrary/Unix/
    )

    set (CMAKE_CXX_FLAGS "-W -Wall -Wextra")

    # TODO Check validity of path
    set(NUMPY_INCLUDE_DIR "${PYTHON_INCLUDE_DIRS}/site-packages/numpy/core/include")
endif(UNIX)

include_directories(
        include/native/timeEstimatorLibrary
)

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

target_link_libraries(
    ${CUR_PROJ}
    PRIVATE
    third_party::pybind11
    ${PYTHON_LIBRARIES}
)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories($<TARGET_PROPERTY:third_party::pybind11,INCLUDE_DIRECTORIES>)

file(READ ${CMAKE_CURRENT_LIST_DIR}/VERSION.txt version)
sampo_install_python_module(TARGETS ${CUR_PROJ} VERSION ${version} NAME ${CUR_PROJ})

# Test executable

if (ENABLE_TEST_EXECUTABLE)
    message("${PYTHON_INCLUDE_DIRS}")

    # include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

    sampo_add_headers_and_sources(TARGETS include/native src)
    include_directories($<TARGET_PROPERTY:third_party::pybind11,INCLUDE_DIRECTORIES>)

    add_executable(TARGETS ${TARGETS_headers} ${TARGETS_sources})

    target_include_directories(
            TARGETS
            PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            "C:/Users/stasb/AppData/Local/Programs/Python/Python310/include"
    )

    target_link_libraries(
            TARGETS
            PRIVATE
            third_party::pybind11
            ${PYTHON_LIBRARIES}
    )
endif()
