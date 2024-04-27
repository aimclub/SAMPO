cmake_policy(SET CMP0091 NEW)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Название текущего компонента
get_filename_component(CUR_PROJ ${CMAKE_CURRENT_SOURCE_DIR} NAME)

sampo_add_headers_and_sources(${CUR_PROJ} include/native src)
Python3_add_library(${CUR_PROJ} MODULE ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})

sampo_common_targets_options(${CUR_PROJ})
target_include_directories(
    ${CUR_PROJ}
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    ${CUR_PROJ}
    PRIVATE
    third_party::pybind11
)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## Setup DLL support ##
if (WIN32)
    set(DLLOADER_SRC
            modules/native/include/native/timeEstimatorLibrary/Windows/DLLoader.h
    )

    include_directories(
            include/native/timeEstimatorLibrary/Windows/
    )

    message("Python dirs: ${PYTHON_INCLUDE_DIRS}/../Lib/site-packages/numpy/core/include")

    # this now works only for Windows, for other OS this path should have another construction
    set(NUMPY_INCLUDE_DIR "${PYTHON_INCLUDE_DIRS}/../Lib/site-packages/numpy/core/include")

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

include_directories($<TARGET_PROPERTY:third_party::pybind11,INCLUDE_DIRECTORIES>)
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

file(READ ${CMAKE_CURRENT_LIST_DIR}/VERSION.txt version)
sampo_install_python_module(TARGETS ${CUR_PROJ} VERSION ${version} NAME ${CUR_PROJ})

add_executable(TARGETS ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})

target_include_directories(
        TARGETS
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
        TARGETS
        PRIVATE
        third_party::pybind11
)
