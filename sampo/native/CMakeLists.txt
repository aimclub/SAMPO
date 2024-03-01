cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0091 NEW)

#set(CMAKE_C_COMPILER "C:\\msys64\\mingw64\\bin\\gcc")
#set(CMAKE_CXX_COMPILER "C:\\msys64\\mingw64\\bin\\g++")
#set(CMAKE_C_COMPILER "gcc")
#set(CMAKE_CXX_COMPILER "g++")

# project(native CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(PythonLibs REQUIRED)

include_guard(GLOBAL)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if (NOT SAMPO_STANDALONE_CONFIGURE)
    project(SAMPO LANGUAGES CXX)
    set(SAMPO_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
endif()

set(SAMPO_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX d)
set(SAMPO_RUNTIME_DEPS_PATHS "")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(SAMPO_SYSTEM_DEPS_EXCLUDE [Ss]ystem32/ api-ms- ext-ms- python libstd libgcc libc libm ld-linux)
set(SAMPO_SYSTEM_DEPS_INCLUDE vcruntime msvcp vcomp libomp libiomp)

set_property(GLOBAL PROPERTY SAMPO_RUNTIME_DEPS_EXCLUDE)
set_property(GLOBAL PROPERTY SAMPO_RUNTIME_DEPS_INCLUDE)
set_property(GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_EXCLUDE ${SAMPO_SYSTEM_DEPS_EXCLUDE})
set_property(GLOBAL APPEND PROPERTY SAMPO_SYSTEM_DEPS_INCLUDE ${SAMPO_SYSTEM_DEPS_INCLUDE})

# Remove MinSizeRel from configurations
set(
    CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo"
    CACHE STRING "Specifies the available build types (configurations) on multi-config generators"
    FORCE
)

set(CONAN_SERVER "https://center.conan.io/" CACHE STRING "Conan server address.")
option(CONAN_VERIFY_SSL "Conan use ssl." ON)

option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis." OFF)

# Enable all все cmake utils
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/)

enable_testing()
include(GoogleTest)

find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)

# Setting right paths on Linux
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(OpenMP)
# FIX: CMake не находит библиотек OpenMP для линковки на Windows с компилятором clang
if (WIN32 AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_link_libraries(OpenMP::OpenMP_CXX INTERFACE libomp)
endif()
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES SAMPO_FINDPACKAGE_NAME "OpenMP")
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES SAMPO_FINDPACKAGE_VERSION "2.0")
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES SAMPO_FINDPACKAGE_EXACT_VERSION False)

include(sampo_simd_detector)
include(sampo_subdirectory)
include(sampo_glob_sources)
include(sampo_install_utils)
include(sampo_simd_helper)
include(sampo_common)
include(sampo_clang_tools)
include(ci)

sampo_add_subdirectory(third_party)
sampo_add_subdirectory(modules)
sampo_add_ci_copy_deps()

if (NOT SAMPO_STANDALONE_CONFIGURE)
    sampo_install_runtime_deps()
endif()


set(CPACK_GENERATOR "ZIP" CACHE STRING "Generator for CPack.")
set(CPACK_OUTPUT_FILE_PREFIX ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "CPack output prefix.")
set_property(CACHE CPACK_GENERATOR PROPERTY STRINGS "ZIP;TGZ;DEB;NSIS")
mark_as_advanced(CPACK_GENERATOR)

set(CPACK_ARCHIVE_COMPONENT_INSTALL True)
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_ALL Runtime Development)

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "SAMPO")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEB_COMPONENT_INSTALL ON)

if (NOT SAMPO_STANDALONE_CONFIGURE)
    include(CPack)
endif()

message(${CMAKE_SOURCE_DIR}/modules/native/include/${CUR_PROJ})
include_directories(${CMAKE_SOURCE_DIR}/modules/native/include/${CUR_PROJ})

#include_directories(
#        modules/native/timeEstimatorLibrary
#)
#
#add_executable(native
#        native.cpp native.h                                # main files
#        dtime.cpp dtime.h                                  # Time implementation
#        python_deserializer.cpp python_deserializer.h      # custom Python datastruct handler
#        workgraph.h contractor.h modules/native/basic_types.h             # custom Python types definition
#        pycodec.h                                          # basic type transcoder
#        modules/native/chromosome_evaluator.cpp modules/native/chromosome_evaluator.h genetic.h evaluator_types.h # algorithms
#        modules/native/utils/use_numpy.cpp modules/native/utils/use_numpy.h
#        time_estimator.h time_estimator.cpp
#        ${DLLOADER_SRC}
#        external.h)

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    message("OpenMP found")
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

#message(${PYTHON_INCLUDE_DIRS})
#message(${NUMPY_INCLUDE_DIR})

#include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
#target_link_libraries(native ${PYTHON_LIBRARIES})
#target_link_libraries(native ${CMAKE_DL_LIBS})

#find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)

