cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0091 NEW)

set(CMAKE_CXX_STANDARD 20)

include_guard(GLOBAL)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if (NOT SAMPO_STANDALONE_CONFIGURE)
    project(SAMPO LANGUAGES CXX)
    set(SAMPO_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
endif()

set(SAMPO_CXX_STANDARD 20)
# set(CMAKE_DEBUG_POSTFIX d)
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
# FIX: CMake does not find OpenMP for linking on Windows with clang
if (WIN32 AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_link_libraries(OpenMP::OpenMP_CXX INTERFACE libomp)
endif()
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES SAMPO_FINDPACKAGE_NAME "OpenMP")
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES SAMPO_FINDPACKAGE_VERSION "2.0")
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES SAMPO_FINDPACKAGE_EXACT_VERSION False)

include(sampo_subdirectory)
include(sampo_glob_sources)
include(sampo_install_utils)
include(sampo_common)
include(ci)

sampo_add_subdirectory(third_party)
sampo_add_subdirectory(native)
sampo_add_ci_copy_deps()

if (NOT SAMPO_STANDALONE_CONFIGURE)
    sampo_install_runtime_deps()
endif()
