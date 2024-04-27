# add_third_party folder 
macro(add_third_party folder)
    set(options ADDITONAL_ALIAS)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(PARSED "${options}" "" "${multiValueArgs}" ${ARGN})

    if (NOT IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${folder}")
        message(FATAL_ERROR "No such third_party folder '${folder}'. Typo in the base folder name?")
    endif()

    file(GLOB files "${folder}/*")
    if (NOT files)
        message(FATAL_ERROR "submodule ${folder} is missing. To fix try run:  git submodule update --init")
    endif()

    sampo_add_subdirectory(${folder} EXCLUDE_FROM_ALL)
    if (NOT PARSED_TARGETS)
        get_targets_in_subdirectory(folder_targets ${folder})
    else()
        set(folder_targets ${PARSED_TARGETS})
    endif()

    if (PARSED_ADDITONAL_ALIAS)
        sampo_add_targets_alias(third_party::${PARSED_ADDITONAL_ALIAS} ${folder_targets})
    else()
        sampo_add_targets_alias(third_party ${folder_targets})
    endif()
endmacro()


#macro(add_runtime_deps_path)
#
#endmacro()

#
# Добавляем third party сабмодули
#
add_third_party(pybind11)

# 
# Добавляем third party собранные бинарники через систему контроля зависимостей (в честности conan)
#
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/conan.cmake")
if (${CONAN_VERIFY_SSL})
    conan_add_remote(NAME bnipi_sampo INDEX 1 URL ${CONAN_SERVER} VERIFY_SSL True)
else()
    conan_add_remote(NAME bnipi_sampo INDEX 1 URL ${CONAN_SERVER} VERIFY_SSL False)
endif()
# Устанавливаем бесшумный режим conan.cmake, иначе будет очень шумно
set(CONAN_CMAKE_SILENT_OUTPUT ON)
# Автоматическое определение настроек (компилятор, версия, runtime library и тд.)
conan_cmake_autodetect(settings OUTPUT_QUIET ON)
conan_check(REQUIRED)

# Далее добавляем директории скриптов conan для каждого third party
add_third_party(boost-conan)
add_third_party(gtest-conan)
add_third_party(google-benchmark-conan)


# В конце обязательно повышаем область видимости глобальных переменных, измененных в результате работы conan.
set(sampo_RUNTIME_DEPS_PATHS ${sampo_RUNTIME_DEPS_PATHS} PARENT_SCOPE)