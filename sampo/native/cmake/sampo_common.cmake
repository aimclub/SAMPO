function(sampo_add_targets_alias alias)
    foreach(target ${ARGN})
        if (${target} MATCHES "CONAN_LIB::.*")
            continue()
        endif()

        get_target_property(target_type ${target} TYPE)
        if (NOT target_type STREQUAL "EXECUTABLE")
            add_library(${alias}::${target} ALIAS ${target})
        endif()
    endforeach()   
endfunction()

function(sampo_common_targets_options)
    foreach(target ${ARGN})
        set_target_properties(${target} PROPERTIES CXX_STANDARD ${SAMPO_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)


        get_target_property(target_type ${target} TYPE)
        if (${target_type} STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif()
        
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:MSVC>:/utf-8>>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/utf-8>>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:MSVC>:/MP>>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/MP>>")
        target_link_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/ignore:4099>>")
        target_compile_options(${target} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fvisibility=hidden -Wall -Wextra -Wno-psabi -Wno-strict-aliasing>)

        if (NOT ${target_type} STREQUAL "EXECUTABLE")
            target_compile_definitions(${target} PRIVATE SAMPO_EXPORTS)
            if (NOT WIN32)
                set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
            endif()
        else()
            target_compile_definitions(${target} PRIVATE SAMPO_IMPORTS)
        endif()

        # message("Target: ${target}")

        string(REPLACE "." "_" target_name $<TARGET_NAME:${target}>)
        string(REPLACE "-" "_" target_name ${target_name})
        target_compile_definitions(${target} PRIVATE SAMPO_COMPONENT_NAME=${target_name})

    endforeach()
endfunction()

function(sampo_target_public_header target headers)
    set(one_val_args DESTINATION)
    cmake_parse_arguments(PARSED "" "${one_val_args}" "" ${ARGN})

    get_target_property(count ${target} SAMPO_PUBLIC_HEADER_COUNT)
    if (count STREQUAL "count-NOTFOUND")
        set(count 0)
    endif()

    set_target_properties(${target} PROPERTIES SAMPO_PUBLIC_HEADER_${count} ${headers})
    if (PARSED_DESTINATION)
        set_target_properties(${target} PROPERTIES SAMPO_PUBLIC_HEADER_${count}_DESTINATION ${PARSED_DESTINATION})
    endif()

    math(EXPR count "${count}+1")
    set_target_properties(${target} PROPERTIES SAMPO_PUBLIC_HEADER_COUNT ${count})
endfunction()

function(sampo_add_unit_test name sources)
    add_executable(${name} ${sources})
    sampo_common_targets_options(${name})
    target_link_libraries(${name} PRIVATE third_party::GTest::gtest third_party::GTest::gtest_main)

    # gtest_add_tests добавляет тест даже в случае, если цель исключена из `all`,
    # поэтому приходится отдельно это проверять, чтобы не добавить в тесты цель, которая не будет собрана.
    is_excluded_from_all(${name} excluded)
    if (NOT ${excluded})
        gtest_add_tests(TARGET ${name} SOURCES ${sources} TEST_LIST tests)
    
        foreach(test ${tests})
            _add_test_runtime_deps_path(${test})
        endforeach()
    endif()
endfunction()

function(get_targets_from_all output)
    set(targets)
    _get_targets_recursive(targets ${SAMPO_SOURCE_DIR})
    list(REMOVE_DUPLICATES targets)
    set(${output} ${targets} PARENT_SCOPE)
endfunction()

function(is_excluded_from_all target result)
    get_target_property(target_excluded ${target} EXCLUDE_FROM_ALL)
    if (target_excluded)
        set(${result} True PARENT_SCOPE)
        return()
    endif()

    get_target_property(dir ${target} SOURCE_DIR)
    while(NOT ${dir} STREQUAL ${SAMPO_SOURCE_DIR})
        get_property(directory_excluded DIRECTORY ${dir} PROPERTY EXCLUDE_FROM_ALL)
        if (directory_excluded)
            set(${result} True PARENT_SCOPE)
            return()
        endif()
        get_filename_component(dir ${dir} DIRECTORY)
    endwhile()
    
    set(${result} False PARENT_SCOPE)
endfunction()

macro(_get_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        _get_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target ${current_targets})
        is_excluded_from_all(${target} excluded)
        if (NOT ${excluded})
            list(APPEND ${targets} ${current_targets})
        endif()
    endforeach()
endmacro()
