function(sampo_add_ci_copy_deps)
    set(targets_with_deps)
    set(supported_types MODULE_LIBRARY SHARED_LIBRARY EXECUTABLE)

    get_targets_from_all(targets)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/copy_depth.cmake "")

    foreach(target ${targets})
        get_target_property(type ${target} TYPE)
        if (${type} IN_LIST supported_types)
            list(APPEND targets_with_deps ${target})
        endif()
    endforeach()

    set(PACKAGE_TARGETS)
    if (NOT MSVC)
        set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
    else()
        set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    endif()

    get_property(SAMPO_SYSTEM_DEPS_EXCLUDE GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_EXCLUDE)
    get_property(SAMPO_SYSTEM_DEPS_INCLUDE GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_INCLUDE)

    foreach(target ${targets_with_deps})
        list(APPEND PACKAGE_TARGETS "$<TARGET_FILE:${target}>")
        list(APPEND SAMPO_SYSTEM_DEPS_EXCLUDE "$<TARGET_NAME:${target}>")
    endforeach()

    list(JOIN PACKAGE_TARGETS " " PACKAGE_TARGETS)
    list(JOIN SAMPO_SYSTEM_DEPS_INCLUDE " " RUNTIME_DEPS_INCLUDE)
    list(JOIN SAMPO_SYSTEM_DEPS_EXCLUDE " " RUNTIME_DEPS_EXCLUDE)
    list(JOIN SAMPO_RUNTIME_DEPS_PATHS " " RUNTIME_DEPS_PATHS)

    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/utils/resources/copy_depth.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/copy_depth.cmake.in
        @ONLY
    )

    file(
        GENERATE
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copy_depth-$<CONFIG>.cmake
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/copy_depth.cmake.in
    )

    add_custom_target(
        ci_copy_deps
        COMMAND ${CMAKE_COMMAND} "-P ${CMAKE_CURRENT_BINARY_DIR}/copy_depth-$<CONFIG>.cmake"
    )
endfunction()
