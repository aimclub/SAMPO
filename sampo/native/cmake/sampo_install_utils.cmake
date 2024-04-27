function(sampo_install_python_module)
    set(one_value_args VERSION NAME)
    set(multiargs TARGETS)
    cmake_parse_arguments(PARSED "" "${one_value_args}" "${multiargs}" ${ARGN})

    set(temdir ${CMAKE_CURRENT_BINARY_DIR}/wheel)

    set(PACKAGE_NAME ${PARSED_NAME})
    set(PACKAGE_VERSION ${PARSED_VERSION})
    set(OUTPUT_DIR ${CMAKE_INSTALL_PREFIX}/wheels)
    set(BINARY_DIR ${temdir}/${PARSED_NAME})

    foreach(target ${PARSED_TARGETS})
        list(APPEND PACKAGE_TARGETS "$<TARGET_FILE:${target}>")
    endforeach()
    list(JOIN PACKAGE_TARGETS " " PACKAGE_TARGETS)

    get_property(SAMPO_SYSTEM_DEPS_EXCLUDE GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_EXCLUDE)
    get_property(SAMPO_SYSTEM_DEPS_INCLUDE GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_INCLUDE)

    list(JOIN SAMPO_SYSTEM_DEPS_INCLUDE " " RUNTIME_DEPS_INCLUDE)
    list(JOIN SAMPO_SYSTEM_DEPS_EXCLUDE " " RUNTIME_DEPS_EXCLUDE)
    list(JOIN SAMPO_RUNTIME_DEPS_PATHS " " RUNTIME_DEPS_PATHS)

    configure_file(
        ${SAMPO_ROOT_DIR}/cmake/utils/resources/install_python_module_depth.cmake.in
        ${temdir}/install_python_module_depth.cmake
        @ONLY
    )
    configure_file(${SAMPO_ROOT_DIR}/cmake/utils/resources/setup.py.in ${temdir}/setup.py @ONLY)
    configure_file(${SAMPO_ROOT_DIR}/cmake/utils/resources/__init__.py.in ${BINARY_DIR}/__init__.py @ONLY)
    file(COPY ${SAMPO_ROOT_DIR}/cmake/utils/resources/glibc_utils.py DESTINATION ${temdir})
    file(READ ${temdir}/install_python_module_depth.cmake CODE_COMMAND)

    install(TARGETS ${PARSED_TARGETS} LIBRARY DESTINATION ${BINARY_DIR} COMPONENT Runtime)
    install(CODE ${CODE_COMMAND} COMPONENT Runtime)
    install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} setup.py WORKING_DIRECTORY ${temdir})" COMPONENT Runtime)

endfunction()

function(sampo_install_executable)
    set(multiargs TARGETS)
    cmake_parse_arguments(PARSED "" "" "${multiargs}" ${ARGN})
    install(TARGETS ${PARSED_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime)
endfunction()

function(sampo_install_runtime_deps)
    set(multiargs PRE_INCLUDE_REGEXES PRE_EXCLUDE_REGEXES POST_INCLUDE_REGEXES POST_EXCLUDE_REGEXES DIRECTORIES)
    cmake_parse_arguments(PARSED "" "" "${multiargs}" ${ARGN})

    get_targets_from_all(targets)

    set(targets_with_deps)
    set(supported_types MODULE_LIBRARY SHARED_LIBRARY EXECUTABLE)

    foreach(target ${targets})
        get_target_property(type ${target} TYPE)
        if (${type} IN_LIST supported_types)
            list(APPEND targets_with_deps ${target})
        endif()
    endforeach()

    get_property(SAMPO_SYSTEM_DEPS_EXCLUDE GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_EXCLUDE)
    get_property(SAMPO_SYSTEM_DEPS_INCLUDE GLOBAL PROPERTY SAMPO_SYSTEM_DEPS_INCLUDE)
    get_property(SAMPO_RUNTIME_DEPS_EXCLUDE GLOBAL PROPERTY SAMPO_RUNTIME_DEPS_EXCLUDE)
    get_property(SAMPO_RUNTIME_DEPS_INCLUDE GLOBAL PROPERTY SAMPO_RUNTIME_DEPS_INCLUDE)

    # install IMPORTED_RUNTIME_ARTIFACTS еще раз настраивает установку целей, поэтому используем компонент no_use
    install(IMPORTED_RUNTIME_ARTIFACTS ${targets_with_deps} RUNTIME_DEPENDENCY_SET rdset COMPONENT no_use)
    foreach(component IN_LIST Runtime Development)
        install(
            RUNTIME_DEPENDENCY_SET rdset 
                PRE_INCLUDE_REGEXES ${PARSED_PRE_INCLUDE_REGEXES} ${SAMPO_SYSTEM_DEPS_INCLUDE} ${SAMPO_RUNTIME_DEPS_INCLUDE}
                PRE_EXCLUDE_REGEXES ${PARSED_PRE_EXCLUDE_REGEXES} ${SAMPO_SYSTEM_DEPS_EXCLUDE} ${SAMPO_RUNTIME_DEPS_EXCLUDE}
                POST_INCLUDE_REGEXES ${PARSED_POST_INCLUDE_REGEXES} ${SAMPO_SYSTEM_DEPS_INCLUDE} ${SAMPO_RUNTIME_DEPS_INCLUDE}
                POST_EXCLUDE_REGEXES ${PARSED_POST_EXCLUDE_REGEXES} ${SAMPO_SYSTEM_DEPS_EXCLUDE} ${SAMPO_RUNTIME_DEPS_EXCLUDE}
                DIRECTORIES ${SAMPO_RUNTIME_DEPS_PATHS} ${PARSED_DIRECTORIES}
            COMPONENT ${component}
        )
    endforeach()
endfunction()
