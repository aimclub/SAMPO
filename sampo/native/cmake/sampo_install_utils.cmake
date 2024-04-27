#[==[
@defgroup module CMake
#]==]

#[==[
\ingroup module
\brief Задает установку для библиотек с минимальным набором необходимых настроек

~~~
sampo_install_library(TARGETS [target...] [EXPORT [name]] [ARCH_INDEPENDENT])
~~~

\param[in] TARGETS [target...] Список библиотек для установки
\param[in] [EXPORT [name]] Опционально, сформировать <a href="https://cmake.org/cmake/help/latest/command/install.html#export">export-файл</a>
с указанным именем пакета.
\param[in] [ARCH_INDEPENDENT] Опционально, экспортировать пакет, как независимый от платформы.

Задает установку для библиотек с минимальным набором необходимых настроек, которые включают в себя:
- Компонентную установку (Runtime, Development)
- Формирование export-файла `<target-name>.targets.cmake.in`
- Формирование файлов `<target-name>-configVersion.cmake`
для возможности последующего использования библиотек с помощью `find_package(<target-name>)` в сторонних проектах.

`sampo_install_library` имеет  максимально приближеный к стандартной функции <a href="https://cmake.org/cmake/help/latest/command/install.html">install</a> интерфейс.

Компонены Runtime и Development устанавливаются со стандартными префиксами в зависимости от ОС.

\note `ARCH_INDEPENDENT` необходимо применять только для header-only библиотек.

<b>Пример использования:</b>
\code
sampo_install_library(TARGETS foo EXPORT foo ARCH_INDEPENDENT)
\endcode

<b>See also:
@ref sampo_install_executable,
<a href="https://cmake.org/cmake/help/latest/command/install.html">install</a>,
<a href="https://cmake.org/cmake/help/book/mastering-cmake/chapter/Install.html">Компоненты Runtime/Development</a>,
<a href="https://cmake.org/cmake/help/latest/command/find_package.html">find_package</a> 
</b>
#]==]
function(sampo_install_library)
    set(optional_args ARCH_INDEPENDENT)
    set(one_val_args EXPORT VERSION)
    set(multiargs TARGETS PUBLIC_HEADER)
    cmake_parse_arguments(PARSED "${optional_args}" "${one_val_args}" "${multiargs}" ${ARGN})

    set(library_component Runtime)
    if (WIN32)
        set(library_component Development)
    endif()

    install(
        TARGETS ${PARSED_TARGETS}
        EXPORT ${PARSED_EXPORT}
        LIBRARY
            COMPONENT ${library_component}
        ARCHIVE
            COMPONENT Development
        RUNTIME
            COMPONENT Runtime
    )

    if (WIN32)
        install(TARGETS ${PARSED_TARGETS} RUNTIME COMPONENT Development)
    else()
        install(TARGETS ${PARSED_TARGETS} LIBRARY COMPONENT Development RUNTIME COMPONENT Development)
    endif()

    _install_public_header(PARSED_TARGETS)

    if (PARSED_EXPORT)
        set(export_temdir ${CMAKE_CURRENT_BINARY_DIR}/cmake)
        install (TARGETS ${PARSED_TARGETS} EXPORT ${PARSED_EXPORT}-targets)
        _write_export_file(PARSED_TARGETS ${export_temdir}/${PARSED_EXPORT}.cmake.in)

        configure_package_config_file(
            ${export_temdir}/${PARSED_EXPORT}.cmake.in ${export_temdir}/${PARSED_EXPORT}-config.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
            NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO
        )

        if (PARSED_ARCH_INDEPENDENT)
            write_basic_package_version_file(
                ${export_temdir}/${PARSED_EXPORT}-configVersion.cmake
                VERSION ${PARSED_VERSION} COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT
            )
        else()
            write_basic_package_version_file(${export_temdir}/${PARSED_EXPORT}-configVersion.cmake VERSION ${PARSED_VERSION} COMPATIBILITY AnyNewerVersion)
        endif()

        install(EXPORT ${PARSED_EXPORT}-targets NAMESPACE sampo:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake COMPONENT Development)
        export(TARGETS ${PARSED_TARGETS} NAMESPACE sampo:: FILE ${PARSED_EXPORT}-targets.cmake)

        install (
            FILES
                ${export_temdir}/${PARSED_EXPORT}-config.cmake
                ${export_temdir}/${PARSED_EXPORT}-configVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
            COMPONENT Development
        )
    endif()

    # Добавляем выходы всех целей в исключения для поиска зависимостей
    foreach(target ${PARSED_TARGETS})
        get_target_property(pref ${target} PREFIX)
        get_target_property(name ${target} OUTPUT_NAME)
        get_target_property(suff ${target} SUFFIX)

        if (NOT pref)
            set(pref "")
        endif()
        if (NOT name)
            get_target_property(name ${target} NAME)
        endif()
        if (NOT suff)
            set(suff "")
        endif()

        set(binary_name "${pref}${name}${suff}")
 
        set_property(GLOBAL APPEND PROPERTY SAMPO_RUNTIME_DEPS_EXCLUDE "${binary_name}.*")
    endforeach()
    # Обязательно повышаем область видимости
endfunction()

#[==[
\ingroup module
\brief Задает установку Python модуля

~~~
sampo_install_python_module(TARGETS [target...] VERSION version NAME package_name)
~~~

\param[in] TARGETS [target...] Цели для установки
\param[in] VERSION version Версия Python модуля
\param[in] NAME package_name Имя Python пакета.

Задает Python модуля в виде собранного wheel пакета. Сборка включает в себя:
- Формирование setup.py файла
- Формирование __init__.py файла
- Копирование Runtime зависимостей в корень будущего пакета
- Сборку wheel пакета
- Установку пакета в `${CMAKE_INSTALL_PREFIX}/wheels`

Установка производится для компонента Runtime.

<b>Пример использования:</b>
\code
sampo_install_python_module(TARGET foo VERSION foo 0.0.1)
\endcode
#]==]
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

#[==[
\ingroup module
\brief Задает установку для исполняемых файлов с минимальным набором необходимых настроек

~~~
sampo_install_executable(TARGETS [target...])
~~~

\param[in] TARGETS [target...] Список библиотек для установки

Задает установку для исполняемых файлов с минимальным набором необходимых настроек, которые включают в себя:
- Компонентную установку (Runtime)

`sampo_install_executable` имеет  максимально приближеный к стандартной функции <a href="https://cmake.org/cmake/help/latest/command/install.html">install</a> интерфейс.

<b>Пример использования:</b>
\code
sampo_install_executable(TARGETS foo)
\endcode

<b>See also:
@ref sampo_install_library,
<a href="https://cmake.org/cmake/help/latest/command/install.html">install</a>,
<a href="https://cmake.org/cmake/help/book/mastering-cmake/chapter/Install.html">Компоненты Runtime/Development</a>,
<a href="https://cmake.org/cmake/help/latest/command/find_package.html">find_package</a> 
</b>
#]==]
function(sampo_install_executable)
    set(multiargs TARGETS)
    cmake_parse_arguments(PARSED "" "" "${multiargs}" ${ARGN})
    install(TARGETS ${PARSED_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime)
endfunction()

#[==[
\ingroup module
\brief Задает установку зависимостей для цели, если зависимости собираются в проекте

~~~
sampo_install_internal_deps(TARGETS [target...])
~~~

\param[in] TARGETS [target...] Список целей для установки зависимостей

Задает установку зависимостей для цели, если зависимости собираются в проекте. Функция работает следующим образом:
1. получает все линкуемые к `target` зависимости
2. выбирает из зависимостей те, у которых свойство <a href="https://cmake.org/cmake/help/latest/prop_tgt/IMPORTED.html">IMPORTED</a> не задано
3. для выбранных зависимостей вызывается импорт скрипта `cmake_install.cmake`.
Это позволяет выполнять установку всех внутренних зависимостей цели в случае standalone-сборки.

<b>Пример использования:</b>
\code
if (${${CUR_PROJ}_OUTSIDE_OF_CORE})
    sampo_install_internal_deps(TARGETS ${CUR_PROJ})
endif()
\endcode

\note @ref sampo_install_internal_deps должна вызываться только в случае standalone-сборки и только для standalone-целей.

<b>See also:
@ref sampo_install_library,
<a href="https://cmake.org/cmake/help/latest/command/install.html">install</a>
</b>
#]==]
function(sampo_install_internal_deps)
    if (NOT SAMPO_STANDALONE_CONFIGURE)
        return()
    endif()

    set(multiargs TARGETS)
    cmake_parse_arguments(PARSED "" "" "${multiargs}" ${ARGN})

    foreach(target ${PARSED_TARGETS})
        get_target_property(ilink_deps ${target} INTERFACE_LINK_LIBRARIES)

        if (NOT ilink_deps)
            continue()
        endif()

        foreach(dependency ${ilink_deps})
            get_target_property(is_imported ${dependency} IMPORTED)
            get_target_property(directory ${dependency} SOURCE_DIR)
            file(RELATIVE_PATH rpath ${SAMPO_SOURCE_DIR} ${directory})

            if (${is_imported})
                continue()
            endif()
            
            install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${rpath}/cmake_install.cmake" ALL_COMPONENTS)
        endforeach()
    endforeach()
endfunction()

#[==[
\ingroup module
\brief Выполняет поиск и копирование runtime зависимостей для целей из `all` (`ALL_BUILD`)

~~~
sampo_install_runtime_deps(
    PRE_INCLUDE_REGEXES [regex...]
    PRE_EXCLUDE_REGEXES [regex...]
    POST_INCLUDE_REGEXES [regex...]
    POST_EXCLUDE_REGEXES [regex...]
    DIRECTORIES [directory...]
)
~~~

\param[in] PRE_INCLUDE_REGEXES [regex...] Список регулярных выражений для влючения зависимостей перед их разрешением
\param[in] PRE_EXCLUDE_REGEXES [regex...] Список регулярных выражений для исключения зависимостей перед их разрешением
\param[in] POST_INCLUDE_REGEXES [regex...] Список регулярных выражений для влючения зависимостей после их разрешения
\param[in] POST_EXCLUDE_REGEXES [regex...] Список регулярных выражений для исключения зависимостей после их разрешения
\param[in] DIRECTORIES [directory...] Дополнительные каталоги поиска зависимостей

Выполняет поиск и копирование runtime зависимостей для целей, при этом дополнительно:
- Директории из пемеренной `SAMPO_RUNTIME_DEPS_PATHS` включаются в поиск
- Автоматически игнорируются зависимости вида `api-ms-`, `ext-ms-` для ОС Windows
- Автоматически игнорируются зависимости входящие в пакет
<a href="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170">Microsoft Visual C++ Redistributable</a> для ОС Windows
- Автоматически игнорируются зависимости, которые собираются в проекте (они либо будут установлены через @ref sampo_install_library, либо через @ref sampo_install_internal_deps)

`sampo_install_runtime_deps` имеет  максимально приближеный к стандартной функции
<a href="https://cmake.org/cmake/help/latest/command/install.html#runtime-dependency-set">install</a> интерфейс.

`sampo_install_runtime_deps` <b>обязательно необходимо использовать для shared-библиотек и исполняемых целей.</b>

<b>Пример использования:</b>
\code
# Пример показывает вызов функции, однако применять ее в таком виде не рекомендуется!
sampo_install_runtime_deps(
    PRE_INCLUDE_REGEXES "tbb*" "libtbb*" PRE_EXCLUDE_REGEXES "*threads*"
    DIRECTORIES "$ENV{SYSTEM32}" "/usr/lib"
)
\endcode

\note - Функция `sampo_install_runtime_deps` должна быть вызвана ровно один раз в процессе конфигурации
\note - При добавление в проект third party зависимостей <b>необходимо вносить пути к их runtime компонентам в переменную `SAMPO_RUNTIME_DEPS_PATHS`</b>,
тогда `sampo_install_runtime_deps` будет находить их автоматически.
\note - Старайтесь избегать использования аргумента `DIRECTORIES` в `sampo_install_runtime_deps`
\note - Наилучшая ситуация - вызов `sampo_install_runtime_deps(TARGETS [target...])` без дополнительных параметров

<b>See also:
<a href="https://cmake.org/cmake/help/latest/command/install.html">install</a>,
<a href="https://cmake.org/cmake/help/latest/command/install.html#runtime-dependency-set">Installing Runtime Dependencies</a>
</b>
#]==]
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

macro(_install_public_header targets)
    foreach(target ${${targets}})
        get_target_property(count ${target} SAMPO_PUBLIC_HEADER_COUNT)
        if (NOT count)
            continue()
        endif()

        math(EXPR count "${count}-1")       
        foreach(i RANGE ${count})
            _install_public_header_impl(${target} ${i})
        endforeach()
    endforeach()
endmacro()

macro(_install_public_header_impl target header_index)
    get_target_property(headers ${target} SAMPO_PUBLIC_HEADER_${header_index})
    get_target_property(include_dirs ${target} INCLUDE_DIRECTORIES)
    get_target_property(iface_include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(dest ${target} SAMPO_PUBLIC_HEADER_${header_index}_DESTINATION)

    # Если DESTINATION не был передан в sampo_target_public_header, то пытаемся использовать
    # $<INSTALL_INTERFACE:path>. Если нет и его, то мы не знаем куда устанавливать и выбрасываем ошибку.
    if (NOT dest)
        string(REGEX MATCH "<INSTALL_INTERFACE:(.*)>" _ ${iface_include_dirs})
        if (NOT CMAKE_MATCH_1)
            message(
                FATAL_ERROR
                "PUBLIC_HEADER destination for target ${target} not found. "
                "You have to set $<INSTALL_INTERFACE> or provide DESTINATION in sampo_target_public_header."
            )
        else()
            string(REPLACE "$<INSTALL_PREFIX>" "${CMAKE_INSTALL_PREFIX}" dest ${CMAKE_MATCH_1})
        endif()
    endif()

    # Каталоги для поиска относительного пути берутся из двух вариантов:
    # INCLUDE_DIRECTORIES и INTERFACE_INCLUDE_DIRECTORIES (для INTERFACE-библиотеки).
    # Эти переменные могут содержать generator-expression, в этом случае вытаскивем реальные пути из $<BUILD_INTERFACE:path>. 
    set(search_dirs "")
    foreach(dir ${include_dirs} ${iface_include_dirs})
        if (NOT dir)
            continue()
        endif()

        # Если dir содержить $<BUILD_INTERFACE:path>, то парсим
        if (${dir} MATCHES "<BUILD_INTERFACE:")
            string(REGEX MATCH "<BUILD_INTERFACE:(.+)>" _ ${dir})
            list(APPEND search_dirs ${CMAKE_MATCH_1})
        # Иначе проверяем не является ли это до сих пор generator-expression вида <INSTALL_INTERFACE или <TARGET_PROPERTY
        elseif (NOT ${dir} MATCHES "<INSTALL_INTERFACE:" AND NOT ${dir} MATCHES "<TARGET_PROPERTY:")
            list(APPEND search_dirs ${dir})
        endif()
    endforeach()

    # Для каждого хедера ищем его относительный путь относительно search_dirs, этот относительный путь будет постфиксом к CMAKE_INSTALL_PREFIX.
    # Это необходимо для того, чтобы сохранить структуру экспортируемых заголовков точно такой же, как при сборке.
    foreach(header ${headers})
        foreach(dir ${search_dirs})
            file(RELATIVE_PATH rpath ${dir} "${CMAKE_CURRENT_LIST_DIR}/${header}")
              if (EXISTS "${dir}/${rpath}")
                get_filename_component(folder ${rpath} DIRECTORY)
                install(FILES ${header} DESTINATION "${dest}/${folder}" COMPONENT Development)
            endif()
        endforeach()
    endforeach()
endmacro()

macro(_write_export_file targets filename)
    set(find_dependency_str "")
    set(package_init_str "@PACKAGE_INIT@")

    foreach(target ${${targets}})
        get_target_property(ilink_deps ${target} INTERFACE_LINK_LIBRARIES)

        if (NOT ilink_deps)
            continue()
        endif()

        foreach(dependency ${ilink_deps})
            get_target_property(is_imported ${dependency} IMPORTED)

            if (${is_imported})
                get_target_property(pname ${dependency} SAMPO_FINDPACKAGE_NAME)
                get_target_property(version ${dependency} SAMPO_FINDPACKAGE_VERSION)
                get_target_property(use_exact ${dependency} SAMPO_FINDPACKAGE_EXACT_VERSION)
                get_target_property(components ${dependency} SAMPO_FINDPACKAGE_COMPONENTS)
                get_target_property(prelude ${dependency} SAMPO_FINDPACKAGE_PRELUDE)

                if (NOT pname)
                    message(
                        FATAL_ERROR
                        "Could not create export file:\n"
                        "\tinformation about dependency ${dependency} not found.\n"
                        "\tCheck CMakeLists.txt for ${dependency} in third_party directory."
                    )
                endif()

                set(prelude_str "")
                set(version_str "")
                set(components_str "")

                if (prelude AND NOT prelude STREQUAL "")
                    set(prelude_str "${prelude}\n")
                endif()
                if (components AND NOT components STREQUAL "")
                    set(components_str "COMPONENTS ${components}")
                endif()
                if (version AND NOT version STREQUAL "")
                    set(version_str "${version}")
                endif()
                if (use_exact)
                    set(version_str "${version_str} EXACT")
                endif()

                set(find_dependency_str "${find_dependency_str}\n${prelude_str}")
                set(find_dependency_str "${find_dependency_str}\nfind_package(${pname} ${version_str} REQUIRED ${components_str})")
            else()
                set(find_dependency_str "${find_dependency_str}\nfind_dependency(${dependency})")
            endif()
        endforeach()
    endforeach()

    file(
        CONFIGURE OUTPUT ${filename}
        CONTENT 
"# This file is automaticaly generated for export the sampo::@PARSED_EXPORT@ library
# which should be passed to the target_link_libraries command.

@package_init_str@
@find_dependency_str@

if (NOT TARGET sampo::@PARSED_EXPORT@)
    include(\"\${CMAKE_CURRENT_LIST_DIR\}/@PARSED_EXPORT@-targets.cmake\")
endif()
"
        @ONLY
    )
endmacro()
