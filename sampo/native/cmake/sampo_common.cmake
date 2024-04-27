#[==[
@defgroup module CMake
#]==]

#[==[
\ingroup module
\brief Устанавливает ALIAS префикс для переданных целей

~~~
sampo_add_targets_alias(prefix [target...])
~~~

\param[in] prefix Префикс
\param[in] [target...] Список целей

Устанавливает ALIAS префикс для переданных целей.

<b>Пример использования:</b>
\code
## В результате цели будут доступны как third_party::libp7 third_party::libftp
sampo_add_targets_alias(third_party libp7 libftp)
target_link_libraries(foo third_party::libp7 third_party::libftp)
\endcode
#]==]
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

#[==[
\ingroup module
\brief Устанавливает базовые настройки для целей

~~~
sampo_common_targets_options([target...])
~~~

\param[in] [target...] Список целей

Устанавливает базовые настройки для целей. Базовые настройки для каждой цели включают в себя:
- Параллельную сборку цели компилятором MSVC
- Кодировку файлов `UTF-8`
- `-fPIC` опцию для shared и static библиотек на ОС Linux
- Диррективу препроцессора `SAMPO_COMPONENT_NAME`, которая внутри цели позволяет использовать имя цели
- Форматирование кода при помощи clang-format
- Статический анализ кода при помощи clang-tidy, если установлен флаг ENABLE_CLANG_TIDY (-DENABLE_CLANG_TIDY=ON).

<b>See also:</b> <a href="https://cmake.org/cmake/help/latest/prop_tgt/POSITION_INDEPENDENT_CODE.html">-fPIC</a> 
#]==]
function(sampo_common_targets_options)
    foreach(target ${ARGN})
        set_target_properties(${target} PROPERTIES CXX_STANDARD ${SAMPO_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)


        get_target_property(target_type ${target} TYPE)
        if (${target_type} STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif()

        sampo_add_clang_tidy(${target})
        # sampo_add_clang_format(${target})
        
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:MSVC>:/utf-8>>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/utf-8>>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:MSVC>:/MP>>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/MP>>")
        target_link_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/ignore:4099>>")
        target_compile_options(${target} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fvisibility=hidden -Wno-psabi -Wno-strict-aliasing>)

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

#[==[
\ingroup module
\brief Добавляет публичные заголовки для цели

~~~
sampo_target_public_header(target [header...] [DESTINATION <dest>])
~~~

\param[in] target Целей
\param[in] [header...] Список заголовчных файлов
\param[in] [DESTINATION <dest>] Опционально, относительный путь установки.


Добавляет публичные заголовки для цели. Публичные заголовки, установленные через `sampo_target_public_header`,
используются в дальнейшем функцией `@ref sampo_install_library`.

Если опциональный параметр `[DESTINATION <dest>]` не задан, то для установки заголовков будет выбрат путь относительно
<a href="https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:INSTALL_INTERFACE">$<INSTALL_INTERFACE></a>.
В случае, если цель не содержит `$<INSTALL_INTERFACE>`, `@ref sampo_target_public_header` выдаст ошибку.

Макрос добавляет свойства цели `SAMPO_PUBLIC_HEADER_COUNT`, `SAMPO_PUBLIC_HEADER_${i}`, `SAMPO_PUBLIC_HEADER_${i}_DESTINATION`.
Макрос может быть вызван несколько раз, если нужно установить разные группы заговловков.


<b>Пример использования:</b>

Пусть имеется некоторая цель, у которой есть следующая структура заголовков:
\code
include/
|-- api/
|------ api.hpp
|-- core/
|------ core.hpp
\endcode

Для того, чтобы установить цель с ее заголовками с сохранением структуры каталогов, необходимо:

`CMakeLists.txt:`
\code
...
sampo_target_public_header(target include/api/api.hpp)
sampo_target_public_header(target include/core/core.hpp)
sampo_install_library(TARGETS target)
\endcode

После выполнения команды `install` каталог `CMAKE_INSTALL_PREFIX` будет содержать следующую файловую структуру:
\code
CMAKE_INSTALL_PREFIX/
|-- api/
|------ api.hpp
|-- core/
|------ core.hpp
\endcode
#]==]
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

#[==[
\ingroup module
\brief Добавляет unit-test

~~~
sampo_add_unit_test(name [sources...])
~~~

\param[in] name Имя теста
\param[in] [sources...] Список заголовчных файлов


Добавляет unit-test, линкует его с фреймворком тестирования, добавляет тест в глобальный список тестов для 
<a href="https://cmake.org/cmake/help/latest/manual/ctest.1.html">ctest</a>. `EXCLUDE_FROM_ALL` цели игнорируются.


<b>Пример использования:</b>

\code
sampo_add_unit_test(test_${CUR_PROJ} ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})
\endcode


<b>See also:</b>
@ref sampo_add_benchmark
<a href="https://cmake.org/cmake/help/latest/manual/ctest.1.html">ctest</a>
<a href="https://cmake.org/cmake/help/latest/module/GoogleTest.html">GoogleTest</a>.
#]==]
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

#[==[
\ingroup module
\brief Добавляет benchmark

~~~
sampo_add_benchmark(name [sources...])
~~~

\param[in] name Имя теста
\param[in] [sources...] Список заголовчных файлов


Добавляет benchmark, линкует его с фреймворком для бенчмарков, добавляет benchmark в глобальный список тестов для 
<a href="https://cmake.org/cmake/help/latest/manual/ctest.1.html">ctest</a>. `EXCLUDE_FROM_ALL` цели игнорируются.


<b>Пример использования:</b>

\code
sampo_add_benchmark(test_${CUR_PROJ} ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})
\endcode


<b>See also:</b>
@ref sampo_add_unit_test
<a href="https://cmake.org/cmake/help/latest/manual/ctest.1.html">ctest</a>
<a href="https://github.com/google/benchmark">Benchmark</a>.
#]==]
function(sampo_add_benchmark name sources)
    add_executable(${name} ${sources})
    sampo_common_targets_options(${name})
    target_link_libraries(${name} PRIVATE third_party::benchmark::benchmark)

    # Benchmark допускает конструкцию вида for (_ : state), что вызывает предупреждение unused-parameter,
    # поэтому отключаем для бенчмарков
    target_compile_options(${name} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unused-parameter>)

    # add_test добавляет тест даже в случае, если цель исключена из `all`,
    # поэтому приходится отдельно это проверять, чтобы не добавить в тесты цель, которая не будет собрана.
    is_excluded_from_all(${name} excluded)
    if (NOT ${excluded})
        add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
        _add_test_runtime_deps_path(${name})
    endif()
endfunction()

#[==[
\ingroup module
\brief Возвращает все цели, которые включены в `all` (`ALL_BUILD`) и видимы в IDE.

~~~
get_targets_from_all(output)
~~~

\param[out] output Список заголовчных файлов

Возвращает все цели, которые включены в `all` (`ALL_BUILD`) и видимы в IDE.
#]==]
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

#[==[
\ingroup module
\brief Показать все свойства цели

~~~
print_target_properties(target)
~~~

\param[in] target Цель


Функция для отладки системы сборки.
#]==]
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
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

# Добавляет пути поиска зависимостей для выполнения теста через ctest
function(_add_test_runtime_deps_path test)
    if (WIN32)
        set_property(TEST ${test} PROPERTY ENVIRONMENT "PATH=${SAMPO_RUNTIME_DEPS_PATHS};$ENV{PATH}")
    else()
        set_property(TEST ${test} PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${SAMPO_RUNTIME_DEPS_PATHS}:$ENV{PATH}")
    endif()
endfunction()
