#[==[
@defgroup module CMake
#]==]


#[==[
\ingroup module
\brief Выполняет поиск заголовочных файлов и файлов с исходным кодом

~~~
sampo_add_headers_and_sources(prefix headers_path sources_path)
~~~

\param[in] prefix Префикс для выходных переменных
\param[in] headers_path Путь к поиску заголовочных файлов
\param[in] sources_path Путь к поиску файлов с исходным кодом
\param[out] Переменные `${prefix}_headers`, `${prefix}_sources`, содержашие заголовочные файлы и файлы с исходным кодом соответственно

Выполняет поиск заголовочных файлов и файлов с исходным кодом. Поиск выполняется по следующим правилам:
- Дирректория `headers_path` для поиска заголовочных файлов 
- Дирректория `sources_path` для поиска заголовочных файлов с исходным кодом
- Директория `headers_path/include` для поиска заголовочных файлов
- Директория `sources_path/src` для поиска файлов с исходным кодом


<b>Примеры использования:</b>
\code
sampo_add_headers_and_sources(${CUR_PROJ} . .)
add_library(${CUR_PROJ} INTERFACE ${${CUR_PROJ}_headers})
\endcode
\code
sampo_add_headers_and_sources(${CUR_PROJ} . .)
add_library(${CUR_PROJ} SHARED ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})
\endcode
\code
sampo_add_headers_and_sources(${CUR_PROJ} .)
add_executable(${CUR_PROJ} ${${CUR_PROJ}_headers} ${${CUR_PROJ}_sources})
\endcode

\note В качестве префикса рекомендуется всегда использовать имя текущего компонента (`${CUR_PROJ}`)

<b>See also:
@ref sampo_add_headers_only
#]==]
function(sampo_add_headers_and_sources prefix headers_path sources_path)
    _search_sources(
        ${prefix}_headers
        ${headers_path}/*.hpp ${headers_path}/*.h
        ${headers_path}/include/*.hpp ${headers_path}/include/*.h  
    )
    _search_sources(
        ${prefix}_sources
        ${sources_path}/*.cpp ${sources_path}/*.c
        ${sources_path}/src/*.cpp ${sources_path}/src/*.c
    )

    set(${prefix}_headers ${${prefix}_headers} PARENT_SCOPE)
    set(${prefix}_sources ${${prefix}_sources} PARENT_SCOPE)
endfunction()

#[==[
\ingroup module
\brief Выполняет поиск заголовочных файлов

~~~
sampo_add_headers_only(prefix path)
~~~

\param[in] prefix Префикс для выходных переменных
\param[in] path Путь к поиску заголовочных файлов
\param[out] Переменная `${prefix}_headers`, содержашая заголовочные файлы и файлы

Выполняет поиск заголовочных файлов. Поиск выполняется по следующим правилам:
- Дирректория `path` для поиска заголовочных файлов
- Директория `path/include` для поиска заголовочных файлов


<b>Пример использования:</b>
\code
sampo_add_headers_only(${CUR_PROJ} .)
add_library(${CUR_PROJ} INTERFACE ${${CUR_PROJ}_headers})
\endcode

\note В качестве префикса рекомендуется всегда использовать имя текущего компонента (`${CUR_PROJ}`)

<b>See also:
@ref sampo_add_headers_and_sources
#]==]
function(sampo_add_headers_only prefix path)
    _search_sources(
        ${prefix}_headers
        ${path}/*.hpp ${path}/*.h
        ${path}/include/*.hpp ${path}/include/*.h
    )
    set(${prefix}_headers ${${prefix}_headers} PARENT_SCOPE)
endfunction()

macro(_search_sources list_name)
    file(GLOB __tmp RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${ARGN})
    list(APPEND ${list_name} ${__tmp})
endmacro()