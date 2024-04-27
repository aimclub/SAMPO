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