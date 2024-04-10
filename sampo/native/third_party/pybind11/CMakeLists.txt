#[=[
    Конфигурация пакета, add_subdirectory 
#]=]
#set(BUILD_TESTING False)
#set(BUILD_SHARED_LIBS False)
#add_subdirectory(eigen EXCLUDE_FROM_ALL)

add_library(pybind11 INTERFACE IMPORTED GLOBAL)
target_include_directories(pybind11 SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)

#[=[
    Дополнительная конфигурация пакета
#]=]


#[=[
    Опции поиска пакета и runtime зависимостей при экспорте целей, который зависят от него
#]=]
set_target_properties(pybind11 PROPERTIES SEISM_FINDPACKAGE_NAME "pybind11")
set_target_properties(pybind11 PROPERTIES SEISM_FINDPACKAGE_VERSION "3.4")
set_target_properties(pybind11 PROPERTIES SEISM_FINDPACKAGE_EXACT_VERSION False)