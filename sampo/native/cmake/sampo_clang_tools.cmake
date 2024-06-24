#[==[
@defgroup module CMake
#]==]

#[==[
\ingroup module
\brief Задает выполнение статического анализа цели при ее сборке с помощью
<a href="https://clang.llvm.org/extra/clang-tidy/">clang-tidy</a>.

~~~
sampo_add_clang_tidy(target)
~~~

\param[in] target Цель для статического анализа

Задает выполнение статического анализа цели при ее сборке с помощью
<a href="https://clang.llvm.org/extra/clang-tidy/">clang-tidy</a> если указан флаг ENABLE_CLANG_TIDY (-DENABLE_CLANG_TIDY=ON).

\note Функция автоматически вызывается для всех целей, переданных в @ref sampo_common_targets_options. Отдельное использование не предполагается.

<b>See also:
@ref sampo_common_targets_options,
<a href="https://clang.llvm.org/extra/clang-tidy/">clang-tidy</a>
</b>
#]==]
function(sampo_add_clang_tidy target)
    if (NOT ENABLE_CLANG_TIDY)
        message("Not configuring clang-tidy")
        return()
    endif()

    message("Configuring clang-tidy")

    if (MSVC)
        # См. https://learn.microsoft.com/en-us/cpp/code-quality/clang-tidy?view=msvc-170
        set_target_properties(
            ${target} PROPERTIES 
            VS_GLOBAL_RunCodeAnalysis true
            VS_GLOBAL_EnableMicrosoftCodeAnalysis false
            VS_GLOBAL_EnableClangTidyCodeAnalysis true
            VS_GLOBAL_ClangTidyChecks "-* \"\"--config-file=${SAMPO_SOURCE_DIR}/.clang-tidy"
        )
    else()
        set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANGTIDY_EXECUTABLE};--config-file=${SAMPO_SOURCE_DIR}/.clang-tidy")
    endif()
endfunction()

#[==[
\ingroup module
\brief Задает форматирование кода для цели при ее сборке с помощью
<a href="https://clang.llvm.org/docs/ClangFormat.html">clang-format</a>.

~~~
sampo_add_clang_format(target)
~~~

\param[in] target Цель для форматирования кода

Задает форматирование кода для цели при ее сборке с помощью
<a href="https://clang.llvm.org/docs/ClangFormat.html">clang-format</a>. Поиск файлов с исходным годом выполняется рекурсивно по расширениям
h hpp hh c cc cxx cpp. Поиск начинается с директории с CMakeLists.txt для данной цели (
    <a href="https://cmake.org/cmake/help/latest/prop_tgt/SOURCE_DIR.html">SOURCE_DIR</a> 
)

\note Функция автоматически вызывается для всех целей, переданных в @ref sampo_common_targets_options. Отдельное использование не предполагается.

<b>See also:
@ref sampo_common_targets_options,
<a href="https://clang.llvm.org/docs/ClangFormat.html">clang-format</a>
</b>
#]==]
function(sampo_add_clang_format target)
    set(expr h hpp hh c cc cxx cpp)
    get_target_property(dir ${target} SOURCE_DIR)
    list(TRANSFORM expr PREPEND "${dir}/*.")
    file(GLOB_RECURSE SOURCE_FILES FOLLOW_SYMLINKS LIST_DIRECTORIES false ${expr})
    add_custom_command(TARGET ${target} PRE_BUILD COMMAND ${CLANGFORMAT_EXECUTABLE} -style=file:"${SAMPO_SOURCE_DIR}/.clang-format" -i ${SOURCE_FILES})
endfunction()