set(CONAN_NAME boost)
set(CONAN_VERSION 1.80.0)
set(CONAN_OPTIONS)
set(CONAN_SETTINGS)
set(CONAN_FINDPACKAGE_PREFIX)

set(USE_SHARED_LIBS True)

set(FINDPACKAGE_PACKAGE_NAME Boost)
set(FINDPACKAGE_COMPONENTS system filesystem)

if (${USE_SHARED_LIBS})
    set(FINDDEPENDENCY_PRELUDE "set(Boost_USE_STATIC_LIBS False)")
else()
    set(FINDDEPENDENCY_PRELUDE "set(Boost_USE_STATIC_LIBS True)")
endif()

include(${SAMPO_ROOT_DIR}/cmake/conan_package.cmake)
