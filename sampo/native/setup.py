import os
import pathlib
from distutils.core import setup, Extension
from distutils.errors import DistutilsPlatformError, CCompilerError, DistutilsExecError

import numpy
from setuptools.command.build_ext import build_ext as build_ext_orig

ext_modules = [
    Extension("native",
              include_dirs=[numpy.get_include(), "timeEstimatorLibrary", "timeEstimatorLibrary/Windows"],
              sources=[
                       # "basic_types.h",
                       # "contractor.h",
                       # "dtime.h",
                       "dtime.cpp",
                       # "native.h",
                       "native.cpp",
                       # "pycodec.h",
                       # "python_deserializer.h",
                       "python_deserializer.cpp",
                       "chromosome_evaluator.cpp",
                       # "timeEstimatorLibrary/Windows/DLLoader.h",
                       # "workgraph.h"
              ],
              extra_compile_args=['-fopenmp', '/openmp'],
              extra_link_args=['-lgomp']),
]


class BuildFailed(Exception):
    pass

class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Release'  # 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DCMAKE_C_COMPILER=gcc',
            '-DCMAKE_CXX_COMPILER=g++'
        ]

        # example of build args
        build_args = [
            '--config', config
            # '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '-DCMAKE_CXX_COMPILER=g++', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": [CMakeExtension(".")], "cmdclass": {"build_ext": build_ext}}
    )

setup(
    name='native',
    version='0.0.1',
    author='user',
    author_email='user@user.ru',
    description='C extension',
    ext_modules=ext_modules,
)
