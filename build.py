# -*- coding: utf-8 -*-
import os
import pathlib
from distutils.core import setup, Extension
from distutils.errors import DistutilsPlatformError, CCompilerError, DistutilsExecError

import numpy
from setuptools.command.build_ext import build_ext as build_ext_orig

packages = \
['sampo',
 'sampo.generator',
 'sampo.generator.config',
 'sampo.generator.environment',
 'sampo.generator.pipeline',
 'sampo.generator.utils',
 'sampo.metrics',
 'sampo.metrics.resources_in_time',
 'sampo.native',
 'sampo.scheduler',
 'sampo.scheduler.genetic',
 'sampo.scheduler.heft',
 'sampo.scheduler.multi_agency',
 'sampo.scheduler.resource',
 'sampo.scheduler.timeline',
 'sampo.scheduler.topological',
 'sampo.scheduler.utils',
 'sampo.schemas',
 'sampo.structurator',
 'sampo.utilities',
 'sampo.utilities.generation',
 'sampo.utilities.sampler',
 'sampo.utilities.visualization']

package_data = \
{'': ['*']}

install_requires = \
['deap>=1.3.3,<1.4.0',
 'matplotlib>=3.6.2,<3.7.0',
 'numpy>=1.23.5,<1.24.0',
 'pandas>=1.5.2,<1.6.0',
 'pathos>=0.3.0,<0.3.1',
 'plotly>=5.11.0,<5.12.0',
 'pytest>=7.2.0,<7.3.0',
 'scipy>=1.9.3,<1.10.0',
 'seaborn>=0.12.1,<0.13.0',
 'sortedcontainers>=2.4.0,<2.5.0',
 'toposort>=1.7,<2.0']


ext_modules = [
    Extension("native",
              include_dirs=[numpy.get_include()],
              sources=[
                       # "basic_types.h",
                       # "contractor.h",
                       # "dtime.h",
                       "sampo/native/dtime.cpp",
                       # "native.h",
                       "sampo/native/native.cpp",
                       # "pycodec.h",
                       # "python_deserializer.h",
                       "sampo/native/python_deserializer.cpp",
                       "sampo/native/chromosome_evaluator.cpp",
                       # "workgraph.h"
              ],),
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
        config = 'Release' # 'Debug' if self.debug else 'Release'
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
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": ExtBuilder},
            "zip_safe": False,
        }
    )
    

if __name__ == "__main__":
    setup(
        name='sampo',
        version='0.1.1.115',
        description='Open-source framework for adaptive manufacturing processes scheduling',
        long_description='None',
        author='iAirLab',
        author_email='iairlab@yandex.ru',
        maintainer='None',
        maintainer_email='None',
        url='None',
        packages=packages,
        package_data=package_data,
        install_requires=install_requires,
        ext_modules=ext_modules,
        python_requires='>=3.10,<3.11'
    )
