from distutils.command.build_ext import build_ext
from distutils.errors import DistutilsPlatformError, CCompilerError, DistutilsExecError
from distutils.extension import Extension

ext_modules = [
    Extension("sampo.native",
              include_dirs=["sampo/native"],
              sources=[
                       # "basic_types.h",
                       # "contractor.h",
                       # "dtime.h",
                       "sampo/native/dtime.cpp",
                       # "native.h",
                       "sampo/native/native.cpp",
                       # "pycodec.h",
                       "sampo/native/pycodec.cpp",
                       # "python_deserializer.h",
                       "sampo/native/python_deserializer.cpp",
                       # "workgraph.h"
              ],),
]


class BuildFailed(Exception):
    pass


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
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}}
    )
