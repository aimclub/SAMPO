import os
import sys
import ctypes

from typing import Optional


def glibc_version_string() -> Optional[str]:
    "Returns glibc version string, or None if not using glibc."
    return glibc_version_string_confstr() or glibc_version_string_ctypes()


def glibc_version_string_confstr() -> Optional[str]:
    "Primary implementation of glibc_version_string using os.confstr."
    # os.confstr is quite a bit faster than ctypes.DLL. It's also less likely
    # to be broken or missing. This strategy is used in the standard library
    # platform module:
    # https://github.com/python/cpython/blob/fcf1d003bf4f0100c9d0921ff3d70e1127ca1b71/Lib/platform.py#L175-L183
    if sys.platform == "win32":
        return None
    try:
        gnu_libc_version = os.confstr("CS_GNU_LIBC_VERSION")
        if gnu_libc_version is None:
            return ''
        # os.confstr("CS_GNU_LIBC_VERSION") returns a string like "glibc 2.17":
        _, version = gnu_libc_version.split()
    except (AttributeError, OSError, ValueError):
        # os.confstr() or CS_GNU_LIBC_VERSION not available (or a bad value)...
        return None
    return version


def glibc_version_string_ctypes() -> Optional[str]:
    "Fallback implementation of glibc_version_string using ctypes."

    try:
        import ctypes
    except ImportError:
        return None

    # ctypes.CDLL(None) internally calls dlopen(NULL), and as the dlopen
    # manpage says, "If filename is NULL, then the returned handle is for the
    # main program". This way we can let the linker do the work to figure out
    # which libc our process is actually using.
    try:
        process_namespace = ctypes.CDLL(None)
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except Exception:
        # Symbol doesn't exist -> therefore, we are not linked to
        # glibc.
        return ''

    # Call gnu_get_libc_version, which returns a string like "2.5"
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str = gnu_get_libc_version()
    # py2 / py3 compatibility:
    if not isinstance(version_str, str):
        version_str = version_str.decode("ascii")

    return version_str
