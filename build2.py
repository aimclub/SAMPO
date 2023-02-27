# build2.py

from typing import Any, Dict

from setuptools_cpp import CMakeExtension, ExtensionBuilder

ext_modules = [
    # An extension with a custom <project_root>/src/ext2/CMakeLists.txt:
    CMakeExtension(f"native.ext", sourcedir="sampo/native"),
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        }
    )