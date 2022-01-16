# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        os.makedirs(extdir, exist_ok=True)
        cmake_build_dir = os.getenv("cmake_build_dir")
        for f in Path(cmake_build_dir).rglob("*"):
            if (
                str(f).endswith(".so")
                or str(f).endswith(".dll")
                or str(f).endswith(".lib")
            ):
                shutil.copy(f, extdir)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="cyy_torch_cpp_extension",
    version="0.1",
    author="cyy",
    author_email="cyyever@outlook.com",
    description="Python binding for this C++ lib",
    url="https://github.com/cyyever/torch_cpp_extension",
    ext_modules=[CMakeExtension("cmake_example")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.6",
)
