# Available at setup time due to pyproject.toml
import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import which

import ninja
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.2"


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        if platform.system() == "Windows" and which("vswhere") is not None:
            vs_path = subprocess.check_output(
                ["vswhere", "-latest", "-prerelease", "-property", "installationPath"]
            )
            if not vs_path:
                vs_path = subprocess.check_output(
                    ["vswhere", "-property", "installationPath"]
                )
            if vs_path:
                vs_path = Path(vs_path.decode().strip())
                old_cwd = Path.cwd()
                os.chdir(os.path.join(vs_path, "VC", "Auxiliary", "Build"))
                vs_environ = subprocess.check_output(
                    ["cmd.exe", "/c", "call vcvarsall.bat x64 && set"]
                )
                if vs_environ:
                    for env in vs_environ.decode().splitlines():
                        env = env.strip()
                        if "=" in env:
                            elements = env.split("=")
                            if len(elements) == 2:
                                os.environ[elements[0]] = elements[1]
                os.chdir(old_cwd)

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_PYTHON_EXT=on",
        ]
        build_args = []
        if platform.system() == "Windows":
            build_args += ["--config", "release"]
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
        cmake_args += [
            "-GNinja",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
        ]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


setup(
    name="cyy_torch_cpp_extension",
    version=__version__,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    ext_modules=[CMakeExtension("cyy_torch_cpp_extension")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.11",
)
