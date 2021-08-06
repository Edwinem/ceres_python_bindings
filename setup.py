import sys

from cmake_build_extension import BuildExtension, CMakeExtension
from setuptools import setup
from pathlib import Path

setup(
    ext_modules=[
        CMakeExtension(
            name="CMakeProject",
            install_prefix="PyCeres",
            cmake_depends_on=["pybind11"],
            disable_editable=False,
            cmake_configure_options=[
                # This option points CMake to the right Python interpreter, and helps
                # the logic of FindPython3.cmake to find the active version
                f"-DPython3_ROOT_DIR={Path(sys.prefix)}"]
        )
    ],
    cmdclass=dict(build_ext=BuildExtension),
)