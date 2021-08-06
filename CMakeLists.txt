cmake_minimum_required(VERSION 3.02)
project(PyCeres)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Ceres REQUIRED)
find_package(Glog REQUIRED)
find_package(LAPACK)

include_directories(${CERES_INCLUDE_DIR})

# Since we are linking to ceres then we have to access the compile flags used to build it.
add_definitions("-DCERES_IS_LINKED")

add_subdirectory(pybind11)
pybind11_add_module(PyCeres python_bindings/python_module.cpp
        python_bindings/ceres_examples_module.cpp
        python_bindings/custom_cpp_cost_functions.cpp
#        python_bindings/pytorch_cost_function.h
#        python_bindings/pytorch_cost_function.cpp
        )

find_package(Eigen3 3.3 CONFIG REQUIRED
        HINTS ${HOMEBREW_INCLUDE_DIR_HINTS})
if (EIGEN3_FOUND)
    message("-- Found Eigen version ${EIGEN3_VERSION_STRING}: ${EIGEN3_INCLUDE_DIRS}")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*)" AND
            EIGEN3_VERSION_STRING VERSION_LESS 3.3.4)
        # As per issue #289: https://github.com/ceres-solver/ceres-solver/issues/289
        # the bundle_adjustment_test will fail for Eigen < 3.3.4 on aarch64.
        message(FATAL_ERROR "-- Ceres requires Eigen version >= 3.3.4 on aarch64. "
                "Detected version of Eigen is: ${EIGEN3_VERSION_STRING}.")
    endif()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(PyCeres PRIVATE ${CERES_LIBRARY} ${GLOG_LIBRARY} ${LAPACK_LIBRARIES})

#############
## Install ##
#############

install(
        TARGETS PyCeres
        EXPORT PyCeresTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the exported targets
install(
        EXPORT PyCeresTargets
        FILE PyCeresTargets.cmake
        NAMESPACE PyCeres::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyCeres)


set(PyCeres_BUILDDIR "${CMAKE_BINARY_DIR}/PyCeres")
# For the python package we need an init file
file(
        GENERATE
        OUTPUT "${PyCeres_BUILDDIR}/__init__.py"
        CONTENT "from PyCeres.PyCeres import *\n")

# Install the __init__.py file
install(
        FILES "${PyCeres_BUILDDIR}/__init__.py"
        DESTINATION ${CMAKE_INSTALL_PREFIX})

#install(
#    TARGETS pybind11_bindings
#    COMPONENT bindings
#    LIBRARY DESTINATION ${MYMATH_INSTALL_PREFIX}
#    ARCHIVE DESTINATION ${MYMATH_INSTALL_PREFIX}
#    RUNTIME DESTINATION ${MYMATH_INSTALL_PREFIX})

