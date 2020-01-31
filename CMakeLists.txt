cmake_minimum_required(VERSION 3.02)
project(ceres_bindings)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/ceres-solver/include)
include_directories(${PROJECT_SOURCE_DIR}/ceres-solver/internal)

include_directories(${PROJECT_SOURCE_DIR}/ceres-solver/config)

add_definitions(-DCERES_NO_THREADS)

add_subdirectory(pybind11)
pybind11_add_module(PyCeres python_bindings/python_module.cpp python_bindings/ceres_examples_module.cpp)

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

target_link_libraries(PyCeres PRIVATE /home/niko/programming/ceres_bindings/ceres-solver/build_release/lib/libceres.a -lglog -llapack -lcxsparse)

#add_executable(ceres_bindings main.cpp python_bindings/python_module.cpp)
