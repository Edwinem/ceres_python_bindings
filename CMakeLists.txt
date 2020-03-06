cmake_minimum_required(VERSION 3.02)
project(ceres_python_bindings)


option(WITH_PYTORCH "Enables PyTorch defined Cost Functions" OFF)

if(${WITH_PYTORCH})
    #PyTorch by default is build with old C++ ABI. So we use that option here
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/programming/python/env/lib/python3.6/site-packages/torch/share/cmake/Torch")
    find_package(Torch REQUIRED)
    include_directories(${TORCH_INCLUDE_DIRS})

    add_definitions("-DWITH_PYTORCH")
endif(${WITH_PYTORCH})

set(CERES_FOLDER $ENV{HOME}/programming/libs/ceres-solver)

include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${CERES_FOLDER}/include)
include_directories(${CERES_FOLDER}/internal)

include_directories(${CERES_FOLDER}/config)

add_definitions(-DCERES_NO_THREADS)


list(APPEND PYTORCH_FILES "")
if(${WITH_PYTORCH})
    list(APPEND PYTORCH_FILES python_bindings/pytorch_cost_function.h
            python_bindings/pytorch_cost_function.cpp
            python_bindings/pytorch_module.cpp)
endif(${WITH_PYTORCH})

add_subdirectory(pybind11)
pybind11_add_module(PyCeres python_bindings/python_module.cpp
        python_bindings/ceres_examples_module.cpp
        python_bindings/custom_cpp_cost_functions.cpp
        python_bindings/pytorch_cost_function.h
        python_bindings/pytorch_cost_function.cpp
        ${PYTORCH_FILES})

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

target_link_libraries(PyCeres PRIVATE ${CERES_FOLDER}/build/lib/libceres.a ${CERES_FOLDER}/glog/build/libglog.a -llapack -lcxsparse)

if(${WITH_PYTORCH})
    target_link_libraries(PyCeres PRIVATE "${TORCH_LIBRARIES}")
    add_executable(torchscript tests/pytorch_test.cpp)
    target_link_libraries(torchscript "${TORCH_LIBRARIES}")
    set_property(TARGET torchscript PROPERTY CXX_STANDARD 14)
endif(${WITH_PYTORCH})
