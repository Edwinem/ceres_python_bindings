
include_directories(${PROJECT_SOURCE_DIR}/ceres_python_bindings/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/internal)

set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(${PROJECT_SOURCE_DIR}/ceres_python_bindings/pybind11)
pybind11_add_module(PyCeres ${PROJECT_SOURCE_DIR}/ceres_python_bindings/python_bindings/python_module.cpp
        ${PROJECT_SOURCE_DIR}/ceres_python_bindings/python_bindings/ceres_examples_module.cpp
        ${PROJECT_SOURCE_DIR}/ceres_python_bindings/python_bindings/custom_cpp_cost_functions.cpp)


target_link_libraries(PyCeres PRIVATE ceres)
