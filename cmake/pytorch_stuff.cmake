# Placeholder for old Pytorch stuff

list(APPEND PYTORCH_FILES "")
option(WITH_PYTORCH "Enables PyTorch defined Cost Functions" OFF)

if(${WITH_PYTORCH})
    #PyTorch by default is build with old C++ ABI. So we use that option here
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/programming/python/env/lib/python3.6/site-packages/torch/share/cmake/Torch")
    find_package(Torch REQUIRED)
    include_directories(${TORCH_INCLUDE_DIRS})

    add_definitions("-DWITH_PYTORCH")
endif(${WITH_PYTORCH})
if(${WITH_PYTORCH})
    list(APPEND PYTORCH_FILES python_bindings/pytorch_cost_function.h
            python_bindings/pytorch_cost_function.cpp
            python_bindings/pytorch_module.cpp)
endif(${WITH_PYTORCH})
if(${WITH_PYTORCH})
    target_link_libraries(PyCeres PRIVATE "${TORCH_LIBRARIES}")
    add_executable(torchscript tests/pytorch_test.cpp)
    target_link_libraries(torchscript "${TORCH_LIBRARIES}")
    set_property(TARGET torchscript PROPERTY CXX_STANDARD 14)
endif(${WITH_PYTORCH})