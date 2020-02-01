# Ceres Python Wrapper

This project uses pybind11 to wrap Ceres with a python interface.

## Build Setup
There are two different ways to build this library. The easiest and recommended
way is to build it along with the Ceres library. The other way builds it 
standalone, and is more error prone.

### Recommended: Build Alongside Ceres

Clone the repository at https://github.com/Edwinem/ceres_python_bindings

Initialize and download the pybind11 submodule
```shell
git clone https://github.com/Edwinem/ceres_python_bindings
cd ceres_python_bindings
git submodule init
git submodule update
```

Copy and paste the **ceres_python_bindings** directory to your ceres-solver directory.
Your ceres directory should now look something like this.
  ```
  ceres-solver/
  │
  ├── CMakeLists.txt
  ├── include
  ├── ...
  │
  ├── ceres_python_bindings/ - your copied folder
  │   ├── pybind11 
  │   ├── python_bindings
      ├── ...
  │   └── AddToCeres.cmake - file to include in Ceres CMakeLists.txt
```

Open up your **ceres-solver/CMakeLists.txt** and add the following to the end
of the file.

```
include(ceres_python_bindings/AddToCeres.cmake)
```

Build Ceres as you would normally.

## How to use


## Status
Custom Cost functions work