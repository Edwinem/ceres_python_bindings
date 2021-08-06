**WARNING THIS IS CURRENTLY EXTREMELY EXPERIMENTAL**

In order to bypass the fundamental slowness of Python (due to GIL and other factors). This
library optionally provides the capability to utilize PyTorch's TorchScript.
This allows you to define a CostFunction in Python, but bypass having to touch
it when solving the Ceres::Problem.

Right now the only the standalone version of this bindings support it. Lots of
the paths are hardcoded. So you will have to change them to.

To enable this functionality you must do the following things.

- Enable the option in cmake by turning on _WITH_PYTORCH_
- Build Ceres and GLOG that you link to with _-D_GLIBCXX_USE_CXX11_ABI=0_
    - The default PyTorch libs that you download from pip and other package managers
      is built with the old C++ ABI.

Note this will break normal functionality as all Python instantiations now requires
a *import Torch* before you import *PyCeres*.

Currently the TorchScript is passed by serialized files. 