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
  │   ├── ...
  │   └── AddToCeres.cmake - file to include in Ceres CMakeLists.txt
```

Open up your **ceres-solver/CMakeLists.txt** and add the following to the end
of the file.

```
include(ceres_python_bindings/AddToCeres.cmake)
```

Build Ceres as you would normally.

## How to 

Assuming you built the library correctly you should now have a file called **PyCeres.so**. 
It probably more likely looks something like this **PyCeres.cpython-36m-x86_64-linux-gnu.so**.
Mark down the location of this file. This location is what you have to add to
python **sys.path** in order to use the library. An example of how to do this can
be seen below.

```python
pyceres_location="..."
import sys
sys.path.insert(0, pyceres_location)
```

After this you can now run 
```python
import PyCeres
```

to utilize the library.

You should peruse some of the examples listed below. It works almost exactly
like Ceres in C++. The only care you have to take is that the parameters you 
pass to the AddResidualBlock() function is a numpy array.

### Basic HelloWorld

Code for this example can be found in **python_tests/ceres_hello_world_example.py**

This example is the same as the hello world example from Ceres. 

```python
pyceres_location="../../build/lib" # assuming library was built along with ceres
# and cmake directory is called build
import sys
sys.path.insert(0, pyceres_location)

import PyCeres # Import the Python Bindings
import numpy as np

# The variable to solve for with its initial value.
initial_x=5.0
x=np.array([initial_x]) # Requires the variable to be in a numpy array

# Here we create the problem as in normal Ceres
problem=PyCeres.Problem()

# Creates the CostFunction. This example uses a C++ wrapped function which 
# returns the Autodiffed cost function used in the C++ example
cost_function=PyCeres.CreateHelloWorldCostFunction()

# Add the costfunction and the parameter numpy array to the problem
problem.AddResidualBlock(cost_function,None,x) 

# Setup the solver options as in normal ceres
options=PyCeres.SolverOptions()
# Ceres enums live in PyCeres and require the enum Type
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout=True
summary=PyCeres.Summary()
# Solve as you would normally
PyCeres.Solve(options,problem,summary)
print(summary.BriefReport() + " \n")
print( "x : " + str(initial_x) + " -> " + str(x) + "\n")
```


### CostFunction in Python

This library allows you to create your own custom CostFunction in Python to be
used with the Ceres Solver.

An custom CostFunction in Python can be seen here.
```python
# function f(x) = 10 - x.
# Comes from ceres/examples/helloworld_analytic_diff.cc
class QuadraticCostFunction(PyCeres.CostFunction):
    def __init__(self):
        # MUST BE CALLED. Initializes the Ceres::CostFunction class
        super().__init__()
        
        # MUST BE CALLED. Sets the size of the residuals and parameters
        self.set_num_residuals(1) 
        self.set_parameter_block_sizes([1])

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self,parameters, residuals, jacobians):
        x=parameters[0][0]

        residuals[0] = 10 - x

        if (jacobians!=None): # check for Null
            jacobians[0][0] = -1

        return True
```

Some things to be aware of for a custom CostFunction

* residuals is a numpy array
* parameters,jacobians are lists of numpy arrays ([arr1,arr2,...])
    * Indexing works similar to Ceres C++. parameters[i] is the ith parameter block
    * You must always index into the list first. Even if it only has 1 value. 
* You must call the base constructor with super.

### CostFunction defined in C++
It is possible to define your custom CostFunction in C++ and utilize it within 
the python framework. In order to do this we provide a file **python_bindings/custom_cpp_cost_functions.cpp**.
which provides a place to write your own wrapper code. The easiest way to do this
is create an initialization function that creates your custom CostFunction class
and returns a ceres::CostFunction* to it. That function should then be wrapped in
the *void add_custom_cost_functions(py::module& m)* function.

It should end up looking something like this.
```cpp

#include <CUSTOM_HEADER_FILES_WITH_COST_FUNCTION>

// Create a function that initiliazes your CostFunction and returns a ceres::CostFunction*

ceres::CostFunction* CreateCustomCostFunction(arg1,arg2,...){
    return new CustomCostFunction(arg1,arg2,...);
}

// In file custom_cpp_cost_function add the following line

void add_custom_cost_functions(py::module &m) {
    // ....
    m.def("CreateCustomCostFunction",&CreateCustomCostFunction);
}

```

We provide a basic example of this in **custom_cpp_cost_functions.cpp**. 
Note you are responsible for ensuring that all the dependencies and includes are
set correctly for your library.


## Warnings:

* Remember Ceres was designed with a C++ memory model. So you have to be careful
when using it from Python. The main problem is that Python does not really have
the concept of giving away ownership of memory. So it may try to delete something
that Ceres still believes is valid.
* Careful with wrapping AutodiffCostfunction. It takes over the memory of a cost
functor which can cause errors.


## TODOs
- [ ] The wrapper code that wraps the Evaluate pointers(residuals,parameters,..)
    needs a lot of improvement and optimization. We really need this to be a zero copy
operation.
- [ ] Wrap all the variables for Summary and other classes
- [ ] LocalParameterizations and Lossfunctions need to be properly wrapped
- [ ] Callbacks need to be wrapped
- [ ] Investigate how to wrap a basic python function for evaluate rather than 
go through the CostFunction( something like in the C api).
- [ ] Add docstrings for all the wrapped stuff
- [X] Add a place for users to define their CostFunctions in C++
- [ ] Evaluate speed of Python Cost Function vs C++
- [ ] Clean up AddResidualBlock() and set up the correct error checks
- [ ] Figure out how google log should work with Python
- [ ] Figure out if we can somehow use Jax numpy arrays




## Status
Custom Cost functions work