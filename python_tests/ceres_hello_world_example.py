""" Contains Ceres HelloWorld Example in Python

This file contains the Ceres HelloWorld Example except it uses Python Bindings.

"""
import os
pyceres_location="" # Folder where the PyCeres lib is created
if os.getenv('PYCERES_LOCATION'):
    pyceres_location=os.getenv('PYCERES_LOCATION')
else:
    pyceres_location="../../build/lib" # If the environment variable is not set
    # then it will assume this directory. Only will work if built with Ceres and
    # through the normal mkdir build, cd build, cmake .. procedure
import sys
sys.path.insert(0, pyceres_location)


import PyCeres # Import the Python Bindings
import numpy as np

# The variable to solve for with its initial value.
initial_x=5.0
x=np.array([initial_x])

# Build the Problem
problem=PyCeres.Problem()

# Set up the only cost function (also known as residual). This uses a helper function written in C++ as Autodiff
# cant be used in Python. It returns a CostFunction*
cost_function=PyCeres.CreateHelloWorldCostFunction()

problem.AddResidualBlock(cost_function,None,x)

options=PyCeres.SolverOptions()
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR # Ceres enums live in PyCeres and require the enum Type
options.minimizer_progress_to_stdout=True
summary=PyCeres.Summary()
PyCeres.Solve(options,problem,summary)
print(summary.BriefReport() + " \n")
print( "x : " + str(initial_x) + " -> " + str(x) + "\n")


