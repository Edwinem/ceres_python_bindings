""" Contains Ceres HelloWorld Example in Python

This file contains the Ceres HelloWorld Example except it uses Python Bindings.

"""
import sys
sys.path.insert(0, "../build") # location of the PyCeres lib. By default this assumes you build the library
# using the stand mkdir build , cd build, cmake ..


import PyCeres # Import the Python Bindings
import numpy as np

# The variable to solve for with its initial value.
initial_x=5.0
x=np.array([initial_x])

# Build the Problem
problem=PyCeres.Problem()

# Set up the only cost function (also known as residual). This uses a helper function written in C++ as Autodiff
# cant be used in Python. Aside from that the return value is the same as in the C++ example.
cost_function=PyCeres.CreateHelloWorldCostFunction()

problem.AddResidualBlock(cost_function,None,x)

options=PyCeres.SolverOptions()
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR # Ceres enums live in PyCeres and require the enum Type
options.minimizer_progress_to_stdout=True
summary=PyCeres.SolverSummary()
PyCeres.Solve(options,problem,summary)
print(summary.BriefReport() + " \n")
print( "x : " + str(initial_x) + " -> " + str(x) + "\n")


