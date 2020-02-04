""" Contains Ceres Rosenbrock example in Python
"""
import os

pyceres_location = ""  # Folder where the PyCeres lib is created
if os.getenv('PYCERES_LOCATION'):
    pyceres_location = os.getenv('PYCERES_LOCATION')
else:
    pyceres_location = "../../build/lib"  # If the environment variable is not set
    # then it will assume this directory. Only will work if built with Ceres and
    # through the normal mkdir build, cd build, cmake .. procedure
import sys

sys.path.insert(0, pyceres_location)

import PyCeres  # Import the Python Bindings
import numpy as np

from jax import grad

# f(x,y) = (1-x)^2 + 100(y - x^2)^2;
def CalcCost(x, y):
    return (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)

class Rosenbrock(PyCeres.FirstOrderFunction):
    def __init__(self):
        super().__init__()

    def Evaluate(self,parameters, cost, gradient):
        x = parameters[0]
        y = parameters[1]
        cost[0] = CalcCost(x,y)
        if not(gradient is None):
            gradient[0],gradient[1] = grad(CalcCost,(0,1))(x,y)
        return True

    def NumParameters(self):
        return 2


parameters = [-1.2, 1.0]

np_params=np.array(parameters)

options=PyCeres.GradientProblemOptions()
options.minimizer_progress_to_stdout = True

summary=PyCeres.GradientProblemSummary()
problem=PyCeres.GradientProblem(Rosenbrock())
PyCeres.Solve(options, problem, np_params, summary)

print(summary.FullReport() + "\n")
print("Initial x: " + str(-1.2) + " y: " + str(1.0) )
print( "Final   x: " + str(np_params[0]) + " y: " + str(np_params[1]))
