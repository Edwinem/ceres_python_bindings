"""
Solves a g2o pose graph dataset using Ceres

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

import PyCeres  # Import the Python Bindings
import numpy as np
from jax import grad


def residual_calc():
    return 10 - param_input

# function f(x) = 10 - x.
class PoseResidual(PyCeres.CostFunction):
    def __init__(self,dx,dy,dtheta):
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([1,1,1,1,1,1])
        self.dx=dx
        self.dy=dy
        self.dtheta=dtheta
        self.sqrt_info=np.identity(3)*0.05

    def Evaluate(self,parameters, residuals, jacobians):
        xa=parameters[0][0]
        ya=parameters[1][0]
        yaw_a=parameters[2][0]
        xb=parameters[3][0]
        yb=parameters[4][0]
        yaw_b=parameters[5][0]


        residuals[0] = residual_calc(x)

        if (jacobians!=None):
            jacobians[0][0] = grad(residual_calc)(x)

        return True



def RunHelloWorldAutoDiff():
    cost_function = HelloWorldAutoDiff()

    data = [0.5]
    np_data = np.array(data)

    print(np_data)

    problem = PyCeres.Problem()

    problem.AddResidualBlock(cost_function, None, np_data)
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)
    print(summary.BriefReport())
    print ("x : " + str(0.5) + " -> " + str(np_data[0]))

RunHelloWorldAutoDiff()