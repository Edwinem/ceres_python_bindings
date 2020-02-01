'''

Example of using a basic cost function with autodiff provided by Jax

'''

import sys

sys.path.insert(0, "../build")  # location of the PyCeres lib. By default this assumes you build the library
# using the stand mkdir build , cd build, cmake ..

import PyCeres  # Import the Python Bindings
import numpy as np
from jax import grad, jit, vmap


def residual_calc(param_input):
    return 10 - param_input

# function f(x) = 10 - x.
class HelloWorldAutoDiff(PyCeres.CostFunction):
    def __init__(self):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self,parameters, residuals, jacobians):
        x=parameters[0][0]

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