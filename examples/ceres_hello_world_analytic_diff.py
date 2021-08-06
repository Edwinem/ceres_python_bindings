import PyCeres  # Import the Python Bindings
import numpy as np
import pytest


# A CostFunction implementing analytically derivatives for the
# function f(x) = 10 - x.
# Comes from ceres/examples/helloworld_analytic_diff.cc
class QuadraticCostFunction(PyCeres.CostFunction):
    def __init__(self):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        x = parameters[0][0]

        residuals[0] = 10 - x

        if (jacobians != None):
            jacobians[0][0] = -1

        return True


def RunQuadraticFunction():
    cost_function = QuadraticCostFunction()

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
    print("x : " + str(0.5) + " -> " + str(np_data[0]))


RunQuadraticFunction()
