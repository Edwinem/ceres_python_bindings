""" Contains Ceres Rosenbrock example in Python
"""

import numpy as np

import PyCeres  # Import the Python Bindings


# f(x,y) = (1-x)^2 + 100(y - x^2)^2;
class Rosenbrock(PyCeres.FirstOrderFunction):
    def __init__(self):
        super().__init__()

    def Evaluate(self, parameters, cost, gradient):
        x = parameters[0]
        y = parameters[1]

        cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
        if not (gradient is None):
            gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x
            gradient[1] = 200.0 * (y - x * x)
        return True

    def NumParameters(self):
        return 2


parameters = [-1.2, 1.0]

np_params = np.array(parameters)

options = PyCeres.GradientProblemOptions()
options.minimizer_progress_to_stdout = True

summary = PyCeres.GradientProblemSummary()
problem = PyCeres.GradientProblem(Rosenbrock())
PyCeres.Solve(options, problem, np_params, summary)

print(summary.FullReport() + "\n")
print("Initial x: " + str(-1.2) + " y: " + str(1.0))
print("Final   x: " + str(np_params[0]) + " y: " + str(np_params[1]))
