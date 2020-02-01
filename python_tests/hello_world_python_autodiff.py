# Ceres Solver Python Bindings
# Copyright Nikolaus Mitchell. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: nikolausmitchell@gmail.com (Nikolaus Mitchell)

'''

Example of using a basic cost function with autodiff provided by Jax

'''

import os
pyceres_location="" # Folder where the PyCeres lib is created
if os.getenv('PYCERES_LOCATION'):
    pyceres_location=os.getenv('PYCERES_LOCATION')
else:
    pyceres_location="../build" # If the environment variable is not set then it will assume this directory. Only will
    # work if built standalone and through the normal mkdir build, cd build, cmake .. procedure

import sys
sys.path.insert(0, pyceres_location)

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