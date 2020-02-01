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

""" Contains Ceres HelloWorld Example in Python

This file contains the Ceres HelloWorld Example except it uses Python Bindings.

"""
import os
pyceres_location="" # Folder where the PyCeres lib is created
if os.getenv('PYCERES_LOCATION'):
    pyceres_location=os.getenv('PYCERES_LOCATION')
else:
    pyceres_location="../build" # If the environment variable is not set then it will assume this directory. Only will
    # work if built standalone and through the normal mkdir build, cd build, cmake .. procedure

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
# cant be used in Python. Aside from that the return value is the same as in the C++ example.
cost_function=PyCeres.CreateHelloWorldCostFunction()

problem.AddResidualBlock(cost_function,None,x)

options=PyCeres.SolverOptions()
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR # Ceres enums live in PyCeres and require the enum Type
options.minimizer_progress_to_stdout=True
summary=PyCeres.Summary()
PyCeres.Solve(options,problem,summary)
print(summary.BriefReport() + " \n")
print( "x : " + str(initial_x) + " -> " + str(x) + "\n")


