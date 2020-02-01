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

""" Contains Ceres Simple Bundle Adjustment in Python

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
import argparse

parser = argparse.ArgumentParser(description='Solves a Bundle Adjustment problem')
parser.add_argument('file', help='File from http://grail.cs.washington.edu/projects/bal')
args = parser.parse_args()

if len(sys.argv)==1:
    sys.exit("No file provided")

file=args.file


bal_problem= PyCeres.BALProblem()

bal_problem.LoadFile(file)

problem=PyCeres.Problem()

observations=bal_problem.observations()
cameras=bal_problem.cameras()
points=bal_problem.points()

numpy_points=np.array(points)
numpy_points=np.reshape(numpy_points,(-1,3))
numpy_cameras=np.array(cameras)
numpy_cameras=np.reshape(numpy_cameras,(-1,9))
print(numpy_points[0])


for i in range(0,bal_problem.num_observations()):
    cost_function=PyCeres.CreateSnavelyCostFunction(observations[2*i+0],observations[2*i+1])
    cam_index=bal_problem.camera_index(i)
    point_index=bal_problem.point_index(i)
    problem.AddResidualBlock(cost_function,None,numpy_cameras[cam_index],numpy_points[point_index])


options=PyCeres.SolverOptions()
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_SCHUR
options.minimizer_progress_to_stdout=True

summary=PyCeres.Summary()
PyCeres.Solve(options,problem,summary)
print(summary.FullReport())

# Compare with CPP version

PyCeres.SolveBALProblemWithCPP(bal_problem)
cpp_points=bal_problem.points()
cpp_points=np.array(cpp_points)
cpp_points=np.reshape(cpp_points,(-1,3))
print(" For point 1 Python has a value of " + str(numpy_points[0]) + " \n")
print(" Cpp solved for point 1 a value of " + str(cpp_points[0]))
