// Ceres Solver Python Bindings
// Copyright Nikolaus Mitchell. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: nikolausmitchell@gmail.com (Nikolaus Mitchell)



#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/normal_prior.h>

// Needed due to forward decls
#include <ceres/problem_impl.h>
#include <ceres/residual_block.h>
#include <ceres/parameter_block.h>

#include <iostream>
#include <string>

namespace py = pybind11;

// Used for overloaded functions in C++11
template<typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Forward decls for additionally modules
void add_pybinded_ceres_examples(py::module &m);
void add_custom_cost_functions(py::module &m);

ceres::Problem CreatePythonProblem() {
  ceres::Problem::Options o;
  o.local_parameterization_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  o.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  o.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  return ceres::Problem(o);
}

// Function to create Problem::Options with DO_NOT_TAKE_OWNERSHIP
// This is cause we want Python to manage our memory not Ceres
ceres::Problem::Options CreateNoOwnershipOption() {
  ceres::Problem::Options o;
  o.local_parameterization_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  o.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  o.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  return o;
}

class PyCostFunction : public ceres::CostFunction {
 public:
  /* Inherit the constructors */
  using ceres::CostFunction::CostFunction;
  using ceres::CostFunction::set_num_residuals;

  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const override {
    pybind11::gil_scoped_acquire gil;

    // Resize the vectors passed to python to the proper size. And set the
    // pointer values
    if (!cached_flag) {
      parameters_vec.reserve(this->parameter_block_sizes().size());
      jacobians_vec.reserve(this->parameter_block_sizes().size());
      residuals_wrap = py::array_t<double>(num_residuals(), residuals, dummy);
      for (size_t idx = 0; idx < parameter_block_sizes().size(); ++idx) {
        parameters_vec.emplace_back(py::array_t<double>(this->parameter_block_sizes()[idx],
                                                        parameters[idx],
                                                        dummy));
        jacobians_vec.emplace_back(py::array_t<double>(
            this->parameter_block_sizes()[idx] * num_residuals(),
            jacobians[idx],
            dummy));

      }
      cached_flag = true;

    }

    // Check if the pointers have change and if they have then change them
    auto info = residuals_wrap.request(true);
    if (info.ptr != residuals) {
      residuals_wrap = py::array_t<double>(num_residuals(), residuals, dummy);
    }
    info = parameters_vec[0].request(true);
    if (info.ptr != parameters) {
      for (size_t idx = 0; idx < parameters_vec.size(); ++idx) {
        parameters_vec[idx] =
            py::array_t<double>(this->parameter_block_sizes()[idx],
                                parameters[idx],
                                dummy);
      }
    }
    if (jacobians) {
      info = jacobians_vec[0].request(true);
      if (info.ptr != jacobians) {
        for (size_t idx = 0; idx < jacobians_vec.size(); ++idx) {
          jacobians_vec[idx] = py::array_t<double>(
              this->parameter_block_sizes()[idx] * num_residuals(),
              jacobians[idx],
              dummy);
        }
      }
    }

    pybind11::function overload =
        pybind11::get_overload(static_cast<const ceres::CostFunction *>(this),
                               "Evaluate");
    if (overload) {
      if (jacobians) {
        auto o = overload.operator()<pybind11::return_value_policy::reference>(
            parameters_vec,
            residuals_wrap,
            jacobians_vec);
        return pybind11::detail::cast_safe<bool>(std::move(o));
      } else {
        auto o = overload.operator()<pybind11::return_value_policy::reference>(
            parameters_vec,
            residuals_wrap,
            nullptr);
        return pybind11::detail::cast_safe<bool>(std::move(o));
      }
    }
    pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(
        Ceres::CostFunction) "::" "Evaluate \"");
  }

 private:
  // Vectors used to pass double pointers to python as pybind does not wrap
  // double pointers(**) like Ceres uses.
  // Mutable so they can be modified by the const function.
  mutable std::vector<py::array_t<double>> parameters_vec;
  mutable std::vector<py::array_t<double>> jacobians_vec;
  mutable bool cached_flag = false; // Flag used to determine if the vectors
  // need to be resized
  mutable py::array_t<double> residuals_wrap; // Buffer to contain the residuals
  // pointer
  mutable py::str dummy; // Dummy variable for pybind11 so it doesn't make a
  // copy

};

class PyLossFunction : public ceres::LossFunction {
 public:
  /* Inherit the constructors */
  using ceres::LossFunction::LossFunction;

  void Evaluate(double sq_norm, double out[3]) const override {

  }

};

class PyLocalParameterization : public ceres::LocalParameterization {
  /* Inherit the constructors */
  using ceres::LocalParameterization::LocalParameterization;

  bool Plus(const double *x,
            const double *delta,
            double *x_plus_delta) const {
    assert(false);
    return true;
  }
  bool ComputeJacobian(const double *x, double *jacobian) const {
    assert(false);
    return true;
  }

  bool MultiplyByJacobian(const double *x,
                          const int num_rows,
                          const double *global_matrix,
                          double *local_matrix) const {
    assert(false);
    return true;
  }

  // Size of x.
  int GlobalSize() const override {
    PYBIND11_OVERLOAD_PURE(
        int, /* Return type */
        ceres::LocalParameterization,      /* Parent class */
        GlobalSize,          /* Name of function in C++ (must match Python name) */
    );
  }

  // Size of delta.
  int LocalSize() const override {
    PYBIND11_OVERLOAD_PURE(
        int, /* Return type */
        ceres::LocalParameterization,      /* Parent class */
        LocalSize,          /* Name of function in C++ (must match Python name) */
    );
  }

};

class PyEvaluationCallBack : public ceres::EvaluationCallback {
 public:
  /* Inherit the constructors */
  using ceres::EvaluationCallback::EvaluationCallback;

  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) override {
    PYBIND11_OVERLOAD_PURE(
        void, /* Return type */
        ceres::EvaluationCallback,      /* Parent class */
        PrepareForEvaluation,          /* Name of function in C++ (must match Python name) */
        evaluate_jacobians, new_evaluation_point      /* Argument(s) */
    );
  }

};

class PyFirstOrderFunction : public ceres::FirstOrderFunction {
 public:
  /* Inherit the constructors */
  using ceres::FirstOrderFunction::FirstOrderFunction;

  int NumParameters() const override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
        pybind11::get_overload(static_cast<const ceres::FirstOrderFunction *>(this),
                               "NumParameters");
    if (overload) {
      auto o = overload();
      return pybind11::detail::cast_safe<int>(std::move(o));
    }

    pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(
        ceres::FirstOrderFunction) "::" "NumParameters \"");
  }

  bool Evaluate(const double *const parameters,
                double *cost,
                double *gradient) const override {
    pybind11::gil_scoped_acquire gil;
    if (!cached_flag) {
      parameters_wrap = py::array_t<double>(NumParameters(), parameters, dummy);
      gradient_wrap = py::array_t<double>(NumParameters(), gradient, dummy);
      cost_wrap = py::array_t<double>(1, cost, dummy);
      cached_flag = true;
    }

    // Check if the pointers have change and if they have then change them
    auto info = cost_wrap.request(true);
    if (info.ptr != cost) {
      cost_wrap = py::array_t<double>(1, cost, dummy);
    }
    info = parameters_wrap.request(true);
    if (info.ptr != parameters) {
      parameters_wrap = py::array_t<double>(NumParameters(), parameters, dummy);
    }
    if (gradient) {
      info = gradient_wrap.request(true);
      if (info.ptr != gradient) {
        gradient_wrap = py::array_t<double>(NumParameters(), gradient, dummy);
      }
    }
    pybind11::function overload =
        pybind11::get_overload(static_cast<const ceres::FirstOrderFunction *>(this),
                               "Evaluate");
    if (overload) {
      if (gradient) {
        auto o = overload.operator()<pybind11::return_value_policy::reference>(
            parameters_wrap,
            cost_wrap,
            gradient_wrap);
        return pybind11::detail::cast_safe<bool>(std::move(o));
      } else {
        auto o = overload.operator()<pybind11::return_value_policy::reference>(
            parameters_wrap,
            cost_wrap,
            nullptr);
        return pybind11::detail::cast_safe<bool>(std::move(o));
      }
    }
    pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(
        ceres::FirstOrderFunction) "::" "Evaluate \"");
  }

 private:
  // Numpy arrays to pass to python that wrap the pointers
  // Mutable so they can be modified by the const function.
  mutable py::array_t<double> parameters_wrap;
  mutable py::array_t<double> gradient_wrap;
  mutable bool cached_flag = false; // Flag used to determine if the vectors
  // need to be resized
  mutable py::array_t<double> cost_wrap; // Buffer to contain the cost ptr
  mutable py::str dummy; // Dummy variable for pybind11 so it doesn't make a
  // copy

};

class PyIterationCallback : public ceres::IterationCallback {
 public:
  /* Inherit the constructors */
  using ceres::IterationCallback::IterationCallback;

  ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
    PYBIND11_OVERLOAD_PURE(
        ceres::CallbackReturnType, /* Return type */
        ceres::IterationCallback,      /* Parent class */
        operator(),          /* Name of function in C++ (must match Python name) */
        summary      /* Argument(s) */
    );
  }

};

// Hacky Wrapper for ceres::FirstOrderFunction.
// Essentially the problem is that GradientProblem takes ownership of the
// passed in function. In order to stop a double delete from happening we
// instead use this class. It wraps the ceres::FirstOrderFunction* pointer.
// This function is then passed to GradientProblem. GradientProblem will then
// delete this class instead of ceres::FirstOrderFunction. Python is free to
// delete the FirstOrderFunction* without worrying about a double delete.
class FirstOrderFunctionWrapper : public ceres::FirstOrderFunction {
 public:
  explicit FirstOrderFunctionWrapper(FirstOrderFunction *real_function)
      : function_(real_function) {}
  bool Evaluate(const double *const parameters,
                double *cost, double *gradient) const override {
    return function_->Evaluate(parameters, cost, gradient);
  }
  int NumParameters() const override {
    return function_->NumParameters();
  }

 private:
  FirstOrderFunction *function_;
};

// Same as FirstOrderFunctionWrapper
class CostFunctionWrapper : public ceres::CostFunction {

  explicit CostFunctionWrapper(ceres::CostFunction *real_cost_function)
      : cost_function_(real_cost_function) {
    this->set_num_residuals(cost_function_->num_residuals());
    *(this->mutable_parameter_block_sizes()) =
        cost_function_->parameter_block_sizes();
  }

  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const override {
    return cost_function_->Evaluate(parameters, residuals, jacobians);
  }
 private:
  CostFunction *cost_function_;
};

// Parses a numpy array and extracts the pointer to the first element.
// Requires that the numpy array be either an array or a row/column vector
double *ParseNumpyData(py::array_t<double> &np_buf) {
  py::buffer_info info = np_buf.request();
  // This is essentially just all error checking. As it will always be the info
  // ptr
  if (info.ndim > 2) {
    std::string error_msg("Number of dimensions must be <=2. This function"
                          "only allows either an array or row/column vector(2D matrix) "
                              + std::to_string(info.ndim));
    throw std::runtime_error(
        error_msg);
  }
  if (info.ndim == 2) {
    // Row or Column Vector. Represents 1 parameter
    if (info.shape[0] == 1 || info.shape[1] == 1) {
    } else {
      std::string error_msg
          ("Matrix is not a row or column vector and instead has size "
               + std::to_string(info.shape[0]) + "x"
               + std::to_string(info.shape[1]));
      throw std::runtime_error(
          error_msg);
    }
  }
  return (double *) info.ptr;
}

PYBIND11_MODULE(PyCeres, m) {
  m.doc() = "Ceres wrappers"; // optional module docstring'

  py::enum_<ceres::Ownership>(m, "Ownership")
      .value("DO_NOT_TAKE_OWNERSHIP", ceres::Ownership::DO_NOT_TAKE_OWNERSHIP)
      .value("TAKE_OWNERSHIP", ceres::Ownership::TAKE_OWNERSHIP)
      .export_values();

  py::enum_<ceres::MinimizerType>(m, "MinimizerType")
      .value("LINE_SEARCH", ceres::MinimizerType::LINE_SEARCH)
      .value("TRUST_REGION", ceres::MinimizerType::TRUST_REGION);

  py::enum_<ceres::LineSearchType>(m, "LineSearchType")
      .value("ARMIJO", ceres::LineSearchType::ARMIJO)
      .value("WOLFE", ceres::LineSearchType::WOLFE);

  py::enum_<ceres::LineSearchDirectionType>(m, "LineSearchDirectionType")
      .value("BFGS", ceres::LineSearchDirectionType::BFGS)
      .value("LBFGS", ceres::LineSearchDirectionType::LBFGS)
      .value("NONLINEAR_CONJUGATE_GRADIENT",
             ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT)
      .value("STEEPEST_DESCENT",
             ceres::LineSearchDirectionType::STEEPEST_DESCENT);

  py::enum_<ceres::LineSearchInterpolationType>(m,
                                                "LineSearchInterpolationType")
      .value("BISECTION", ceres::LineSearchInterpolationType::BISECTION)
      .value("CUBIC", ceres::LineSearchInterpolationType::CUBIC)
      .value("QUADRATIC", ceres::LineSearchInterpolationType::QUADRATIC);

  py::enum_<ceres::NonlinearConjugateGradientType>(m,
                                                   "NonlinearConjugateGradientType")
      .value("FLETCHER_REEVES",
             ceres::NonlinearConjugateGradientType::FLETCHER_REEVES)
      .value("HESTENES_STIEFEL",
             ceres::NonlinearConjugateGradientType::HESTENES_STIEFEL)
      .value("POLAK_RIBIERE",
             ceres::NonlinearConjugateGradientType::POLAK_RIBIERE);

  py::enum_<ceres::LinearSolverType>(m, "LinearSolverType")
      .value("DENSE_NORMAL_CHOLESKY",
             ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY)
      .value("DENSE_QR", ceres::LinearSolverType::DENSE_QR)
      .value("SPARSE_NORMAL_CHOLESKY",
             ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY)
      .value("DENSE_SCHUR", ceres::LinearSolverType::DENSE_SCHUR)
      .value("SPARSE_SCHUR", ceres::LinearSolverType::SPARSE_SCHUR)
      .value("ITERATIVE_SCHUR", ceres::LinearSolverType::ITERATIVE_SCHUR)
      .value("CGNR", ceres::LinearSolverType::CGNR);

  py::enum_<ceres::DoglegType>(m, "DoglegType")
      .value("TRADITIONAL_DOGLEG",
             ceres::DoglegType::TRADITIONAL_DOGLEG)
      .value("SUBSPACE_DOGLEG", ceres::DoglegType::SUBSPACE_DOGLEG);

  py::enum_<ceres::TrustRegionStrategyType>(m, "TrustRegionStrategyType")
      .value("LEVENBERG_MARQUARDT",
             ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT)
      .value("DOGLEG", ceres::TrustRegionStrategyType::DOGLEG);

  py::enum_<ceres::PreconditionerType>(m, "PreconditionerType")
      .value("IDENTITY",
             ceres::PreconditionerType::IDENTITY)
      .value("JACOBI", ceres::PreconditionerType::JACOBI)
      .value("SCHUR_JACOBI", ceres::PreconditionerType::SCHUR_JACOBI)
      .value("CLUSTER_JACOBI", ceres::PreconditionerType::CLUSTER_JACOBI)
      .value("CLUSTER_TRIDIAGONAL",
             ceres::PreconditionerType::CLUSTER_TRIDIAGONAL)
      .value("SUBSET", ceres::PreconditionerType::SUBSET);

  py::enum_<ceres::VisibilityClusteringType>(m, "VisibilityClusteringType")
      .value("CANONICAL_VIEWS",
             ceres::VisibilityClusteringType::CANONICAL_VIEWS)
      .value("SINGLE_LINKAGE", ceres::VisibilityClusteringType::SINGLE_LINKAGE);

  py::enum_<ceres::DenseLinearAlgebraLibraryType>(m,
                                                  "DenseLinearAlgebraLibraryType")
      .value("EIGEN",
             ceres::DenseLinearAlgebraLibraryType::EIGEN)
      .value("LAPACK", ceres::DenseLinearAlgebraLibraryType::LAPACK);

  py::enum_<ceres::SparseLinearAlgebraLibraryType>(m,
                                                   "SparseLinearAlgebraLibraryType")
      .value("SUITE_SPARSE",
             ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE)
      .value("CX_SPARSE", ceres::SparseLinearAlgebraLibraryType::CX_SPARSE)
      .value("EIGEN_SPARSE",
             ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE)
      .value("ACCELERATE_SPARSE",
             ceres::SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE)
      .value("NO_SPARSE",
             ceres::SparseLinearAlgebraLibraryType::NO_SPARSE);

  py::enum_<ceres::LoggingType>(m, "LoggingType")
      .value("SILENT",
             ceres::LoggingType::SILENT)
      .value("PER_MINIMIZER_ITERATION",
             ceres::LoggingType::PER_MINIMIZER_ITERATION);

  py::enum_<ceres::CovarianceAlgorithmType>(m, "CovarianceAlgorithmType")
      .value("DENSE_SVD",
             ceres::CovarianceAlgorithmType::DENSE_SVD)
      .value("SPARSE_QR",
             ceres::CovarianceAlgorithmType::SPARSE_QR);

  py::enum_<ceres::CallbackReturnType>(m, "CallbackReturnType")
      .value("SOLVER_CONTINUE", ceres::CallbackReturnType::SOLVER_CONTINUE)
      .value("SOLVER_ABORT", ceres::CallbackReturnType::SOLVER_ABORT)
      .value("SOLVER_TERMINATE_SUCCESSFULLY",
             ceres::CallbackReturnType::SOLVER_TERMINATE_SUCCESSFULLY);

  py::enum_<ceres::DumpFormatType>(m, "DumpFormatType")
      .value("CONSOLE", ceres::DumpFormatType::CONSOLE)
      .value("TEXTFILE", ceres::DumpFormatType::TEXTFILE);

  using options=ceres::Problem::Options;
  py::class_<ceres::Problem::Options> option(m, "ProblemOptions");
  option.def(py::init(&CreateNoOwnershipOption)); // Ensures default is that
  // Python manages memory
  option.def_readwrite("cost_function_ownership",
                       &options::cost_function_ownership);
  option.def_readwrite("loss_function_ownership",
                       &options::loss_function_ownership);
  option.def_readwrite("local_parameterization_ownership",
                       &options::local_parameterization_ownership);
  option.def_readwrite("enable_fast_removal", &options::enable_fast_removal);
  option.def_readwrite("disable_all_safety_checks",
                       &options::disable_all_safety_checks);

  py::class_<ceres::internal::ParameterBlock>
      parameter_block(m, "ParameterBlock");
  parameter_block.def(py::init<double *, int, int>());

  py::class_<ceres::internal::ResidualBlock> residual_block(m, "ResidualBlock");
  residual_block.def(py::init<const ceres::CostFunction *,
                              const ceres::LossFunction *,
                              const std::vector<ceres::internal::ParameterBlock *> &,
                              int>());
  residual_block.def("cost_function",
                     &ceres::internal::ResidualBlock::cost_function,
                     py::return_value_policy::reference);
  residual_block.def("loss_function",
                     &ceres::internal::ResidualBlock::loss_function,
                     py::return_value_policy::reference);
  residual_block.def("NumParameterBlocks",
                     &ceres::internal::ResidualBlock::NumParameterBlocks);
  residual_block.def("NumResiduals",
                     &ceres::internal::ResidualBlock::NumResiduals);
  residual_block.def("index", &ceres::internal::ResidualBlock::index);
  residual_block.def("ToString", &ceres::internal::ResidualBlock::ToString);

  py::class_<ceres::Problem> problem(m, "Problem");
  problem.def(py::init(&CreatePythonProblem));
  problem.def(py::init<ceres::Problem::Options>());
  problem.def("NumParameterBlocks", &ceres::Problem::NumParameterBlocks);
  problem.def("NumParameters", &ceres::Problem::NumParameters);
  problem.def("NumResidualBlocks", &ceres::Problem::NumResidualBlocks);
  problem.def("NumResiduals", &ceres::Problem::NumResiduals);
  problem.def("ParameterBlockSize", &ceres::Problem::ParameterBlockSize);
  //problem.def("HasParameterBlock",&ceres::Problem::HasParameterBlock);


//  problem.def("AddResidualBlock",
//              (ceres::ResidualBlockId (ceres::Problem::*)(ceres::CostFunction *,
//                                                          ceres::LossFunction *,
//                                                          double *)) &ceres::Problem::AddResidualBlock);
//  problem.def("AddResidualBlock",
//              (ceres::ResidualBlockId (ceres::Problem::*)(ceres::CostFunction *,
//                                                          ceres::LossFunction *,
//                                                          const std::vector<
//                                                              double *> &)) &ceres::Problem::AddResidualBlock);


  problem.def("AddResidualBlock",
              [](ceres::Problem &myself,
                 ceres::CostFunction *cost,
                 ceres::LossFunction *loss,
                 py::array_t<double> &values) {
                // Should we even do this error checking?
                double *pointer = ParseNumpyData(values);
                return myself.AddResidualBlock(cost, loss, pointer);
              }, py::keep_alive<1, 2>(), // CostFunction
              py::keep_alive<1, 3>(), // LossFunction
              py::return_value_policy::reference);

  problem.def("AddResidualBlock",
              [](ceres::Problem &myself,
                 ceres::CostFunction *cost,
                 ceres::LossFunction *loss,
                 py::array_t<double> &values1,
                 py::array_t<double> &values2) {
                double *pointer1 = ParseNumpyData(values1);
                double *pointer2 = ParseNumpyData(values2);
                return myself.AddResidualBlock(cost, loss, pointer1, pointer2);
              },
              py::keep_alive<1, 2>(), // Cost Function
              py::keep_alive<1, 3>(), // Loss Function
              py::return_value_policy::reference);

  problem.def("AddResidualBlock",
              [](ceres::Problem &myself,
                 ceres::CostFunction *cost,
                 ceres::LossFunction *loss,
                 std::vector<py::array_t<double>> &values) {
                std::vector<double *> pointer_values;
                for (int idx = 0; idx < values.size(); ++idx) {
                  pointer_values.push_back(ParseNumpyData(values[idx]));
                }
                return myself.AddResidualBlock(cost, loss, pointer_values);
              },
              py::keep_alive<1, 2>(), // Cost Function
              py::keep_alive<1, 3>(), // Loss Function
              py::return_value_policy::reference);

  py::class_<ceres::Solver::Options> solver_options(m, "SolverOptions");
  using s_options=ceres::Solver::Options;
  solver_options.def(py::init<>());
  solver_options.def("IsValid", &s_options::IsValid);
  solver_options.def_readwrite("minimizer_type", &s_options::minimizer_type);
  solver_options.def_readwrite("line_search_direction_type",
                               &s_options::line_search_direction_type);
  solver_options.def_readwrite("line_search_type",
                               &s_options::line_search_type);
  solver_options.def_readwrite("nonlinear_conjugate_gradient_type",
                               &s_options::nonlinear_conjugate_gradient_type);
  solver_options.def_readwrite("max_lbfgs_rank", &s_options::max_lbfgs_rank);
  solver_options.def_readwrite("use_approximate_eigenvalue_bfgs_scaling",
                               &s_options::use_approximate_eigenvalue_bfgs_scaling);
  solver_options.def_readwrite("line_search_interpolation_type",
                               &s_options::line_search_interpolation_type);
  solver_options.def_readwrite("min_line_search_step_size",
                               &s_options::min_line_search_step_size);
  solver_options.def_readwrite("line_search_sufficient_function_decrease",
                               &s_options::line_search_sufficient_function_decrease);
  solver_options.def_readwrite("max_line_search_step_contraction",
                               &s_options::max_line_search_step_contraction);
  solver_options.def_readwrite("min_line_search_step_contraction",
                               &s_options::min_line_search_step_contraction);
  solver_options.def_readwrite("max_num_line_search_step_size_iterations",
                               &s_options::max_num_line_search_step_size_iterations);
  solver_options.def_readwrite("max_num_line_search_direction_restarts",
                               &s_options::max_num_line_search_direction_restarts);
  solver_options.def_readwrite("line_search_sufficient_curvature_decrease",
                               &s_options::line_search_sufficient_curvature_decrease);
  solver_options.def_readwrite("max_line_search_step_expansion",
                               &s_options::max_line_search_step_expansion);
  solver_options.def_readwrite("trust_region_strategy_type",
                               &s_options::trust_region_strategy_type);
  solver_options.def_readwrite("dogleg_type", &s_options::dogleg_type);
  solver_options.def_readwrite("use_nonmonotonic_steps",
                               &s_options::use_nonmonotonic_steps);
  solver_options.def_readwrite("max_consecutive_nonmonotonic_steps",
                               &s_options::max_consecutive_nonmonotonic_steps);
  solver_options.def_readwrite("max_num_iterations",
                               &s_options::max_num_iterations);
  solver_options.def_readwrite("max_solver_time_in_seconds",
                               &s_options::max_solver_time_in_seconds);
  solver_options.def_readwrite("num_threads", &s_options::num_threads);
  solver_options.def_readwrite("initial_trust_region_radius",
                               &s_options::initial_trust_region_radius);
  solver_options.def_readwrite("max_trust_region_radius",
                               &s_options::max_trust_region_radius);
  solver_options.def_readwrite("min_trust_region_radius",
                               &s_options::min_trust_region_radius);
  solver_options.def_readwrite("min_relative_decrease",
                               &s_options::min_relative_decrease);
  solver_options.def_readwrite("min_lm_diagonal", &s_options::min_lm_diagonal);
  solver_options.def_readwrite("max_lm_diagonal", &s_options::max_lm_diagonal);
  solver_options.def_readwrite("max_num_consecutive_invalid_steps",
                               &s_options::max_num_consecutive_invalid_steps);
  solver_options.def_readwrite("function_tolerance",
                               &s_options::function_tolerance);
  solver_options.def_readwrite("gradient_tolerance",
                               &s_options::gradient_tolerance);
  solver_options.def_readwrite("parameter_tolerance",
                               &s_options::parameter_tolerance);
  solver_options.def_readwrite("linear_solver_type",
                               &s_options::linear_solver_type);
  solver_options.def_readwrite("preconditioner_type",
                               &s_options::preconditioner_type);
  solver_options.def_readwrite("visibility_clustering_type",
                               &s_options::visibility_clustering_type);
  solver_options.def_readwrite("dense_linear_algebra_library_type",
                               &s_options::dense_linear_algebra_library_type);
  solver_options.def_readwrite("sparse_linear_algebra_library_type",
                               &s_options::sparse_linear_algebra_library_type);
  solver_options.def_readwrite("use_explicit_schur_complement",
                               &s_options::use_explicit_schur_complement);
  solver_options.def_readwrite("use_postordering",
                               &s_options::use_postordering);
  solver_options.def_readwrite("dynamic_sparsity",
                               &s_options::dynamic_sparsity);
  solver_options.def_readwrite("use_mixed_precision_solves",
                               &s_options::use_mixed_precision_solves);
  solver_options.def_readwrite("max_num_refinement_iterations",
                               &s_options::max_num_refinement_iterations);
  solver_options.def_readwrite("use_inner_iterations",
                               &s_options::use_inner_iterations);
  solver_options.def_readwrite("minimizer_progress_to_stdout",
                               &s_options::minimizer_progress_to_stdout);

  py::class_<ceres::CostFunction, PyCostFunction /* <--- trampoline*/>(m,
                                                                       "CostFunction")
      .def(py::init<>())
      .def("num_residuals", &ceres::CostFunction::num_residuals)
      .def("num_parameter_blocks", [](ceres::CostFunction &myself) {
        return myself.parameter_block_sizes().size();
      })
      .def("parameter_block_sizes",
           &ceres::CostFunction::parameter_block_sizes,
           py::return_value_policy::reference)
      .def("set_num_residuals", &PyCostFunction::set_num_residuals)
      .def("set_parameter_block_sizes",
           [](ceres::CostFunction &myself, std::vector<int32_t> &sizes) {
             for (auto s:sizes) {
               const_cast<std::vector<int32_t> &>(myself.parameter_block_sizes()).push_back(
                   s);
             }
           });

  py::class_<ceres::TrivialLoss>(m, "TrivialLoss")
      .def(py::init<>());

  py::class_<ceres::HuberLoss>(m, "HuberLoss")
      .def(py::init<double>());
  py::class_<ceres::SoftLOneLoss>(m, "SoftLOneLoss")
      .def(py::init<double>());

  py::class_<ceres::CauchyLoss>(m, "CauchyLoss")
      .def(py::init<double>());

  py::class_<ceres::LossFunction, PyLossFunction>(m, "LossFunction")
      .def(py::init<>());

  py::class_<ceres::Solver::Summary> solver_summary(m, "Summary");
  using s_summary=ceres::Solver::Summary;
  solver_summary.def(py::init<>());
  solver_summary.def("BriefReport", &ceres::Solver::Summary::BriefReport);
  solver_summary.def("FullReport", &ceres::Solver::Summary::FullReport);
  solver_summary.def("IsSolutionUsable",
                     &ceres::Solver::Summary::IsSolutionUsable);
  solver_summary.def_readwrite("initial_cost",
                               &ceres::Solver::Summary::initial_cost);
  solver_summary.def_readwrite("final_cost",
                               &ceres::Solver::Summary::final_cost);
  solver_summary.def_readwrite("minimizer_type", &s_summary::minimizer_type);
  solver_summary.def_readwrite("line_search_direction_type",
                               &s_summary::line_search_direction_type);
  solver_summary.def_readwrite("line_search_type",
                               &s_summary::line_search_type);
  solver_summary.def_readwrite("nonlinear_conjugate_gradient_type",
                               &s_summary::nonlinear_conjugate_gradient_type);
  solver_summary.def_readwrite("max_lbfgs_rank", &s_summary::max_lbfgs_rank);
  solver_summary.def_readwrite("line_search_interpolation_type",
                               &s_summary::line_search_interpolation_type);
  solver_summary.def_readwrite("trust_region_strategy_type",
                               &s_summary::trust_region_strategy_type);
  solver_summary.def_readwrite("dogleg_type", &s_summary::dogleg_type);
  solver_summary.def_readwrite("visibility_clustering_type",
                               &s_summary::visibility_clustering_type);
  solver_summary.def_readwrite("dense_linear_algebra_library_type",
                               &s_summary::dense_linear_algebra_library_type);
  solver_summary.def_readwrite("sparse_linear_algebra_library_type",
                               &s_summary::sparse_linear_algebra_library_type);
  solver_summary.def_readwrite("termination_type",
                               &s_summary::termination_type);
  solver_summary.def_readwrite("message", &s_summary::message);
  solver_summary.def_readwrite("initial_cost", &s_summary::initial_cost);
  solver_summary.def_readwrite("final_cost", &s_summary::final_cost);
  solver_summary.def_readwrite("fixed_cost", &s_summary::fixed_cost);
  solver_summary.def_readwrite("iterations", &s_summary::iterations);
  solver_summary.def_readwrite("num_successful_steps",
                               &s_summary::num_successful_steps);
  solver_summary.def_readwrite("num_unsuccessful_steps",
                               &s_summary::num_unsuccessful_steps);
  solver_summary.def_readwrite("num_inner_iteration_steps",
                               &s_summary::num_inner_iteration_steps);
  solver_summary.def_readwrite("num_line_search_steps",
                               &s_summary::num_line_search_steps);
  solver_summary.def_readwrite("preprocessor_time_in_seconds",
                               &s_summary::preprocessor_time_in_seconds);
  solver_summary.def_readwrite("minimizer_time_in_seconds",
                               &s_summary::minimizer_time_in_seconds);
  solver_summary.def_readwrite("postprocessor_time_in_seconds",
                               &s_summary::postprocessor_time_in_seconds);
  solver_summary.def_readwrite("total_time_in_seconds",
                               &s_summary::total_time_in_seconds);
  solver_summary.def_readwrite("linear_solver_time_in_seconds",
                               &s_summary::linear_solver_time_in_seconds);
  solver_summary.def_readwrite("num_linear_solves",
                               &s_summary::num_linear_solves);
  solver_summary.def_readwrite("residual_evaluation_time_in_seconds",
                               &s_summary::residual_evaluation_time_in_seconds);

  solver_summary.def_readwrite("num_residual_evaluations",
                               &s_summary::num_residual_evaluations);

  solver_summary.def_readwrite("jacobian_evaluation_time_in_seconds",
                               &s_summary::jacobian_evaluation_time_in_seconds);

  solver_summary.def_readwrite("num_jacobian_evaluations",
                               &s_summary::num_jacobian_evaluations);
  solver_summary.def_readwrite("inner_iteration_time_in_seconds",
                               &s_summary::inner_iteration_time_in_seconds);
  solver_summary.def_readwrite("line_search_cost_evaluation_time_in_seconds",
                               &s_summary::line_search_cost_evaluation_time_in_seconds);
  solver_summary.def_readwrite("line_search_gradient_evaluation_time_in_seconds",
                               &s_summary::line_search_gradient_evaluation_time_in_seconds);
  solver_summary.def_readwrite(
      "line_search_polynomial_minimization_time_in_seconds",
      &s_summary::line_search_polynomial_minimization_time_in_seconds);
  solver_summary.def_readwrite("line_search_total_time_in_seconds",
                               &s_summary::line_search_total_time_in_seconds);
  solver_summary.def_readwrite("num_parameter_blocks",
                               &s_summary::num_parameter_blocks);
  solver_summary.def_readwrite("num_parameters", &s_summary::num_parameters);
  solver_summary.def_readwrite("num_effective_parameters",
                               &s_summary::num_effective_parameters);
  solver_summary.def_readwrite("num_residual_blocks",
                               &s_summary::num_residual_blocks);
  solver_summary.def_readwrite("num_residuals", &s_summary::num_residuals);
  solver_summary.def_readwrite("num_parameter_blocks_reduced",
                               &s_summary::num_parameter_blocks_reduced);
  solver_summary.def_readwrite("num_parameters_reduced",
                               &s_summary::num_parameters_reduced);
  solver_summary.def_readwrite("num_effective_parameters_reduced",
                               &s_summary::num_effective_parameters_reduced);
  solver_summary.def_readwrite("num_residual_blocks_reduced",
                               &s_summary::num_residual_blocks_reduced);
  solver_summary.def_readwrite("num_residuals_reduced",
                               &s_summary::num_residuals_reduced);
  solver_summary.def_readwrite("is_constrained", &s_summary::is_constrained);
  solver_summary.def_readwrite("num_threads_given",
                               &s_summary::num_threads_given);
  solver_summary.def_readwrite("num_threads_used",
                               &s_summary::num_threads_used);
  solver_summary.def_readwrite("linear_solver_ordering_given",
                               &s_summary::linear_solver_ordering_given);
  solver_summary.def_readwrite("linear_solver_ordering_used",
                               &s_summary::linear_solver_ordering_used);
  solver_summary.def_readwrite("schur_structure_given",
                               &s_summary::schur_structure_given);
  solver_summary.def_readwrite("schur_structure_used",
                               &s_summary::schur_structure_used);
  solver_summary.def_readwrite("inner_iterations_given",
                               &s_summary::inner_iterations_given);
  solver_summary.def_readwrite("inner_iterations_used",
                               &s_summary::inner_iterations_used);
  solver_summary.def_readwrite("inner_iteration_ordering_given",
                               &s_summary::inner_iteration_ordering_given);
  solver_summary.def_readwrite("inner_iteration_ordering_used",
                               &s_summary::inner_iteration_ordering_used);

  py::class_<ceres::IterationSummary> iteration_summary(m, "IterationSummary");
  using it_sum=ceres::IterationSummary;
  iteration_summary.def(py::init<>());
  iteration_summary.def_readonly("iteration", &it_sum::iteration);
  iteration_summary.def_readonly("step_is_valid", &it_sum::step_is_valid);
  iteration_summary.def_readonly("step_is_nonmonotonic",
                                 &it_sum::step_is_nonmonotonic);
  iteration_summary.def_readonly("step_is_succesful",
                                 &it_sum::step_is_successful);
  iteration_summary.def_readonly("cost", &it_sum::cost);
  iteration_summary.def_readonly("cost_change", &it_sum::cost_change);
  iteration_summary.def_readonly("gradient_max_norm",
                                 &it_sum::gradient_max_norm);
  iteration_summary.def_readonly("gradient_norm", &it_sum::gradient_norm);
  iteration_summary.def_readonly("step_norm", &it_sum::step_norm);
  iteration_summary.def_readonly("relative_decrease",
                                 &it_sum::relative_decrease);
  iteration_summary.def_readonly("trust_region_radius",
                                 &it_sum::trust_region_radius);
  iteration_summary.def_readonly("eta", &it_sum::eta);
  iteration_summary.def_readonly("step_size", &it_sum::step_size);
  iteration_summary.def_readonly("line_search_function_evaluations",
                                 &it_sum::line_search_function_evaluations);
  iteration_summary.def_readonly("line_search_gradient_evaluations",
                                 &it_sum::line_search_gradient_evaluations);
  iteration_summary.def_readonly("line_search_iterations",
                                 &it_sum::line_search_iterations);
  iteration_summary.def_readonly("linear_solver_iterations",
                                 &it_sum::linear_solver_iterations);
  iteration_summary.def_readonly("iteration_time_in_seconds",
                                 &it_sum::iteration_time_in_seconds);
  iteration_summary.def_readonly("step_solver_time_in_seconds",
                                 &it_sum::step_solver_time_in_seconds);
  iteration_summary.def_readonly("cumulative_time_in_seconds",
                                 &it_sum::cumulative_time_in_seconds);

  py::class_<ceres::IterationCallback,
             PyIterationCallback /* <--- trampoline*/>(m,
                                                       "IterationCallback")
      .def(py::init<>());

  py::class_<ceres::EvaluationCallback,
             PyEvaluationCallBack /* <--- trampoline*/>(m,
                                                        "EvaluationCallback")
      .def(py::init<>());

  py::class_<ceres::FirstOrderFunction,
             PyFirstOrderFunction /* <--- trampoline*/>(m,
                                                        "FirstOrderFunction")
      .def(py::init<>());

  py::class_<ceres::LocalParameterization,
             PyLocalParameterization /* <--- trampoline*/>(m,
                                                           "LocalParameterization")
      .def(py::init<>())
      .def("GlobalSize", &ceres::LocalParameterization::GlobalSize)
      .def("LocalSize", &ceres::LocalParameterization::LocalSize);

  py::class_<ceres::IdentityParameterization>(m, "IdentityParameterization")
      .def(py::init<int>());
  py::class_<ceres::QuaternionParameterization>(m, "QuaternionParameterization")
      .def(py::init<>());
  py::class_<ceres::HomogeneousVectorParameterization>(m,
                                                       "HomogeneousVectorParameterization")
      .def(py::init<int>());
  py::class_<ceres::EigenQuaternionParameterization>(m,
                                                     "EigenQuaternionParameterization")
      .def(py::init<>());
  py::class_<ceres::SubsetParameterization>(m, "SubsetParameterization")
      .def(py::init<int, const std::vector<int> &>());

  py::class_<ceres::GradientProblem> grad_problem(m, "GradientProblem");
  grad_problem.def(py::init([](ceres::FirstOrderFunction *func) {
    ceres::FirstOrderFunction *wrap = new FirstOrderFunctionWrapper(func);
    return ceres::GradientProblem(wrap);
  }), py::keep_alive<1, 2>() // FirstOrderFunction
  );

  grad_problem.def("NumParameters", &ceres::GradientProblem::NumParameters);

  py::class_<ceres::GradientProblemSolver::Options>
      grad_options(m, "GradientProblemOptions");
  using g_options=ceres::GradientProblemSolver::Options;
  grad_options.def(py::init<>());
  grad_options.def("IsValid", &g_options::IsValid);
  grad_options.def_readwrite("line_search_direction_type",
                             &g_options::line_search_direction_type);
  grad_options.def_readwrite("line_search_type",
                             &g_options::line_search_type);
  grad_options.def_readwrite("nonlinear_conjugate_gradient_type",
                             &g_options::nonlinear_conjugate_gradient_type);
  grad_options.def_readwrite("max_lbfgs_rank", &g_options::max_lbfgs_rank);
  grad_options.def_readwrite("use_approximate_eigenvalue_bfgs_scaling",
                             &g_options::use_approximate_eigenvalue_bfgs_scaling);
  grad_options.def_readwrite("line_search_interpolation_type",
                             &g_options::line_search_interpolation_type);
  grad_options.def_readwrite("min_line_search_step_size",
                             &g_options::min_line_search_step_size);
  grad_options.def_readwrite("line_search_sufficient_function_decrease",
                             &g_options::line_search_sufficient_function_decrease);
  grad_options.def_readwrite("max_line_search_step_contraction",
                             &g_options::max_line_search_step_contraction);
  grad_options.def_readwrite("min_line_search_step_contraction",
                             &g_options::min_line_search_step_contraction);
  grad_options.def_readwrite("max_num_line_search_step_size_iterations",
                             &g_options::max_num_line_search_step_size_iterations);
  grad_options.def_readwrite("max_num_line_search_direction_restarts",
                             &g_options::max_num_line_search_direction_restarts);
  grad_options.def_readwrite("line_search_sufficient_curvature_decrease",
                             &g_options::line_search_sufficient_curvature_decrease);
  grad_options.def_readwrite("max_line_search_step_expansion",
                             &g_options::max_line_search_step_expansion);
  grad_options.def_readwrite("max_num_iterations",
                             &g_options::max_num_iterations);
  grad_options.def_readwrite("max_solver_time_in_seconds",
                             &g_options::max_solver_time_in_seconds);
  grad_options.def_readwrite("function_tolerance",
                             &g_options::function_tolerance);
  grad_options.def_readwrite("gradient_tolerance",
                             &g_options::gradient_tolerance);
  grad_options.def_readwrite("parameter_tolerance",
                             &g_options::parameter_tolerance);
  grad_options.def_readwrite("minimizer_progress_to_stdout",
                             &g_options::minimizer_progress_to_stdout);

  py::class_<ceres::GradientProblemSolver::Summary>
      grad_summary(m, "GradientProblemSummary");
  using g_sum=ceres::GradientProblemSolver::Summary;
  grad_summary.def(py::init<>());
  grad_summary.def("BriefReport",
                   &ceres::GradientProblemSolver::Summary::BriefReport);
  grad_summary.def("FullReport",
                   &ceres::GradientProblemSolver::Summary::FullReport);
  grad_summary.def("IsSolutionUsable",
                   &ceres::GradientProblemSolver::Summary::IsSolutionUsable);
  grad_summary.def_readwrite("initial_cost",
                             &ceres::GradientProblemSolver::Summary::initial_cost);
  grad_summary.def_readwrite("final_cost",
                             &ceres::GradientProblemSolver::Summary::final_cost);

  // GradientProblem Solve
  m.def("Solve", [](const ceres::GradientProblemSolver::Options &options,
                    const ceres::GradientProblem &problem,
                    py::array_t<double> &np_params,
                    ceres::GradientProblemSolver::Summary *summary) {
    double *param_ptr = ParseNumpyData(np_params);
    py::gil_scoped_release release;
    ceres::Solve(options, problem, param_ptr, summary);
  });

  // The main Solve function
  m.def("Solve",
        overload_cast_<const ceres::Solver::Options &,
                       ceres::Problem *,
                       ceres::Solver::Summary *>()(&ceres::Solve),
        py::call_guard<py::gil_scoped_release>());

  py::class_<ceres::CRSMatrix> crs_mat(m, "CRSMatrix");
  crs_mat.def(py::init<>());
  crs_mat.def_readwrite("num_cols", &ceres::CRSMatrix::num_cols);
  crs_mat.def_readwrite("num_rows", &ceres::CRSMatrix::num_rows);
  crs_mat.def_readwrite("cols", &ceres::CRSMatrix::cols);
  crs_mat.def_readwrite("rows", &ceres::CRSMatrix::rows);
  crs_mat.def_readwrite("values", &ceres::CRSMatrix::values);

  py::class_<ceres::NumericDiffOptions>
      numdiff_options(m, "NumericDiffOptions");
  numdiff_options.def(py::init<>());
  numdiff_options.def_readwrite("relative_step_size",
                                &ceres::NumericDiffOptions::relative_step_size);
  numdiff_options.def_readwrite("ridders_relative_initial_step_size",
                                &ceres::NumericDiffOptions::ridders_relative_initial_step_size);
  numdiff_options.def_readwrite("max_num_ridders_extrapolations",
                                &ceres::NumericDiffOptions::max_num_ridders_extrapolations);
  numdiff_options.def_readwrite("ridders_epsilon",
                                &ceres::NumericDiffOptions::ridders_epsilon);
  numdiff_options.def_readwrite("ridders_step_shrink_factor",
                                &ceres::NumericDiffOptions::ridders_step_shrink_factor);

  py::class_<ceres::GradientChecker::ProbeResults>
      probe_results(m, "ProbeResults");
  probe_results.def(py::init<>());
  probe_results.def_readwrite("return_value",
                              &ceres::GradientChecker::ProbeResults::return_value);
  probe_results.def_readwrite("residuals",
                              &ceres::GradientChecker::ProbeResults::residuals);
  probe_results.def_readwrite("jacobians",
                              &ceres::GradientChecker::ProbeResults::jacobians);
  probe_results.def_readwrite("local_jacobians",
                              &ceres::GradientChecker::ProbeResults::local_jacobians);
  probe_results.def_readwrite("numeric_jacobians",
                              &ceres::GradientChecker::ProbeResults::numeric_jacobians);
  probe_results.def_readwrite("local_numeric_jacobians",
                              &ceres::GradientChecker::ProbeResults::local_numeric_jacobians);
  probe_results.def_readwrite("maximum_relative_error",
                              &ceres::GradientChecker::ProbeResults::maximum_relative_error);
  probe_results.def_readwrite("error_log",
                              &ceres::GradientChecker::ProbeResults::error_log);

  py::class_<ceres::GradientChecker> gradient_checker(m, "GradientChecker");
  gradient_checker.def(py::init<const ceres::CostFunction *,
                                const std::vector<const ceres::LocalParameterization *> *,
                                const ceres::NumericDiffOptions>());
  gradient_checker.def("Probe",
                       [](ceres::GradientChecker &myself,
                          std::vector<py::array_t<double>> &parameters,
                          double relative_precision,
                          ceres::GradientChecker::ProbeResults *results) {
                         std::vector<double *> param_pointers;
                         for (auto &p:parameters) {
                           param_pointers.push_back(ParseNumpyData(p));
                         }
                         return myself.Probe(param_pointers.data(),
                                             relative_precision,
                                             results);
                       });

  py::class_<ceres::NormalPrior> normal_prior(m, "NormalPrior");
  normal_prior.def(py::init<const ceres::Matrix &, const ceres::Vector &>());

  py::class_<ceres::Context>(m, "Context")
      .def(py::init<>())
      .def("Create", &ceres::Context::Create);

  py::class_<ceres::Covariance::Options> cov_opt(m, "CovarianceOptions");
  using c_opt=ceres::Covariance::Options;
  cov_opt.def_readwrite("sparse_linear_algebra_library_type",
                        &c_opt::sparse_linear_algebra_library_type);
  cov_opt.def_readwrite("algorithm_type", &c_opt::algorithm_type);
  cov_opt.def_readwrite("min_reciprocal_condition_number",
                        &c_opt::min_reciprocal_condition_number);
  cov_opt.def_readwrite("null_space_rank", &c_opt::null_space_rank);
  cov_opt.def_readwrite("num_threads", &c_opt::num_threads);
  cov_opt.def_readwrite("apply_loss_function", &c_opt::apply_loss_function);

  py::class_<ceres::Covariance> cov(m, "Covariance");
  cov.def(py::init<const ceres::Covariance::Options &>());
//  cov.def("Compute",overload_cast_<const std::vector<std::pair<const double*, const double*>>&,
//  ceres::Problem*>()(&ceres::Covariance::Compute));
//  cov.def("Compute",overload_cast_<const std::vector<const double*>&,
//                                   ceres::Problem*>()(&ceres::Covariance::Compute));

  py::class_<ceres::ConditionedCostFunction>
      cond_cost(m, "ConditionedCostFunction");
  cond_cost.def(py::init([](ceres::CostFunction *wrapped_cost_function,
                            const std::vector<ceres::CostFunction *> &conditioners) {
    return new ceres::ConditionedCostFunction(wrapped_cost_function,
                                              conditioners,
                                              ceres::DO_NOT_TAKE_OWNERSHIP);
  }));

  add_pybinded_ceres_examples(m);
  add_custom_cost_functions(m);





  // Untested

  // Things below this line are wrapped ,but are rarely used even in C++ ceres.
  // and thus are not tested.

//  py::class_<ceres::ScaledLoss>(m, "ScaledLoss")
//      .def(py::init<ceres::LossFunction *, double, ceres::Ownership>(),
//           py::arg("ownership") = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP);

}
