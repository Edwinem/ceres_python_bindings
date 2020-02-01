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

// Needed due to forward decls
#include <ceres/problem_impl.h>
#include <ceres/residual_block.h>
#include <ceres/parameter_block.h>

#include <iostream>
#include <string>

namespace py = pybind11;
template<typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

void add_pybinded_ceres_examples(py::module &m);

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

    pybind11::gil_scoped_acquire gil;
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

      // I believe we don't need this as we are returning a bool
//      if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
//        static pybind11::detail::overload_caster_t<bool> caster;
//        return pybind11::detail::cast_ref<bool>(std::move(o), caster);
//      } else


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

void ParseNumpyDataToVector(py::array_t<double> &np_buf,
                            std::vector<double *> &vec) {
  py::buffer_info info = np_buf.request();
  if (info.ndim > 2) {
    std::string error_msg("Number of dimensions must be <=2. This function"
                          "only allows either an array or 2D matrix "
                              + std::to_string(info.ndim));
    throw std::runtime_error(
        error_msg);
  }
  if (info.ndim == 2) {
    // Row or Column Vector. Represents 1 parameter
    if (info.shape[0] == 1 || info.shape[1] == 1) {
      double *ptr = (double *) info.ptr;
      vec.push_back(ptr);

    }
  } else { // is array so just take ptr value
    vec.push_back((double *) info.ptr);
  }
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
                std::vector<double *> pointer_values;
                ParseNumpyDataToVector(values, pointer_values);

                return myself.AddResidualBlock(cost, loss, pointer_values[0]);
              }, py::return_value_policy::reference);
  problem.def("AddResidualBlock",
              [](ceres::Problem &myself,
                 ceres::CostFunction *cost,
                 ceres::LossFunction *loss,
                 py::array_t<double> &values1,
                 py::array_t<double> &values2) {
                std::vector<double *> pointer_values;
                ParseNumpyDataToVector(values1, pointer_values);
                ParseNumpyDataToVector(values2, pointer_values);

                return myself.AddResidualBlock(cost,
                                               loss,
                                               pointer_values[0],
                                               pointer_values[1]);
              }, py::keep_alive<1, 2>(), py::return_value_policy::reference);

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

  py::class_<ceres::LossFunction, PyLossFunction>(m, "LossFunction")
      .def(py::init<>());

  py::class_<ceres::Solver::Summary> solver_summary(m, "Summary");
  solver_summary.def(py::init<>());
  solver_summary.def("BriefReport", &ceres::Solver::Summary::BriefReport);
  solver_summary.def("FullReport", &ceres::Solver::Summary::FullReport);
  solver_summary.def("IsSolutionUsable",
                     &ceres::Solver::Summary::IsSolutionUsable);
  solver_summary.def_readwrite("initial_cost",
                               &ceres::Solver::Summary::initial_cost);
  solver_summary.def_readwrite("final_cost",
                               &ceres::Solver::Summary::final_cost);



  // The main Solve function
  m.def("Solve", overload_cast_<const ceres::Solver::Options &,
                                ceres::Problem *,
                                ceres::Solver::Summary *>()(&ceres::Solve));

  add_pybinded_ceres_examples(m);

}
