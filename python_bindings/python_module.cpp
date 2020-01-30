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

namespace py = pybind11;

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

  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const override {

    // Resize the vectors passed to python to the proper size. And set the
    // pointer values
    parameters_vec.resize(this->parameter_block_sizes().size());
    jacobians_vec.resize(this->parameter_block_sizes().size());
    for (size_t idx = 0; idx < parameter_block_sizes().size(); ++idx) {
      parameters_vec[idx] = const_cast<double *>(parameters[idx]);
      jacobians_vec[idx] = jacobians[idx];
    }

    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
        pybind11::get_overload(static_cast<const ceres::CostFunction *>(this),
                               "Evaluate");
    if (overload) {
      auto o = overload(this->parameters_vec, residuals, this->jacobians_vec);

      // I believe we don't need this as we are returning a bool
//      if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
//        static pybind11::detail::overload_caster_t<bool> caster;
//        return pybind11::detail::cast_ref<bool>(std::move(o), caster);
//      } else


      return pybind11::detail::cast_safe<bool>(std::move(o));
    }
    pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(
        Ceres::CostFunction) "::" "Evaluate \"");
  }

 private:
  // Vectors used to pass double pointers to python as pybind does not wrap
  // double pointers(**) like Ceres uses.
  // Mutable so they can be modified by the const function.
  mutable std::vector<double *> parameters_vec;
  mutable std::vector<double *> jacobians_vec;

};

class PyLossFunction : public ceres::LossFunction {
 public:
  /* Inherit the constructors */
  using ceres::LossFunction::LossFunction;

  void Evaluate(double sq_norm, double out[3]) const override {

  }

};

struct HelloWorldCostFunctor {
  template<typename T>
  bool operator()(const T *const x, T *residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }
};

ceres::CostFunction *CreateHelloWorldCostFunction() {
  return new ceres::AutoDiffCostFunction<HelloWorldCostFunctor,
                                         1,
                                         1>(new HelloWorldCostFunctor);
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
                 py::array_t<double> values) {
                std::vector<double *> pointer_values;
                py::buffer_info info = values.request();
//    if (info.ndim != 1) {
//      throw std::runtime_error("Number of dimensions must be one");
//    }
                double *ptr = (double *) info.ptr;
//                for (size_t idx = 0; idx < info.shape[0]; idx++) {
//                  pointer_values.push_back(&ptr[idx]);
//                  std::cout << &ptr[idx] << std::endl;
//                }
                return myself.AddResidualBlock(cost, loss, ptr);
              }, py::return_value_policy::reference);

  //const int MAX_ADDRESIDUALBLOCK_VARIABLES=10;

//#define REP0(X)
//#define REP1(X) X
//#define REP2(X) REP1(X) X
//#define REP3(X) REP2(X) X
//#define REP4(X) REP3(X) X
//#define REP5(X) REP4(X) X
//#define REP6(X) REP5(X) X
//#define REP7(X) REP6(X) X
//#define REP8(X) REP7(X) X
//#define REP9(X) REP8(X) X
//#define REP10(X) REP9(X) X
//
//#define REP(HUNDREDS,TENS,ONES,X) \
//  REP##HUNDREDS(REP10(REP10(X))) \
//  REP##TENS(REP10(X)) \
//  REP##ONES(X)






//  problem.def("AddResidualBlock",(ceres::ResidualBlockId (ceres::Problem::*)(ceres::CostFunction*,
//                                                   ceres::LossFunction*,
//                                                   double* const* const,
//  int))&ceres::Problem::AddResidualBlock);
  /*
  problem.def("AddResidualBlock",py::overload_cast<ceres::CostFunction*,
                                                   ceres::LossFunction*,
                                                   double*,double*>(&ceres::Problem::AddResidualBlock));*/


  py::class_<ceres::Solver::Options> solver_options(m, "SolverOptions");
  using s_options=ceres::Solver::Options;
  solver_options.def(py::init<>());
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
  solver_options.def_readwrite("minimizer_progress_to_stdout",
                               &s_options::minimizer_progress_to_stdout);

  solver_options.def_readwrite("linear_solver_type",
                               &s_options::linear_solver_type);

  py::class_<ceres::CostFunction, PyCostFunction /* <--- trampoline*/>(m,
                                                                       "CostFunction")
      .def(py::init<>())
      .def("num_residuals", &ceres::CostFunction::num_residuals)
      .def("num_parameter_blocks", [](ceres::CostFunction &myself) {
        return myself.parameter_block_sizes().size();
      });

  py::class_<ceres::LossFunction, PyLossFunction>(m, "LossFunction")
      .def(py::init<>());

  py::class_<ceres::Solver::Summary> solver_summary(m, "SolverSummary");
  solver_summary.def(py::init<>());
  solver_summary.def("BriefReport", &ceres::Solver::Summary::BriefReport);



  // The main Solve function
  m.def("Solve", py::overload_cast<const ceres::Solver::Options &,
                                   ceres::Problem *,
                                   ceres::Solver::Summary *>(&ceres::Solve));

  m.def("CreateHelloWorldCostFunction", &CreateHelloWorldCostFunction);

}
