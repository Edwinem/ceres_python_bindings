#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct ExampleFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }

  static ceres::CostFunction* Create() {
    return new ceres::AutoDiffCostFunction<ExampleFunctor, 1, 1>(
        new ExampleFunctor);
  }
};

void add_custom_cost_functions(py::module& m) {
  // Use pybind11 code to wrap your own cost function which is defined in C++s

  // Here is an example
  m.def("CreateCustomExampleCostFunction", &ExampleFunctor::Create);
}
