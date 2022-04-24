#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

// forward decl needed for residual block return
#include <ceres/residual_block.h>

#ifdef WITH_PYTORCH

#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.

#include "pytorch_cost_function.h"

namespace py = pybind11;

void DoPtr(torch::jit::script::Module* m) {
  std::cout << "ptr works" << std::endl;
}

void DoCopy(torch::jit::script::Module m) {
  std::cout << "copy works " << std::endl;
}

void TestTensor(torch::Tensor a) {
  std::cout << "Tensor Success " << a.data_ptr() << std::endl;
  double* ptr = a.data_ptr<double>();
  std::cout << "Double ptr " << ptr << " after ptr " << std::endl;
}

void TestLiveRun(const std::string& code) {
  std::cout << "jit compiling code " << std::endl;

  auto m = torch::jit::compile(code);
  std::vector<torch::jit::IValue> inputs;
  double param = 1;
  torch::ArrayRef<double> ref(&param, 1);
  inputs.push_back(torch::tensor(ref, torch::dtype(torch::kFloat64))
                       .set_requires_grad(true));
  // inputs.push_back(torch::ones({1}).set_requires_grad(true));

  // Execute the model and turn its output into a tensor.
  torch::jit::IValue out = m->run_method("forward", inputs[0]);

  auto output = out.toTensor();

  std::cout << out << std::endl;

  output.backward();

  auto& grad = inputs[0].toTensor().grad();

  std::cout << grad << std::endl;
}

ceres::CostFunction* CreateTorchCostFunction(
    const std::string& filepath,
    int num_residuals,
    const std::vector<int32_t>& param_sizes) {
  PyTorchCostFunction* cost_func =
      new PyTorchCostFunction(num_residuals, param_sizes);
  try {
    cost_func->module = torch::jit::load(filepath);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    delete cost_func;
    throw std::runtime_error("Serialized model does not exist");
  }

  return cost_func;
}

// Adds pytorch functionality to the module.
void add_torch_functionality(py::module& m) {
  m.def("DoPtr", &DoPtr);
  m.def("DoCopy", &DoCopy);
  m.def("TestTensor", &TestTensor);
  m.def("LiveRun", &TestLiveRun);

  py::class_<ceres::Problem> problem =
      (py::class_<ceres::Problem>)m.attr("Problem");

  problem.def("AddParameterBlockPythonFunc",
              [](ceres::Problem& myself,
                 const std::string& func_src_code,
                 const std::vector<int32_t>& sizes) {
                std::cout << func_src_code << std::endl;
              });

  problem.def("AddParameterBlockTest",
              [](ceres::Problem& myself, torch::jit::script::Module* module) {
                std::cout << "Problem suceess" << std::endl;
              });

  problem.def(
      "AddResidualBlock",
      [](ceres::Problem& myself,
         ceres::CostFunction* cost,
         ceres::LossFunction* loss,
         std::vector<torch::Tensor>& tensors) {
        std::vector<double*> pointers;
        for (int i = 0; i < tensors.size(); ++i) {
          auto& t = tensors[i];
          pointers.push_back(t.data_ptr<double>());
        }
        return myself.AddResidualBlock(cost, loss, pointers);
      },
      py::keep_alive<1, 2>(),  // Cost Function
      py::keep_alive<1, 3>(),  // Loss Function
      py::return_value_policy::reference);

  m.def("CreateTorchCostFunction", &CreateTorchCostFunction);
}

#endif
