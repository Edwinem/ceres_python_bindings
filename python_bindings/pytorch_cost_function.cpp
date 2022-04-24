

#include "pytorch_cost_function.h"

PyTorchCostFunction::PyTorchCostFunction(
    int num_residuals, const std::vector<int32_t>& param_sizes) {
  set_num_residuals(num_residuals);
  *mutable_parameter_block_sizes() = param_sizes;
  inputs.resize(param_sizes.size());
}

bool PyTorchCostFunction::Evaluate(double const* const* parameters,
                                   double* residuals,
                                   double** jacobians) const {
  for (size_t i = 0; i < inputs.size(); ++i) {
    torch::ArrayRef<double> ref(parameters[i], parameter_block_sizes()[i]);
    // if jacobians exist then check if the specific jacobian is set. else false
    bool require_grad = jacobians ? jacobians[i] != nullptr : false;
    inputs[i] = torch::tensor(ref, torch::dtype(torch::kFloat64))
                    .set_requires_grad(require_grad);
  }

  torch::Tensor residual = module.forward(inputs).toTensor();
  for (int i = 0; i < num_residuals(); i++) {
    residuals[i] = residual[i].item<double>();
  }

  if (jacobians) {
    residual.backward();
    for (int i = 0; i < parameter_block_sizes().size(); ++i) {
      int param_size = parameter_block_sizes()[i];
      auto& grad = inputs[i].toTensor().grad();
      if (jacobians[i] && grad.defined()) {
        for (int j = 0; j < param_size; j++) {
          for (int k = 0; k < num_residuals(); k++) {
            jacobians[k][k * param_size + j] =
                grad[k * param_size + j].item<double>();
          }
        }
      }
    }
  }

  return true;
}
