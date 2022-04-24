#pragma once

#ifdef WITH_PYTORCH

#include <ceres/cost_function.h>
#include <torch/script.h>

class PyTorchCostFunction : public ceres::CostFunction {
 public:
  PyTorchCostFunction(int num_residuals,
                      const std::vector<int32_t>& param_sizes);

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override;

 public:
  mutable torch::jit::script::Module module;
  mutable std::vector<torch::jit::IValue> inputs;
};

#endif  // WITH_PYTORCH
