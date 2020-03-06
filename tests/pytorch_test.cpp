#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>


int main(int argc, const char* argv[]) {



  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("../python_tests/example_torch_module.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  std::vector<torch::jit::IValue> inputs;
  double param=1;
  torch::ArrayRef<double> ref(&param, 1);
  inputs.push_back(torch::tensor(ref, torch::dtype(torch::kFloat64)).set_requires_grad(true));
  //inputs.push_back(torch::ones({1}).set_requires_grad(true));



// Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();


  std::cout << output << std::endl;

  output.backward();

  auto &grad = inputs[0].toTensor().grad();

  std::cout << grad << std::endl;

}