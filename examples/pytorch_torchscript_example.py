
import torch # Torch must be imported before PyCeres
import PyCeres  # Import the Python Bindings

class ExampleTorchModule(torch.nn.Module):
    def __init__(self):
        super(ExampleTorchModule, self).__init__()

    def forward(self, input):
        residual=10-input
        return residual

module=ExampleTorchModule()
torchscript = torch.jit.script(module)


# Currently we pass torchscript modules as files
filename="example_torch_module.pt"
torchscript.save(filename)
# Create a PyTorchCostFunction. From a torchscript file. Additionally residual size and parameter block sizes must
# be passed.
torch_cost=PyCeres.CreateTorchCostFunction(filename,1,[1])

# Create the data in a PyTorch tensor
data = [0.5]
tensor=torch.tensor(data,dtype=torch.float64)
tensor_vec=[tensor] # Data must be passed as a list of tensors

# Create problem and options as usual
problem = PyCeres.Problem()
res=problem.AddResidualBlock(torch_cost,None,tensor_vec)

options = PyCeres.SolverOptions()
options.linear_solver_type = PyCeres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout = True
summary = PyCeres.Summary()
PyCeres.Solve(options, problem, summary)
print(summary.BriefReport())
print("x : " + str(0.5) + " -> " + str(tensor[0]))