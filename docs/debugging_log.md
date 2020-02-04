# Debugging log

Contains some notes on problems encoutered trying to wrap Ceres. Could be
usefull for people in the future.

* AddResidualBlock returns ResidualBlock*. If you use AddResidualBlock and don't
capture the result in a local variable then python would capture it and promptly
delete it. Thus deleting the ResidualBlock in your problem. Fix was to add 
py::return_value_policy::reference so python doesn't manage that memory.
* AutodiffCostFunction takes ownership of the CostFunctor. So if it the 
CostFunctor is created in Python then a double free will happen as the python 
garbage collector will delete the AutodiffCostFunction(deletes CostFunctor) and
the CostFunctor.
* Python manages the memory for the cost functions. This means it can delete the 
cost function before the Problem even uses it. In order to avoid this you have
to make the relationship clear that cost function scope is dependent on the
problem. This is done with the py::keep_alive<> command for AddResidualBlock
* End user must call super().__init__() on custom CostFunctions define in
Python. If this is not done then the Base Class CostFunction is never
initialized.
* Seems like as soon as you start touching python stuff like the py::array you 
need to ensure that you have the GIL. The trampoline classes CostFunction would
crash unless the gil was put as the first line. (Ahh the bug starting occuring
because I put release GIL in the Solve functions. Before I had this the GIL was
held and therefore there was no crash)
