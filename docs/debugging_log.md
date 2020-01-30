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