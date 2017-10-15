"""
Copyright John Persano 2017

File name:      simple.py
Description:    An example test optimization with three optimizers.
Commit history:
                - 04/23/2017: Initial version
"""
from algorithms.depreciated.gpo1 import GPO
from algorithms.gd.gd import GradientDescent
from algorithms.pso.np_pso import PSO
from research.optimization_functions import Matya

cost_function = Matya(2)

gpo_result = GPO(constants=(2.5, 0.01), epochs=500, inertia=0).optimize(cost_function)
pso_result = PSO(constants=(2, 2), epochs=500).optimize(cost_function)
gd_result = GradientDescent(learning_rate=0.3, epochs=500).optimize(cost_function)

print(gpo_result)
print(pso_result)
print(gd_result)
