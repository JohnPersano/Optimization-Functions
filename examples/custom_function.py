"""
Copyright John Persano 2017

File name:      custom_function.py
Description:    An example test optimization with three optimizers.
Commit history:
                - 04/23/2017: Initial version
"""
import tensorflow as tf

from algorithms.depreciated.gpo1 import GPO
from algorithms.gd.gd import GradientDescent
from algorithms.pso.np_pso import PSO
from research.optimization_functions import CustomFunction


def custom_tf_function(data):
    """
    Define a Tensorflow compatible function.

    :param data: The tensorflow tensor
    :return: Result
    """
    return tf.reduce_sum(tf.square(data))


def custom_np_function(vector):
    """
    Define a NumPy compatible function.

    :param vector: A single parameter vector
    :return: Result
    """
    summation = 0
    for item in vector:
        summation += item ** 2
    return summation


cost_function = CustomFunction(10)

# Override the cost_function's pre-defined tTensorflow and NumPy functions
cost_function.get_tensorflow_function = custom_tf_function
cost_function.get_numpy_function = custom_np_function

# Perform optimization on the new function
gpo_result = GPO(constants=(2.5, 0.01), epochs=500, inertia=0).optimize(cost_function)
pso_result = PSO(constants=(2, 2), epochs=500).optimize(cost_function)
gd_result = GradientDescent(learning_rate=0.3, epochs=500).optimize(cost_function)

print(gpo_result)
print(pso_result)
print(gd_result)
