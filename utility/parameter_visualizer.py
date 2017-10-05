"""
Copyright John Persano 2017

File name:      parameter_visualizer.py
Description:    A graphing utility for optimization algorithms.
Commit history:
                - 06/05/2017: Initial version
"""


from algorithms.gpo.gpo import GPO

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 20)

import numpy as np
from matplotlib import cm
from optimization_functions import OptimizationFunction


class ParameterVisualizer:
    """
    Graphing utility for optimization algorithms.
    """

    def __init__(self, plot_range: tuple=(0, 5), samples: int=35) -> None:
        """
        Creates a new instance of Graph with any optional parameters.

        :param plot_range: The range to graph as a tuple (from, to)
        :param samples: The amount of samples to take across the range
        """
        self.plot_range = plot_range
        self.samples = samples

    def visualize_gpo(self, function_class: OptimizationFunction) -> None:
        """
        Update the graph with the newest result matrix.

        :param function_class: The optimization functions whose parameters will be visualized
        :return: None
        """

        def visualization_function(x_mesh, y_mesh, samples):
            result_matrix = np.zeros((samples, samples))
            for i in range(samples):
                for j in range(samples):

                    # Defining [0] and [1] to be the same simply means linspace will not be used
                    social_linspace = (abs(x_mesh[i, j]), abs(x_mesh[i, j]))
                    gradient_linspace = (abs(y_mesh[i, j]), abs(y_mesh[i, j]))

                    result = GPO(social_linspace=social_linspace,
                                 gradient_linspace=gradient_linspace,
                                 inertia=0.4,
                                 epochs=1000,
                                 hardware=GPO.Hardware.CPU).optimize(function_class).best_result
                    result_matrix[i, j] = result
                    print(result)
            return result_matrix

        x = np.linspace(self.plot_range[0], self.plot_range[1], self.samples)
        y = np.linspace(self.plot_range[0], self.plot_range[1], self.samples)

        X, Y = np.meshgrid(x, y)

        Z = visualization_function(X, Y, self.samples)

        plt.contour(X, Y, Z, linewidths=0.5, colors='k')
        plt.contourf(X, Y, Z, cmap=cm.get_cmap('jet'))
        plt.colorbar()
        plt.show()
