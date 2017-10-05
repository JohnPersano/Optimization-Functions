"""
Copyright John Persano 2017

File name:      convergence_visualizer.py
Description:    A graphing utility for optimization algorithms.
Commit history:
                - 04/22/2017: Initial version
                - 06/05/2017: Renamed class
"""

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 20)

import numpy as np
from matplotlib import cm
from optimization_functions import OptimizationFunction
from mpl_toolkits.mplot3d import axes3d, Axes3D


class ConvergenceVisualizer:
    """
    Graphing utility for optimization algorithms. 
    """

    def __init__(self, function_class: OptimizationFunction, plot_range: tuple=(-5, 5), step_size: float=0.05) -> None:
        """
        Creates a new instance of Graph with any optional parameters.

        :param function_class: The function to graph
        :param plot_range: The range to graph as a tuple (from, to)
        :param step_size: The step size to generate data
        """

        figure = plt.figure()
        self.subplot = figure.add_subplot(111, projection='3d')
        plt.gca().set_xlim([plot_range[0], plot_range[1]])
        plt.gca().set_ylim([plot_range[0], plot_range[1]])

        x = y = np.arange(plot_range[0], plot_range[1], step_size)
        self.x, self.y = np.meshgrid(x, y)

        z = np.array([function_class.get_numpy_function(x) for x in zip(np.ravel(self.x), np.ravel(self.y))])
        self.z = z.reshape(self.x.shape)

    def update_convergence_visualizer(self, result) -> None:
        """
        Update the graph with the newest result matrix.
        :param result: An optimization result as a tuple (position_matrix, result_vector)
        :return: None
        """
        self.subplot.plot_surface(self.x, self.y, self.z, cmap=cm.get_cmap('jet'), linewidth=0, alpha=0.7)
        self.subplot.scatter(result[2][:, 0], result[2][:, 1], result[3], s=150, edgecolor='', alpha=1.0, color=['r'])
        plt.pause(0.001)
        plt.cla()

    def update_convergence_visualizer_gd(self, result) -> None:
        """
        Update the graph with the newest result vector (for GradientDescent).
        :param result: An optimization result as a tuple (position, result_vector)
        :return: None
        """
        self.subplot.plot_surface(self.x, self.y, self.z, cmap=cm.get_cmap('jet'), linewidth=0, alpha=0.7)
        self.subplot.scatter(result[1][0], result[1][1], result[2], s=150, edgecolor='', alpha=1.0, color=['r'])
        plt.pause(0.001)
        plt.cla()

