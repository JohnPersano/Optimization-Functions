"""
Copyright John Persano 2017

File name:      gradient_descent.py
Description:    Gradient Descent written for Tensorflow.
Commit history:
                - 04/22/2017 - Initial version
"""
from enum import Enum

import tensorflow as tf

from research.optimization_functions import OptimizationFunction
from utility.benchmark import Benchmark
from utility.convergence_visualizer import ConvergenceVisualizer
from utility.log import Log
from utility.report import Report


class GradientDescent:
    """
    Standard Gradient Descent.
    """
    class Hardware(Enum):
        """
        Hardware constants for GradientDescent hardware selection.
        """
        CPU = 1
        GPU_1 = 2
        GPU_2 = 3

    def __init__(self,
                 epochs: int = 500,
                 spawn_range: tuple = (-100, 100),
                 learning_rate: float = 0.3,
                 hardware: Hardware = Hardware.CPU,
                 report_frequency: int = 25,
                 verbose: bool = False,
                 plot: bool = False):
        """
        Creates a new instance of GradientDescent with any optional parameters.
    
        :param epochs: Maximum number of iterations
        :param spawn_range: Population spawn range as a tuple (from, to)
        :param learning_rate: The learning rate or alpha value
        :param hardware: Run optimization on CPU or GPU
        :param report_frequency: Number of epochs between report sampling
        :param verbose: True if should show output
        :param plot: True if should plot each iteration
        """

        # Initialize functional class variables
        self.epochs = epochs
        self.spawn_range = spawn_range
        self.learning_rate = learning_rate
        self.hardware = hardware

        # Initialize non-functional class variables
        self.report_frequency = report_frequency
        self.verbose = verbose
        self.plot = plot

        # Setup class values to be used later on
        self._function_class = None
        self._global_best_particle = None
        self._global_best_result = None
        self.iterative_results = []

    def optimize(self, function_class: OptimizationFunction) -> Report:
        """
        Optimize an OptimizationFunction.
        :param function_class: What OptimizationFunction to optimize
        :return: A plaintext Report
        """

        self._function_class = function_class

        # If verbose, print out the name of the function and its dimensions
        if self.verbose:
            print(self._function_class.get_name(), "with", self._function_class.dimensions, "dimensions")

        # The benchmark will run the optimization
        benchmark_results = Benchmark().run(self._start)

        # An issue occurred with benchmark results,
        if not benchmark_results:
            Log.error("There was an issue getting benchmark results.")
            return Report("Gradient Descent", self.hardware.name, self._function_class.get_name(),
                          self._global_best_result,
                          self._global_best_particle, self.iterative_results, Benchmark.ERROR_CODE,
                          Benchmark.ERROR_CODE)

            # No issues have occurred with the benchmark, show the full report
        return Report("Gradient Descent", self.hardware.name, self._function_class.get_name(),
                      self._global_best_result,
                      self._global_best_particle, self.iterative_results, benchmark_results.runtime,
                      benchmark_results.function_calls)

    def _start(self) -> None:
        """
        Private function, should not be used directly.
        Called by the Benchmark class to start the optimization.
        :return: None
        """

        # If verbose, print out the type of graph being used
        if self.verbose:
            Log.debug("Starting Gradient Descent")

        # Select which hardware to use, default to CPU
        if self.hardware == GradientDescent.Hardware.GPU_1:
            device = "/gpu:0"
            if self.verbose:
                Log.info("Using GPU 1 for optimization")
        elif self.hardware == GradientDescent.Hardware.GPU_2:
            device = "/gpu:1"
            if self.verbose:
                Log.info("Using GPU 2 for optimization")
        else:
            if self.verbose:
                Log.info("Using CPU for optimization")
            device = "/cpu:0"

        tf.reset_default_graph()

        # Use the appropriate hardware
        with tf.device(device):

            # Create a uniformly distributed vector of size [dimensions] within the spawn range
            position = tf.Variable(tf.random_uniform([self._function_class.dimensions],
                                                     self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            best_result = tf.Variable(99999999, dtype=tf.float64)

            # Use the Tensorflow variant as the cost function
            cost_function = self._function_class.get_tensorflow_function

            # Calculate the result of the current position vector
            result = cost_function(position)
            result_condition = tf.less(result, best_result)
            compare = tf.cond(result_condition, lambda: best_result.assign(result), lambda: best_result.assign(best_result))

            # Use the native Tensorflow Gradient Descent
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train = optimizer.minimize(result)

            # Return the best result, the best particle, and the particle matrix
            optimization_result = (best_result, position, best_result, train, compare)

        # Create a new Tensorflow session
        with tf.Session() as session:

            # Initialize the variables once
            session.run(tf.global_variables_initializer())

            # If verbose, create a TensorBoard
            writer = None
            if self.verbose:
                writer = tf.summary.FileWriter("output/tensorboard", session.graph)

            # Graph the optimization if the plot flag is True
            graph = None
            if self.plot:
                graph = ConvergenceVisualizer(self._function_class)

            # Iterate through the max number of epochs
            result = None
            iterative_scalar = 1
            for i in range(0, self.epochs):

                result = session.run(optimization_result)

                # If verbose, print the best result so far
                if self.verbose:
                    Log.debug(result[0])

                # Update the graph if the plot flag is True
                if self.plot:
                    graph.update_convergence_visualizer_gd(result)

                # Capture iterative data
                if (i + 1) % (iterative_scalar * self.report_frequency) == 0:
                    iterative_scalar += 1
                    self.iterative_results.append((i + 1, result[0]))

            self._global_best_result = result[0]
            self._global_best_particle = result[1]

            if self.verbose:
                writer.close()
