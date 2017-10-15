"""
Copyright John Persano 2017

File name:      gpo.py
Description:    Gradient Population Optimization written for Tensorflow.
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


class GPODEP:
    """
    Gradient Population Optimizer. Uses a population of gradients to optimize a function.
    
    Example usage:
        result = GPO.optimize(function)
    """

    class Hardware(Enum):
        """
        Hardware constants for GPO hardware selection.
        """
        CPU = 1
        GPU_1 = 2
        GPU_2 = 3

    def __init__(self,
                 epochs: int = 500,
                 population_size: int = 30,
                 spawn_range: tuple = (-100, 100),
                 constants: tuple = (2, 2.5),
                 velocity_max: float = 100,
                 inertia: tuple = (0.9, 0.4),
                 hardware: Hardware = Hardware.CPU,
                 report_frequency: int = 25,
                 verbose: bool = False,
                 plot: bool = False):
        """
        Creates a new instance of GPO with any optional parameters.
        
        :param epochs: Maximum number of iterations
        :param spawn_range: Population spawn range as a tuple (from, to)
        :param constants: Optimization constants as a tuple (social, gradient)
        :param velocity_max: Maximum iterative velocity
        :param inertia: Inertia value of the velocity calculation (start, end)
        :param hardware: Run optimization on CPU or GPU
        :param report_frequency: Number of epochs between report sampling
        :param verbose: True if should show output
        :param plot: True if should plot each iteration
        """

        # Initialize functional class variables
        self.epochs = epochs
        self.population_size = population_size
        self.spawn_range = spawn_range
        self.social_constant = constants[0]
        self.gradient_constant = constants[1]
        self.velocity_max = velocity_max
        self.inertia = inertia[0]
        self.inertia_step = (inertia[0] - inertia[1]) / epochs
        self.hardware = hardware

        # Initialize non-functional class variables
        self.report_frequency = report_frequency
        self.verbose = verbose
        self.plot = plot

        # Setup class values to be used later on
        self._function_class = None
        self._global_best_position = None
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

        # The benchmark will run the actual optimization
        benchmark_results = Benchmark().run(self._start)

        # An issue occurred with benchmark results
        if not benchmark_results:
            Log.error("There was an issue getting benchmark results.")
            return Report("GPO", self.hardware.name, self._function_class.get_name(),
                          self._global_best_result,
                          self._global_best_position, self.iterative_results, Benchmark.ERROR_CODE,
                          Benchmark.ERROR_CODE)

        # No issues have occurred with the benchmark, show the full report
        return Report("GPO", self.hardware.name, self._function_class.get_name(), self._global_best_result,
                      self._global_best_position, self.iterative_results, benchmark_results.runtime,
                      benchmark_results.function_calls)

    def _start(self) -> None:
        """
        Private function, should not be used directly.
        Called by the Benchmark class to start the optimization.
        :return: None
        """

        if self.verbose:
            Log.info("Starting GPO optimization")

        # Select which hardware to use, default to CPU
        if self.hardware == GPO.Hardware.GPU_1:
            device = "/gpu:0"
            if self.verbose:
                Log.info("Using GPU 1 for optimization")
        elif self.hardware == GPO.Hardware.GPU_2:
            device = "/gpu:1"
            if self.verbose:
                Log.info("Using GPU 2 for optimization")
        else:
            if self.verbose:
                Log.info("Using CPU for optimization")
            device = "/cpu:0"

        # Use the appropriate hardware with Tensorflow
        with tf.device(device):

            ############################################################################################################
            # Initialization
            ############################################################################################################

            # Get the dimensionality of the cost function
            dimensions = self._function_class.dimensions

            # Initialize the best position as a random value
            best_position = tf.Variable(tf.random_uniform([dimensions],
                                                          self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            # Create a uniformly distributed matrix of size [population_size, dimensions] within the spawn range
            position_matrix = tf.Variable(tf.random_uniform([self.population_size, dimensions],
                                                            self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            # Create a velocity matrix of size [population_size, dimensions] with zeros
            velocity_matrix = tf.Variable(tf.zeros([self.population_size, dimensions], dtype=tf.float64))

            # Initialize the inertia as the largest inertia value
            inertia = tf.Variable(self.inertia, dtype=tf.float64)

            # Initialize the inertia step size
            step_size = tf.constant(self.inertia_step, dtype=tf.float64)

            ############################################################################################################
            # Social calculations
            ############################################################################################################

            # Create a uniformly distributed matrix of size [population_size, dimensions] between zero and one
            social_stochastic_matrix = tf.random_uniform([self.population_size, dimensions], 0, 1, dtype=tf.float64)

            # Create a Tensorflow constant for the social constant
            social_constant = tf.constant(self.social_constant, dtype=tf.float64)

            # Multiply the social_stochastic_matrix by the social_constant
            social_scaled_matrix = tf.multiply(social_constant, social_stochastic_matrix)

            # Multiply the social_scaled_matrix by the difference between the best_position and the position_matrix
            social_factor = tf.multiply(social_scaled_matrix, tf.subtract(best_position, position_matrix))

            ############################################################################################################
            # Gradient calculations
            ############################################################################################################

            # Create a Tensorflow constant for the gradient constant
            gradient_constant = tf.constant(self.gradient_constant, dtype=tf.float64)

            # Calculate the results for each position vector in the position_matrix
            result_vector = tf.map_fn(self._function_class.get_tensorflow_function, position_matrix)

            # Calculate the gradient for each position vector in the position_matrix
            gradients = tf.gradients(result_vector, position_matrix)[0]

            clipped_gradient = tf.clip_by_global_norm([gradients], tf.constant(2.5, dtype=tf.float64))[0][0]

            #gradient_factor = tf.clip_by_value(clipped_gradient, -tf.abs(social_factor), tf.abs(social_factor))
            gradient_factor = tf.multiply(clipped_gradient, gradient_constant)

            ############################################################################################################
            # Position update calculations
            ############################################################################################################

            # Decrease inertia linearly
            next_inertia = inertia.assign(tf.subtract(inertia, step_size))

            # Multiply the inertia_constant by the velocity_matrix to get momentum
            momentum = tf.multiply(next_inertia, velocity_matrix)

            # Calculate the new velocity
            velocity = velocity_matrix.assign(tf.add(momentum, tf.subtract(social_factor, gradient_factor)))

            # Clip the velocity to the velocity max
            clipped_velocity = tf.clip_by_value(velocity, -self.velocity_max, self.velocity_max)

            # Add the previous positions to the velocity to get the new positions
            update_particles = position_matrix.assign(tf.add(position_matrix, clipped_velocity))

            # Calculate a new result vector after velocity has been applied to the position_matrix
            new_result_vector = tf.map_fn(self._function_class.get_tensorflow_function, update_particles)

            # Get the best position from the position_matrix
            new_best_position = tf.gather(update_particles, tf.argmin(new_result_vector, 0))

            # If the current best particle is better than the last best particle, update the best particle
            best_position_update = best_position.assign(tf.where(
                tf.less(self._function_class.get_tensorflow_function(new_best_position),
                        self._function_class.get_tensorflow_function(best_position)), new_best_position, best_position))

            # Return the best result, the best particle, and the particle matrix
            optimization_result = (self._function_class.get_tensorflow_function(best_position_update),
                                   best_position_update, position_matrix, new_result_vector)

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

                # Log.error(result[4])

                # Update the graph if the plot flag is True
                if self.plot:
                    graph.update_convergence_visualizer(result)

                # Capture iterative data
                if (i + 1) % (iterative_scalar * self.report_frequency) == 0:
                    iterative_scalar += 1
                    self.iterative_results.append((i + 1, result[0]))

            self._global_best_result = result[0]
            self._global_best_position = result[1]

            if self.verbose:
                writer.close()
