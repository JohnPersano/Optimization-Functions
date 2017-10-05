"""
Copyright John Persano 2017

File name:      gpo.py
Description:    Gradient Population Optimization written for Tensorflow.
Version:        0.3

Commit history:
                - 04/22/2017 - Initial version
                - 06/06/2017 - Added linear space diversity

"""

from enum import Enum
import tensorflow as tf
from optimization_functions import OptimizationFunction
from utility.benchmark import Benchmark
from utility.convergence_visualizer import ConvergenceVisualizer
from utility.log import Log
from utility.report import Report
import numpy as np


# noinspection PyUnresolvedReferences
class GPO:
    """
    Gradient Population Optimizer. Uses a population of gradients to optimize a function.
    
    Example usage:
        result = GPO().optimize(function)
    """

    class Hardware(Enum):
        """
        Hardware constants for GPO hardware selection.
        """
        CPU = 1
        GPU_1 = 2
        GPU_2 = 3

    def __init__(self,
                 epochs: int = 1000,
                 population_size: int = 36,
                 spawn_range: tuple = (-100, 100),
                 social_linspace: tuple = (1.75, 2.25),
                 gradient_linspace: tuple = (3, 5),
                 velocity_max: float = 100,
                 inertia: tuple or float = 0.2,
                 hardware: Hardware = Hardware.CPU,
                 report_frequency: int = 25,
                 verbose: bool = False,
                 plot: bool = False):
        """
        Creates a new instance of GPO with any optional parameters.
        
        :param epochs:            Maximum number of iterations
        :param spawn_range:       Population spawn range as the tuple (from, to)
        :param population_size:   Size of the population as a perfect square
        :param social_linspace:   Linear space for the social constant as the tuple (from, to)
        :param velocity_max:      Maximum iterative velocity
        :param inertia:           Inertia value of the velocity calculation as the tuple (start, end) or a float
        :param hardware:          Run optimization on CPU or GPU
        :param report_frequency:  Number of epochs between report sampling
        :param verbose:           True if should show verbose output
        :param plot:              True if should plot each iteration
        """

        # Initialize functional class variables
        self.epochs = epochs
        self.population_size = population_size
        self.spawn_range = spawn_range
        self.social_linspace = social_linspace
        self.gradient_linspace = gradient_linspace
        self.velocity_max = velocity_max
        self.inertia = inertia[0] if type(inertia) == tuple else inertia
        self.inertia_step = (inertia[0] - inertia[1]) / epochs if type(inertia) == tuple else 0
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

        # Ensure the population size is a perfect square
        if not int(np.sqrt(self.population_size)) ** 2 == self.population_size:
            raise ValueError("The population size must be a perfect square")

        # Get linear spaces for the social and gradient parameters
        social_space = np.linspace(self.social_linspace[0], self.social_linspace[1], np.sqrt(self.population_size))
        grad_space = np.linspace(self.gradient_linspace[0], self.gradient_linspace[1], np.sqrt(self.population_size))

        # Get meshgrids for social and gradient linear spaces which will allow all combinations
        x_mesh, y_mesh = np.meshgrid(social_space, grad_space)

        # Create column vectors for each constant
        self.social_constant = x_mesh.flatten()[:, np.newaxis]
        self.gradient_constant = y_mesh.flatten()[:, np.newaxis]


    def optimize(self, function_class: OptimizationFunction) -> Report:
        """
        Optimize an OptimizationFunction.
        :param function_class: What OptimizationFunction to optimize
        :return: A plaintext Report
        """

        self._function_class = function_class

        self.spawn_range = function_class.get_asymmetric_domain()

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
        return Report("GPO V2", self.hardware.name, self._function_class.get_name(), self._global_best_result,
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

            # Create a position matrix, a velocity matrix, and place them into a tensor
            pm_init = tf.random_uniform([self.population_size, dimensions],
                                        self.spawn_range[0], self.spawn_range[1], dtype=tf.float64)
            vm_init = tf.zeros([self.population_size, dimensions], dtype=tf.float64)
            pv_tensor = tf.Variable(tf.stack([pm_init, vm_init]), dtype=tf.float64)

            # Set the best position variable initially to the first element of the position matrix
            best_position = tf.Variable(tf.gather(pm_init, 0), dtype=tf.float64)

            # Keep a reference to the position matrix
            position_matrix = tf.gather(pv_tensor, 0)

            # Setup the inertia variable and inertia step size constant
            inertia = tf.Variable(self.inertia, dtype=tf.float64)
            step_size = tf.constant(self.inertia_step, dtype=tf.float64)

            # The result vector will hold all of the results of the position matrix applied to the cost function
            result_vector = tf.map_fn(self._function_class.get_tensorflow_function, position_matrix)

            # Figure out the best position vector in the matrix and compare it to the past best position
            new_best_position = tf.gather(position_matrix, tf.argmin(result_vector, 0))
            best_position_update = best_position.assign(tf.where(
                tf.less(self._function_class.get_tensorflow_function(new_best_position),
                        self._function_class.get_tensorflow_function(best_position)), new_best_position, best_position))

            ############################################################################################################
            # Social calculations
            ############################################################################################################

            # Create a Tensorflow constant for the social constant vector
            social_constant = tf.constant(self.social_constant, dtype=tf.float64)

            # Use a stochastic matrix instead of scalar for better stochasticity
            stochastic_matrix = tf.random_uniform([self.population_size, dimensions], 0, 1, dtype=tf.float64)

            # Scale the stochastic matrix by the social constant vector
            social_stochastic_matrix = tf.multiply(social_constant, stochastic_matrix)

            # Calculate the social factor
            social_factor = tf.multiply(social_stochastic_matrix, tf.subtract(best_position, position_matrix))

            ############################################################################################################
            # Gradient calculations
            ############################################################################################################

            # Create a Tensorflow constant for the gradient constant vector
            gradient_constant = tf.constant(self.gradient_constant, dtype=tf.float64)

            # Calculate the gradient for each position row vector in the position_matrix
            gradients = tf.gradients(result_vector, position_matrix)[0]

            # Clip the gradient by the global norm
            ratio = tf.constant(2.5, dtype=tf.float64)
            clipped_gradients = tf.clip_by_global_norm([gradients], ratio)[0][0]

            # Multiply the clipped gradients by the gradient constant vector
            scaled_gradients = tf.multiply(clipped_gradients, gradient_constant)

            # Calculate the gradient factor by clipping the gradients to the social factor matrix
            gradient_factor = tf.clip_by_value(scaled_gradients, -tf.abs(social_factor), tf.abs(social_factor))

            ############################################################################################################
            # Position update calculations
            ############################################################################################################

            # Decrease inertia linearly and calculate momentum
            next_inertia = inertia.assign(tf.subtract(inertia, step_size))
            momentum = tf.multiply(next_inertia, tf.gather(pv_tensor, 1))

            # Calculate the new velocity
            velocity = tf.scatter_update(pv_tensor, 1, tf.add(momentum, tf.subtract(social_factor, gradient_factor)))

            # Clip the velocity to the velocity max
            clipped_velocity = tf.clip_by_value(tf.gather(velocity, 1), -self.velocity_max, self.velocity_max)

            # Add the previous positions to the velocity to get the new positions
            update_positions = tf.scatter_add(pv_tensor, 0, clipped_velocity)

            # Return the best result, the best particle, and the particle matrix
            optimization_result = (self._function_class.get_tensorflow_function(best_position_update),
                                   best_position_update, tf.gather(update_positions, 0), result_vector)

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
                    graph.update_convergence_visualizer(result)

                # Capture iterative data
                if (i + 1) % (iterative_scalar * self.report_frequency) == 0:
                    iterative_scalar += 1
                    self.iterative_results.append((i + 1, result[0]))

            self._global_best_result = result[0]
            self._global_best_position = result[1]

            if self.verbose:
                writer.close()
