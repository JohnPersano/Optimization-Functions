"""
Copyright John Persano 2017

File name:      g_pso.py
Description:    Gradient Particle Swarm Optimization.
Commit history:
                - 04/22/2017 - Initial version
"""

import tensorflow as tf
from enum import Enum
from optimization_functions import OptimizationFunction
from utility.benchmark import Benchmark
from utility.convergence_visualizer import ConvergenceVisualizer
from utility.log import Log
from utility.report import Report


class GPSO:
    """
    Gradient Particle Swarm Optimization.    
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
                 swarm_size: int = 30,
                 spawn_range: tuple = (-100, 100),
                 constants: tuple = (0.15, 2.0, 1.5),
                 velocity_max: int = 100.0,
                 inertia: float = 0.7,
                 hardware: Hardware = Hardware.CPU,
                 report_frequency: int = 25,
                 verbose: bool = False,
                 plot: bool = False):
        """
        Creates a new instance of G-PSO with any optional parameters.
        
        :param epochs: Maximum number of iterations
        :param spawn_range: Population spawn range as a tuple (from, to)
        :param constants: Optimization constants as a tuple (cognitive, social, gradient)
        :param velocity_max: Maximum iterative velocity
        :param inertia: Inertia value of the velocity calculation
        :param hardware: Run optimization on CPU or GPU
        :param report_frequency: Number of epochs between report sampling
        :param verbose: True if should show output
        :param plot: True if should plot each iteration
        """

        # Initialize functional class variables
        self.epochs = epochs
        self.swarm_size = swarm_size
        self.spawn_range = spawn_range
        self.cognitive_constant = constants[0]
        self.social_constant = constants[1]
        self.gradient_constant = constants[2]
        self.velocity_max = velocity_max
        self.inertia = inertia
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

        # The benchmark will run the optimization, decide whether to use cognitive or non-cognitive graph
        if self.cognitive_constant > 0:
            benchmark_results = Benchmark().run(self._start_with_cognitive)
        else:
            benchmark_results = Benchmark().run(self._start_without_cognitive)

        # An issue occurred with benchmark results,
        if not benchmark_results:
            Log.error("There was an issue getting benchmark results.")
            return Report("G-PSO", self.hardware.name, self._function_class.get_name(),
                          self._global_best_result,
                          self._global_best_particle, self.iterative_results, Benchmark.ERROR_CODE,
                          Benchmark.ERROR_CODE)

        # No issues have occurred with the benchmark, show the full report
        return Report("G-PSO", self.hardware.name, self._function_class.get_name(), self._global_best_result,
                      self._global_best_particle, self.iterative_results, benchmark_results.runtime,
                      benchmark_results.function_calls)

    def _start_with_cognitive(self) -> None:
        """
        Private function, should not be used directly.
        Called by the Benchmark class to start the optimization with a cognitive graph.
        :return: None
        """

        # If verbose, print out the type of graph being used
        if self.verbose:
            Log.debug("Using cognitive graph")

        # Select which hardware to use, default to CPU
        if self.hardware == GPSO.Hardware.GPU_1:
            device = "/gpu:0"
            if self.verbose:
                Log.info("Using GPU 1 for optimization")
        elif self.hardware == GPSO.Hardware.GPU_2:
            device = "/gpu:1"
            if self.verbose:
                Log.info("Using GPU 2 for optimization")
        else:
            if self.verbose:
                Log.info("Using CPU for optimization")
            device = "/cpu:0"

        # Use the appropriate hardware
        with tf.device(device):

            ############################################################################################################
            # Initialization
            ############################################################################################################

            # Calculate the dimensionality of the cost function
            dimensions = self._function_class.dimensions

            # Initialize the best particle as a random value
            best_particle = tf.Variable(tf.random_uniform([dimensions],
                                                          self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            # Create a uniformly distributed matrix of size [population_size, dimensions] within the spawn range
            particle_matrix = tf.Variable(tf.random_uniform([self.swarm_size, dimensions],
                                                            self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            ############################################################################################################
            # Cognitive calculations
            ############################################################################################################

            # Initialize the cognitive matrix as a copy of the particle matrix
            cognitive_matrix = tf.Variable(particle_matrix.initialized_value())

            # Calculate the result vector for the particle matrix
            result_vector = tf.map_fn(self._function_class.get_tensorflow_function, particle_matrix)

            # Calculate the result vector for the cognitive matrix
            cognitive_result_vector = tf.map_fn(self._function_class.get_tensorflow_function, cognitive_matrix)

            # Create a boolean mask and update the cognitive matrix with any better particles
            cognitive_mask = tf.less(result_vector, cognitive_result_vector)
            updated_cognitive_matrix = cognitive_matrix.assign(tf.where(cognitive_mask,
                                                                        particle_matrix, cognitive_matrix))

            # Calculate the cognitive factor using a stochastic matrix
            cognitive_stochastic_matrix = tf.multiply(tf.constant(self.cognitive_constant, dtype=tf.float64),
                                                      tf.random_uniform([self.swarm_size, dimensions],
                                                                        minval=0, maxval=1, dtype=tf.float64))
            cognitive_factor = tf.multiply(cognitive_stochastic_matrix,
                                           (tf.subtract(updated_cognitive_matrix, particle_matrix)))

            ############################################################################################################
            # Social calculations
            ############################################################################################################

            # Calculate the social factor using a stochastic matrix
            social_stochastic_matrix = tf.multiply(tf.constant(self.social_constant, dtype=tf.float64),
                                                   tf.random_uniform([self.swarm_size, dimensions],
                                                                     minval=0, maxval=1, dtype=tf.float64))
            social_factor = tf.multiply(social_stochastic_matrix, tf.subtract(best_particle, particle_matrix))

            # Calculate the gradient factor using a stochastic matrix
            gradient_stochastic_matrix = tf.multiply(tf.constant(self.gradient_constant, dtype=tf.float64),

                                                     tf.random_uniform([self.swarm_size, dimensions],
                                                                       minval=0, maxval=1, dtype=tf.float64))

            ############################################################################################################
            # Gradient calculations
            ############################################################################################################

            # Calculate the gradient values of the particle matrix
            gradient_matrix = tf.gradients(tf.map_fn(self._function_class.get_tensorflow_function, particle_matrix),
                                           particle_matrix)[0]

            # Calculate the standard deviation
            gradient_sq_difference = tf.square(tf.subtract(gradient_matrix, tf.reduce_mean(gradient_matrix)))
            standard_deviation = tf.sqrt(tf.reduce_mean(gradient_sq_difference))

            # The standard deviation divisor should never be zero
            standard_deviation_divisor = tf.add(standard_deviation, 1)

            # Regularize the gradient matrix by dividing it by the gradient standard deviation
            regularized_gradients = tf.divide(gradient_matrix, standard_deviation_divisor)

            # Calculate the gradient factor using a stochastic matrix
            gradient_factor = tf.multiply(gradient_stochastic_matrix, regularized_gradients)

            ############################################################################################################
            # Position update calculations
            ############################################################################################################

            # Calculate the new velocity
            calculate_velocity = tf.subtract(tf.add(cognitive_factor, social_factor), gradient_factor)

            # Clip the velocity to the max velocity value
            update_velocity = tf.clip_by_value(calculate_velocity, -self.velocity_max, self.velocity_max)

            # Update the particle matrix with the new velocities
            update_particles = particle_matrix.assign(tf.add(particle_matrix, update_velocity))

            # Calculate a new result vector after velocity has been applied to the particle matrix
            new_result_vector = tf.map_fn(self._function_class.get_tensorflow_function, update_particles)

            # Get the best particle from the particle matrix
            new_best_particle = tf.gather(update_particles, tf.argmin(new_result_vector, 0))

            # If the current best particle is better than the last best particle, update the best particle
            best_particle_update = best_particle.assign(tf.where(
                tf.less(self._function_class.get_tensorflow_function(new_best_particle),
                        self._function_class.get_tensorflow_function(best_particle)), new_best_particle, best_particle))

            # Return the best result, the best particle, and the particle matrix
            optimization_result = (self._function_class.get_tensorflow_function(best_particle_update),
                                   best_particle_update, particle_matrix, new_result_vector)

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
            self._global_best_particle = result[1]

            if self.verbose:
                writer.close()

    def _start_without_cognitive(self) -> None:
        """
        Private function, should not be used directly.
        Called by the Benchmark class to start the optimization without a cognitive graph.
        :return: None
        """

        # If verbose, print out the type of graph being used
        if self.verbose:
            Log.debug("Using non-cognitive graph")

        # Select which hardware to use, default to CPU
        if self.hardware == GPSO.Hardware.GPU_1:
            device = "/gpu:0"
            if self.verbose:
                Log.info("Using GPU 1 for optimization")
        elif self.hardware == GPSO.Hardware.GPU_2:
            device = "/gpu:1"
            if self.verbose:
                Log.info("Using GPU 2 for optimization")
        else:
            if self.verbose:
                Log.info("Using CPU for optimization")
            device = "/cpu:0"

        # Use the appropriate hardware
        with tf.device(device):

            ############################################################################################################
            # Initialization
            ############################################################################################################

            # Calculate the dimensionality of the cost function
            dimensions = self._function_class.dimensions

            # Initialize the best particle as a random value
            best_particle = tf.Variable(tf.random_uniform([dimensions],
                                                          self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            # Create a uniformly distributed matrix of size [population_size, dimensions] within the spawn range
            particle_matrix = tf.Variable(tf.random_uniform([self.swarm_size, dimensions],
                                                            self.spawn_range[0], self.spawn_range[1], dtype=tf.float64))

            ############################################################################################################
            # Social calculations
            ############################################################################################################

            # Calculate the social factor using a stochastic matrix
            social_stochastic_matrix = tf.multiply(tf.constant(self.social_constant, dtype=tf.float64),
                                                   tf.random_uniform([self.swarm_size, dimensions],
                                                                     minval=0, maxval=1, dtype=tf.float64))
            social_factor = tf.multiply(social_stochastic_matrix, tf.subtract(best_particle, particle_matrix))

            # Calculate the gradient factor using a stochastic matrix
            gradient_stochastic_matrix = tf.multiply(tf.constant(self.gradient_constant, dtype=tf.float64),

                                                     tf.random_uniform([self.swarm_size, dimensions],
                                                                       minval=0, maxval=1, dtype=tf.float64))

            ############################################################################################################
            # Gradient calculations
            ############################################################################################################

            # Calculate the gradient values of the particle matrix
            gradient_matrix = tf.gradients(tf.map_fn(self._function_class.get_tensorflow_function, particle_matrix),
                                           particle_matrix)[0]

            # Calculate the standard deviation
            gradient_sq_difference = tf.square(tf.subtract(gradient_matrix, tf.reduce_mean(gradient_matrix)))
            standard_deviation = tf.sqrt(tf.reduce_mean(gradient_sq_difference))

            # The standard deviation divisor should never be zero
            standard_deviation_divisor = tf.add(standard_deviation, 1)

            # Regularize the gradient matrix by dividing it by the gradient standard deviation
            regularized_gradients = tf.divide(gradient_matrix, standard_deviation_divisor)

            # Calculate the gradient factor using a stochastic matrix
            gradient_factor = tf.multiply(gradient_stochastic_matrix, regularized_gradients)

            ############################################################################################################
            # Position update calculations
            ############################################################################################################

            # Calculate the new velocity
            calculate_velocity = tf.subtract(social_factor, gradient_factor)

            # Clip the velocity to the max velocity value
            update_velocity = tf.clip_by_value(calculate_velocity, -self.velocity_max, self.velocity_max)

            # Update the particle matrix with the new velocities
            update_particles = particle_matrix.assign(tf.add(particle_matrix, update_velocity))

            # Calculate a new result vector after velocity has been applied to the particle matrix
            new_result_vector = tf.map_fn(self._function_class.get_tensorflow_function, update_particles)

            # Get the best particle from the particle matrix
            new_best_particle = tf.gather(update_particles, tf.argmin(new_result_vector, 0))

            # If the current best particle is better than the last best particle, update the best particle
            best_particle_update = best_particle.assign(tf.where(
                tf.less(self._function_class.get_tensorflow_function(new_best_particle),
                        self._function_class.get_tensorflow_function(best_particle)), new_best_particle, best_particle))

            # Return the best result, the best particle, and the particle matrix
            optimization_result = (self._function_class.get_tensorflow_function(best_particle_update),
                                   best_particle_update, particle_matrix, new_result_vector)

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
            self._global_best_particle = result[1]

            if self.verbose:
                writer.close()
