"""
Copyright John Persano 2017

File name:      np_pso.py
Description:    Particle Swarm Optimization written for NumPy.
Commit history:
                - 04/22/2017 - Initial version
"""

import inspect
import numpy as np
from optimization_functions import OptimizationFunction
from utility.benchmark import Benchmark
from utility.convergence_visualizer import ConvergenceVisualizer
from utility.log import Log
from utility.report import Report


class N_PSO:
    """
    Particle Swarm Optimization algorithm.
    """

    @staticmethod
    def __get_parameter_dimensions(cost_function):
        """
        Gets the parameter dimensions for a cost function.

        :param cost_function: The cost function
        :return: Tuple (num_dimens, dimens_list)
        """

        # Inspect the signature of the function to get its parameters
        parameters = inspect.signature(cost_function.get_numpy_function).parameters

        param_list = []
        param_dimens = cost_function.dimensions

        # Iterate through the parameters of the function
        for key, value in parameters.items():
            param_list.append((key, 1))

        return param_dimens, param_list

    def __init__(self,
                 epochs: int = 500,
                 swarm_size: int = 30,
                 velocity_max: int = 10,
                 spawn_range: tuple = (-100, 100),
                 constants: tuple = (1.5, 2.0),
                 inertia: tuple = (0.9, 0.4),
                 iterative_report_frequency: int = 25,
                 plot: bool = False,
                 verbose: bool = False):
        """
        Creates a new instance of PSO with any optional parameters.

        :param epochs: Maximum number of iterations
        :param swarm_size: Amount of particles in the swarm
        :param velocity_max: Maximum velocity of the particles
        :param spawn_range: Particle spawn range as a tuple (from, to)
        :param constants: Optimization constants as a tuple (cognitive, social)
        :param inertia: Inertia value of the velocity calculation (start, end)
        :param plot: True if should plot particles
        :param verbose: True if should show output
        """

        # Initialize functional class variables
        self.epochs = epochs
        self.swarm_size = swarm_size
        self.velocity_max = velocity_max
        self.spawn_range = spawn_range
        self.cognitive_constant = constants[0]
        self.social_constant = constants[1]
        self.inertia = inertia[0]
        self.inertia_step = (inertia[0] - inertia[1]) / epochs

        # Initialize non-functional class variables
        self.iterative_report_frequency = iterative_report_frequency
        self.plot = plot
        self.verbose = verbose

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

        # The benchmark will run the actual optimization
        benchmark_results = Benchmark().run(self._start)

        # An issue occurred with benchmark results
        if not benchmark_results:
            Log.error("There was an issue getting benchmark results.")
            return Report("NumPy PSO", "CPU", self._function_class.get_name(), self._global_best_result,
                          self._global_best_particle, self.iterative_results, Benchmark.ERROR_CODE,
                          Benchmark.ERROR_CODE)

        # No issues have occurred with the benchmark, show the full report
        return Report("NumPy PSO", "CPU", self._function_class.get_name(), self._global_best_result,
                      self._global_best_particle, self.iterative_results, benchmark_results.runtime,
                      benchmark_results.function_calls)

    def _start(self) -> None:
        """
        Private function, should not be used directly.
        Called by the Benchmark class to start the optimization.
        :return: None
        """

        if self.verbose:
            Log.info("Starting NumPy PSO optimization")

        ################################################################################################################
        # Initialization
        ################################################################################################################

        # Get the dimensionality of the cost function
        dimensions = self.__get_parameter_dimensions(self._function_class)

        # Create a uniformly distributed matrix of size [swarm_size, dimensions] within the spawn range
        particle_matrix = np.random.uniform(self.spawn_range[0], self.spawn_range[1],
                                            size=(self.swarm_size, dimensions[0])).astype('f')

        # Create a cognitive matrix which will initially be a copy of the particle matrix
        cognitive_matrix = np.copy(particle_matrix)

        # Create a velocity matrix of size [swarm_size, dimensions] with zeros
        velocity_matrix = np.zeros((self.swarm_size, dimensions[0]))

        # Select the global best as the first item
        self._global_best_particle = np.copy(particle_matrix[0])
        self._global_best_result = self._function_class.get_numpy_function(self._global_best_particle)

        # Get the number of rows (particles) in the matrix
        row_count = particle_matrix.shape[0]

        # Create a result vector of size [swarm_size] with zeros
        result_vector = np.zeros(row_count)

        # Graph the optimization if the plot flag is True
        graph = None
        if self.plot:
            graph = ConvergenceVisualizer(self._function_class)

        iterative_scalar = 1
        # Iterate through the epochs
        for epoch in range(0, self.epochs):

            # Iterate through each particle
            for row in range(0, row_count):

                # Calculate the cost for a row of values
                particle_result = self._function_class.get_numpy_function(particle_matrix[row])
                particle_best = self._function_class.get_numpy_function(cognitive_matrix[row])

                # Insert the result into the result vector
                result_vector[row] = particle_best

                # If this result is better than the current global best, update the global best
                if particle_result < self._global_best_result:
                    self._global_best_result = particle_result
                    self._global_best_particle = np.copy(particle_matrix[row])

                # If the result is better than the particle's best, update the particles best
                if particle_result < particle_best:
                    cognitive_matrix[row] = np.copy(particle_matrix[row])

            ############################################################################################################
            # Calculations
            ############################################################################################################

            # Uncomment to use stochastic scalars instead of matrices
            # cognitive_random = np.random.rand()
            # social_random = np.random.rand()

            # Create random matrices of size [population_size, dimensions] between zero and one
            cognitive_random = np.random.rand(*velocity_matrix.shape)
            social_random = np.random.rand(*velocity_matrix.shape)

            # Multiply the stochastic matrices by the corresponding constants
            cognitive_scaled_matrix = self.cognitive_constant * cognitive_random
            social_scaled_matrix = self.social_constant * social_random

            # Calculate the cognitive and social factors
            cognitive_factor = cognitive_scaled_matrix * (cognitive_matrix - particle_matrix)
            social_factor = social_scaled_matrix * (self._global_best_particle - particle_matrix)

            # Decrease inertia linearly
            self.inertia -= self.inertia_step

            # Calculate the new velocity
            velocity_matrix = (self.inertia * velocity_matrix) + cognitive_factor + social_factor
            velocity_matrix = np.clip(velocity_matrix, -self.velocity_max, self.velocity_max)

            # Add the previous particles to the velocity to get the new particles
            particle_matrix += velocity_matrix

            # Update the graph if the plot flag is True
            if self.plot:
                graph.update_convergence_visualizer(((self._global_best_result,
                                                      self._global_best_particle, particle_matrix, result_vector)))

            # Capture iterative data
            if (epoch + 1) % (iterative_scalar * self.iterative_report_frequency) == 0:
                iterative_scalar += 1
                self.iterative_results.append((epoch + 1, self._global_best_result))

            if self.verbose:
                Log.info("Best result: {}".format(self._global_best_result))
                Log.info("Best vector: {}".format(self._global_best_particle))
