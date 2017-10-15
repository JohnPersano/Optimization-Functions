"""
Copyright John Persano 2017

File name:      research_data.py
Description:    Automated research for optimization algorithms.
Commit history:
                - 04/23/2017: Initial version
"""
import csv
import os
import datetime
import numpy as np
from algorithms.pso.tf_pso import TFPSO
from algorithms.gpo.gpo import GPO
from utility.log import Log
from algorithms.gd.gd import GradientDescent
import matplotlib.pyplot as plt
import math

plt.rcParams["figure.figsize"] = (10, 10)


class SpawnDomain:
    SYMMETRIC = 0
    ASYMMETRIC = 1


class Algorithms:
    GPO = 0
    PSO = 1
    GDE = 2


# noinspection PyUnresolvedReferences,PyAttributeOutsideInit
class ResearchData:
    def _handle_gpo_parameters(self, kwargs):

        def validate_gpo_params(gpo_population, gpo_constants, gpo_inertia):

            # Validate the population parameter
            if not gpo_population or not gpo_population >= 1:
                Log.warning("GPO population must be greater than or equal to '1'.")
                return False

            # Validate the constants tuple
            if not type(gpo_constants) is tuple or len(gpo_constants) != 2:
                Log.warning("GPO constants must be a tuple in the form (gradient, social).")
                return False
            else:
                if not gpo_constants[0] >= 0:
                    Log.warning("GPO constants gradient factor must be greater than or equal to '0'.")
                    return False
                if not gpo_constants[1] >= 0:
                    Log.warning("GPO constants social factor must be greater than or equal to '0'.")
                    return False

            # Validate inertia
            if not gpo_inertia >= 0:
                Log.warning("GPO inertia must be greater than or equal to '0'.")
                return False

            # All validations passed
            return True

        self.gpo_population = kwargs.get('gpo_population', None)
        self.gpo_constants = kwargs.get('gpo_constants', None)
        self.gpo_inertia = kwargs.get('gpo_inertia', None)
        self.gpo_meta_opt = kwargs.get('gpo_meta_opt', None)

        # The explicit parameters were not all set
        if self.gpo_population is None or self.gpo_constants is None or self.gpo_inertia is None:

            # The explicit parameters were not all set and neither was the gpo_meta_opt flag
            if not self.gpo_meta_opt:

                # One or more of the parameters was set but not all of them
                if self.gpo_population or self.gpo_constants or self.gpo_inertia:
                    Log.warning("Not all GPO parameters were set. "
                                "Ensure gpo_population, gpo_constants, and gpo_inertia are set if "
                                "explicit parameters were intended.")
                Log.info("GPO parameters not configured, not using GPO.")
                return False

            # The explicit parameters were not set but the gpo_meta_opt flag was
            else:
                Log.debug("GPO parameters will be meta-optimized.")
                return True

        # The explicit parameters were all set
        else:

            # The gpo_meta_opt flag was also set
            if self.gpo_meta_opt:
                Log.error("GPO parameters cannot be specified in addition to GPO meta optimization")
                exit(1)

            # The gpo_meta_opt flag was not also set
            else:

                # Validate the explicit parameters
                if not validate_gpo_params(self.gpo_population, self.gpo_constants, self.gpo_inertia):
                    Log.error("There was an issue with GPO parameters, not using GPO.")
                    return False

                # The explicit parameters were set and they are all valid
                else:
                    Log.info("Using GPO with explicit parameters.")
                    return True

    def _handle_pso_parameters(self, kwargs):

        def validate_pso_params(pso_swarm, pso_constants, pso_inertia):

            # Validate the swarm parameter
            if not pso_swarm >= 1:
                Log.warning("PSO swarm must be greater than or equal to '1'.")
                return False

            # Validate the constants tuple
            if not type(pso_constants) is tuple or len(pso_constants) != 2:
                Log.warning("PSO constants must be a tuple in the form (cognitive, social).")
                return False
            else:
                if not pso_constants[0] >= 0:
                    Log.warning("PSO constants cognitive factor must be greater than or equal to '0'.")
                    return False
                if not pso_constants[1] >= 0:
                    Log.warning("PSO constants social factor must be greater than or equal to '0'.")
                    return False

            # Validate inertia
            if not pso_inertia >= 0:
                Log.warning("PSO inertia must be greater than or equal to '0'.")
                return False

            # All validations passed
            return True

        self.pso_swarm = kwargs.get('pso_swarm', None)
        self.pso_constants = kwargs.get('pso_constants', None)
        self.pso_inertia = kwargs.get('pso_inertia', None)

        # The explicit parameters were not all set
        if self.pso_swarm is None or self.pso_constants is None or self.pso_inertia is None:

            # One or more of the parameters was set but not all of them
            if self.pso_swarm or self.pso_constants or self.pso_inertia:
                Log.warning("Not all PSO parameters were set. "
                            "Ensure pso_swarm, pso_constants, and pso_inertia are set.")
            Log.info("PSO parameters not configured, not using PSO.")
            return False

        # The explicit parameters were all set
        else:

            # Validate the explicit parameters
            if not validate_pso_params(self.pso_swarm, self.pso_constants, self.pso_inertia):
                Log.error("There was an issue with PSO parameters, not using PSO.")
                return False

            # The explicit parameters were set and they are all valid
            else:
                Log.info("Using PSO with explicit parameters.")
                return True

    def __init__(self,
                 cost_function,
                 sample_size: int,
                 epochs: int,
                 velocity_max: int,
                 gpu: bool,
                 spawn_domain: SpawnDomain,
                 **kwargs):
        """

        """

        # General function parameters
        self.cost_function = cost_function
        self.sample_size = sample_size
        self.epochs = epochs
        self.velocity_max = velocity_max
        self.gpu = gpu
        self.spawn_range = self.cost_function.get_asymmetric_domain() if spawn_domain == SpawnDomain.ASYMMETRIC \
            else self.cost_function.get_symmetric_domain()

        self.use_gpo = self._handle_gpo_parameters(kwargs)
        self.use_pso = self._handle_pso_parameters(kwargs)
        self.use_gde = True

        # No function parameters were specified or they were all incorrect
        if not self.use_gpo and not self.use_pso and not self.use_gde:
            Log.error("At least one optimization function should be specified for research.")
            exit(1)

        # Construct a date time string
        date = datetime.datetime.now().strftime("%m-%d-%Y (%H%M)")

        # Construct the name of the result file in the format "MM/DD/YY - function_name = dimensions"
        self.folder_name = "output/results/" + date
        self.folder_name += " - " + cost_function.get_name()
        self.folder_name += " - " + cost_function.dimensions.__str__()

        # Ensure output directories exist
        if not os.path.exists("output"):
            os.mkdir("output")
        if not os.path.exists("output/results"):
            os.mkdir("output/results")
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)

    def run_samples(self):

        gpo_csv_tuples = []
        pso_csv_tuples = []
        gd_csv_tuples = []

        def run_gpo(output_file):

            best_result = math.inf          # Absolute best result of all samples
            best_point = None               # Absolute best point of all samples
            average_result = 0              # Average result of all samples
            average_runtime = 0             # Average runtime of all samples
            average_function_calls = 0      # Average function calls of all samples
            iterative_result_samples = []   # Sample number taken while the optimization is running
            iterative_result_values = []    # Sample result taken while the optimization is running
            for gpo_run in range(0, self.sample_size):
                Log.debug("GPO iteration: {} out of {}".format(gpo_run + 1, self.sample_size))

                hardware = GPO.Hardware.GPU_2 if self.gpu else GPO.Hardware.CPU

                report = GPO(social_constant=(self.gpo_constants[1]),
                             gradient_constant=(self.gpo_constants[0]),
                             epochs=self.epochs, hardware=hardware,
                             spawn_range=self.spawn_range,
                             population_size=self.gpo_population,
                             inertia=self.gpo_inertia).optimize(self.cost_function)

                average_result += report.best_result

                if report.best_result < best_result:
                    best_result = report.best_result
                    best_point = report.best_parameters.copy()

                average_runtime += report.runtime
                iterative_result_samples.append([i[0] for i in report.iterative_results].copy())
                iterative_result_values.append([i[1] for i in report.iterative_results].copy())
                average_function_calls += report.function_calls

            average_iterative_sample = np.mean(np.array(iterative_result_samples), axis=0)
            average_iterative_result = np.mean(np.array(iterative_result_values), axis=0)

            output_file.write("\tGradient Population Optimization run\n")
            output_file.write(
                "\t\tConstants: \n\t\t\tGradient: {}\n\t\t\tSocial: {} \n\t".format(self.gpo_constants[0],
                                                                                self.gpo_constants[1]))
            output_file.write("\t\tInertia: {}\n".format(self.gpo_inertia))
            output_file.write("\t\tPopulation: {}\n".format(self.gpo_population))
            output_file.write("\t\tAverage result: {}\n".format(average_result / self.sample_size))
            output_file.write("\t\tAverage runtime: {}\n".format(average_runtime / self.sample_size))
            output_file.write("\t\tIterative result average:\n")
            for i in range(0, len(average_iterative_sample)):
                gpo_csv_tuples.append((average_iterative_sample[i], average_iterative_result[i]))
                output_file.write(
                    "\t\t\tIteration: {}, Result: {}\n".format(average_iterative_sample[i],
                                                                 average_iterative_result[i]))
            output_file.write("\t\tAverage function calls: {}\n".format(average_function_calls / self.sample_size))

        def run_pso(output_file):

            result_tally = 0
            best_particle = []
            runtime = 0
            iterative_result_samples = []
            iterative_result_values = []
            function_calls = 0
            for pso_run in range(0, self.sample_size):
                Log.warning("PSO iteration: {} out of {}".format(pso_run + 1, self.sample_size))

                hardware = TFPSO.Hardware.CPU
                if self.gpu:
                    hardware = TFPSO.Hardware.GPU_2

                report = TFPSO(constants=self.pso_constants,
                               epochs=self.epochs,
                               spawn_range=self.spawn_range,
                               swarm_size=self.pso_swarm,
                               inertia=self.pso_inertia,
                               hardware=hardware).optimize(self.cost_function)

                result_tally += report.best_result
                best_particle.append(report.best_parameters.copy())
                runtime += report.runtime
                iterative_result_samples.append([i[0] for i in report.iterative_results].copy())
                iterative_result_values.append([i[1] for i in report.iterative_results].copy())
                function_calls += report.function_calls

            average_iterative_sample = np.mean(np.array(iterative_result_samples), axis=0)
            average_iterative_result = np.mean(np.array(iterative_result_values), axis=0)

            # Write the run number to the file
            output_file.write("\tParticle Swarm Optimization run\n")
            output_file.write("\t\tConstants: (Cognitive: {}, Social: {})\n".format(self.pso_constants[0],
                                                                                    self.pso_constants[1]))
            output_file.write("\t\tInertia: {}\n".format(self.pso_inertia))
            output_file.write("\t\tPopulation: {}\n".format(self.pso_swarm))
            output_file.write("\t\tAverage result: {}\n".format(result_tally / self.sample_size))
            output_file.write("\t\tAverage runtime: {}\n".format(runtime / self.sample_size))
            output_file.write("\t\tIterative result average:\n")

            for i in range(0, len(average_iterative_sample)):
                pso_csv_tuples.append((average_iterative_sample[i], average_iterative_result[i]))
                output_file.write("\t\t\tIteration: {}, Result: {}\n".format(average_iterative_sample[i],
                                                                               average_iterative_result[i]))
            output_file.write("\t\tAverage function calls: {}\n".format(function_calls / self.sample_size))

        def run_gde(output_file):

            result_tally = 0
            best_point = []
            runtime = 0
            iterative_result_samples = []
            iterative_result_values = []
            function_calls = 0
            for pso_run in range(0, self.sample_size):
                Log.warning("GDE iteration: {} out of {}".format(pso_run + 1, self.sample_size))

                hardware = GradientDescent.Hardware.CPU
                if self.gpu:
                    hardware = GradientDescent.Hardware.GPU_2

                report = GradientDescent(epochs=self.epochs,
                                         spawn_range=self.spawn_range,
                                         learning_rate=0.04,
                                         hardware=hardware).optimize(self.cost_function)

                result_tally += report.best_result
                best_point.append(report.best_parameters.copy())
                runtime += report.runtime
                iterative_result_samples.append([i[0] for i in report.iterative_results].copy())
                iterative_result_values.append([i[1] for i in report.iterative_results].copy())
                function_calls += report.function_calls

            average_iterative_sample = np.mean(np.array(iterative_result_samples), axis=0)
            average_iterative_result = np.mean(np.array(iterative_result_values), axis=0)

            # Write the run number to the file
            output_file.write("\tGradient Descent Optimization run\n")
            output_file.write("\t\tConstants: Learning Rate: 0.07 \n")
            output_file.write("\t\tAverage result: {}\n".format(result_tally / self.sample_size))
            output_file.write("\t\tAverage runtime: {}\n".format(runtime / self.sample_size))
            output_file.write("\t\tIterative result average:\n")

            for i in range(0, len(average_iterative_sample)):
                gd_csv_tuples.append((average_iterative_sample[i], average_iterative_result[i]))
                output_file.write("\t\t\tIteration: {}, Result: {}\n".format(average_iterative_sample[i],
                                                                               average_iterative_result[i]))
            output_file.write("\t\tAverage function calls: {}\n".format(function_calls / self.sample_size))


        # Create an output file to hold the results in
        output_file = open(os.path.join(self.folder_name, "Report.txt"), "w")

        # Write some preliminary data to the file
        output_file.write(self.cost_function.get_name() + " with {} dimensions\n".format(self.cost_function.dimensions))
        output_file.write("GPU\n") if self.gpu else output_file.write("CPU\n")
        output_file.write("Sample size: {}\n".format(self.sample_size))
        output_file.write("Velocity max: {}\n".format(self.velocity_max))
        output_file.write("Epochs: {}\n".format(self.epochs))

        print("\n\n")

        if self.use_gpo:
            run_gpo(output_file)

        if self.use_pso:
            run_pso(output_file)

        if self.use_gde:
            run_gde(output_file)

        csv_table = open(os.path.join(self.folder_name, "Result_Table.csv"), "w")
        csv_writer = csv.writer(csv_table, delimiter=',')
        csv_writer.writerow(['Iterations', 'GPO', 'PSO', 'GDE'])

        # X-axis is iteration, Y is result
        gpo_x = []
        gpo_y = []

        pso_x = []
        pso_y = []

        gd_x = []
        gd_y = []

        for i in range(0, len(gpo_csv_tuples)):

            csv_write_list = []

            if self.use_gpo:
                gpo_x.append(gpo_csv_tuples[i][0])
                gpo_y.append(gpo_csv_tuples[i][1])
                csv_write_list.extend([gpo_csv_tuples[i][0], gpo_csv_tuples[i][1]])

            if self.use_pso:
                pso_x.append(pso_csv_tuples[i][0])
                pso_y.append(pso_csv_tuples[i][1])
                csv_write_list.extend([pso_csv_tuples[i][1]])

            if self.use_gde:
                gd_x.append(gd_csv_tuples[i][0])
                gd_y.append(gd_csv_tuples[i][1])
                csv_write_list.extend([gd_csv_tuples[i][1]])

            # Writes CSV sheet in the following 'Iteration, GPO, PSO, GDE'. Assumes each iteration report is the same!
            csv_writer.writerow(csv_write_list)
        csv_table.close()

        # Plot and save the convergence graph
        all_y = sorted(gpo_y + pso_y + gd_y)
        max_y = all_y[-1] + (all_y[-1] * .1)

        plt.axis([gpo_x[0], gpo_x[-1], -0.1, max_y])

        if self.use_gpo:
            plt.plot(gpo_x, gpo_y, '-b', label="GPO")

        if self.use_pso:
            plt.plot(pso_x, pso_y, '--r', label="PSO")

        if self.use_gde:
            plt.plot(gd_x, gd_y, ':g', label="GD")

        plt.legend(loc='upper right')

        plt.xlabel("Epochs")
        plt.ylabel("Optimized value")

        dimensions = self.cost_function.dimensions
        if dimensions > 100:
            dimensions = "{:.2e}".format(dimensions)
        else:
            dimensions = dimensions.__str__()

        plt.title(self.cost_function.get_name().title() + " with " + dimensions + " Dimensions")

        plt.savefig(os.path.join(self.folder_name, "Convergence_Graph.png"))

        output_file.close()


