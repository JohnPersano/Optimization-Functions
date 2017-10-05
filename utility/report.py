"""
Copyright John Persano 2017

File name:      report.py
Description:    A report text formatter for optimization algorithms.
Commit history:
                - 03/17/2017: Initial version
"""


class Report:
    """
    A text formatter class used to organize optimization algorithm performance data.

    To use this class, initialize an instance with data and print or write it directly to a file.
    """

    def __init__(self, variant, hardware, function_name, best_result, best_parameters, iterative_results,
                 runtime, function_calls):
        """
        Creates a new instance of Report with any optional parameters.
        :param variant: Name of the optimization algorithm used
        :param hardware: The hardware the algorithm was run on
        :param function_name: Name of the cost function
        :param best_result: The best result
        :param best_parameters: The best parameters
        :param iterative_results: A list of iterative results
        :param runtime: The total runtime of the algorithm
        :param function_calls: Number of function calls used by the algorithm
        """
        self.variant = variant
        self.hardware = hardware
        self.function_name = function_name
        self.best_result = best_result
        self.best_parameters = best_parameters
        self.iterative_results = iterative_results
        self.function_calls = function_calls
        self.runtime = runtime

    def __str__(self):
        variant = "Variant: {}\n".format(self.variant)
        hardware = "Hardware: {}\n".format(self.hardware)
        best_result = "Best result: {}\n".format(self.best_result)
        best_particle = "Best parameters: {}\n".format(self.best_parameters)

        iterative_results = "Iterative results:\n"
        for result in self.iterative_results:
            iterative_results += "\tIteration {}, best result: {}\n".format(result[0], result[1])

        function_calls = "Function calls: {}\n".format(self.function_calls)
        runtime = "Runtime: {} seconds\n".format(self.runtime)

        return variant + hardware + best_result + best_particle + iterative_results + function_calls + runtime
