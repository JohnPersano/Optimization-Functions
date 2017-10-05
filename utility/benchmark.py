"""
Copyright John Persano 2017

File name:      benchmark.py
Description:    A benchmarking utility for optimization algorithms.
Commit history:
                - 03/17/2017: Initial version
                - 03/19/2017: Fixed error where a bad log path would cause the optimization not to run
                - 04/23/2017: Added ensure directories code
"""

from utility.log import Log
import cProfile
import io
import os
import pstats
import re


class Benchmark:
    """
    Handles optimization algorithm benchmarking.
    """

    # Specify an error code number
    ERROR_CODE = -1

    def __init__(self):
        self.function_calls = Benchmark.ERROR_CODE
        self.runtime = Benchmark.ERROR_CODE

    def run(self, cost_function, log_path: str="output/logs/benchmark-log.txt"):
        """
        Run the benchmark utility on a function.
        :param cost_function: The function to run the benchmark on
        :param log_path: A path for the log file
        :return: self
        """

        # Ensure all directories exist
        if not os.path.exists("output"):
            os.mkdir("output")
        if not os.path.exists("output/logs"):
            os.mkdir("output/logs")
        if not os.path.exists("output/results"):
            os.mkdir("output/results")
        if not os.path.exists("output/tensorboard"):
            os.mkdir("output/tensorboard")

        # Save the benchmark to a temporary binary file
        cProfile.runctx('cost_function()', globals(), locals(), "output/benchmark_binary")

        # Write all stats to the log file, return None if an error occurs
        try:
            log_file = open(log_path, 'w')

            # Open the temporary benchmark binary as a stats file
            string_stream = io.StringIO()
            stats = pstats.Stats("output/benchmark_binary", stream=string_stream)
            stats.print_stats()

            log_file.write(string_stream.getvalue())
            log_file.close()
        except IOError as exception:
            Log.error(exception)
            return None

        # Regex to pull the function calls and runtime out of the stats text
        regex_match = re.search('([0-9]*) function calls .*in ([0-9]*.([0-9]*)?) seconds', string_stream.getvalue())
        self.function_calls = int(regex_match.group(1))
        self.runtime = float(regex_match.group(2))

        # Try to cleanup the benchmark binary file
        try:
            os.remove("output/benchmark_binary")
        except IOError as exception:
            Log.error(exception)

        # Return self for one line initialization
        return self
