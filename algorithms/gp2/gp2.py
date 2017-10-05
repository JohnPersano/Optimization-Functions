import cProfile
import io
import os
import pstats
import re

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern

from algorithms.gpo.gpo import GPO
from optimization_functions import Sphere, Rastrigin

np.random.seed(1)


def funct(x):
    return GPO(epochs=1000,
               population_size=36,
               spawn_range=(-100, 100),
               gradient_constant=x[0],
               social_constant=x[1],
               velocity_max=100,
               inertia=x[2],
               hardware=GPO.Hardware.CPU,
               report_frequency=25,
               verbose=False,
               plot=False).optimize(Rastrigin(30)).best_result


def gp():
    # ----------------------------------------------------------------------
    #  First the noiseless case
    # X = np.array([[2, 2, 2], [4, 4, 4], [5, 5, 5]])
    X = np.random.uniform(.1, 3, (10, 3))

    y = []
    for x in X:
        y.append(funct(x))

    print(y)

    # Instanciate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=Matern(), n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    x = np.random.uniform(0, 3, (1000, 3))
    y_pred, sigma = gp.predict(x, return_std=True)
    min = np.argmin(y_pred, 0)

    print(y_pred)

    for i in range(25):
        X = np.vstack([X, x[min]])
        y.append(funct(x[min]))

        gp.fit(X, y)
        y_pred, sigma = gp.predict(x, return_std=True)
        min = np.argmin(y_pred, 0)
        print(funct(x[min]))
        print(x[min])

gp()


# # Save the benchmark to a temporary binary file
# cProfile.runctx('gp()', globals(), locals(), "output/benchmark_binary")
#
# # Write all stats to the log file, return None if an error occurs
# try:
#     log_file = open("output/logs/benchmark-log.txt", 'w')
#
#     # Open the temporary benchmark binary as a stats file
#     string_stream = io.StringIO()
#     stats = pstats.Stats("output/benchmark_binary", stream=string_stream)
#     stats.print_stats()
#
#     log_file.write(string_stream.getvalue())
#     log_file.close()
#
#     regex_match = re.search('([0-9]*) function calls .*in ([0-9]*.([0-9]*)?) seconds', string_stream.getvalue())
#     function_calls = int(regex_match.group(1))
#     runtime = float(regex_match.group(2))
#
#     print("\n\n\n")
#     print(function_calls)
#     print(runtime)
# except IOError as exception:
#     pass
