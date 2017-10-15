"""
Copyright John Persano 2017

File name:      optimization_function.py
Description:    A suite of benchmark functions for optimization algorithms.
Commit history:
                - 04/23/2017: Initial version
"""

import tensorflow as tf
import math

from utility.log import Log


class OptimizationFunction:
    """
    A template class to provide a consistent API for optimization test functions.
    """

    def __init__(self, dimensions=2):
        self.dimensions = dimensions

    def get_name(self):
        return self.__class__.__name__ + " function"

    @staticmethod
    def get_tensorflow_function(tensor):
        return None

    @staticmethod
    def get_numpy_function(vector):
        return None

    @staticmethod
    def get_symmetric_domain():
        return None

    @staticmethod
    def get_asymmetric_domain():
        return None


class CustomFunction(OptimizationFunction):
    pass


class Beale(OptimizationFunction):
    """
    Beale function.
    
    Type:           Multimodal
    Dimensions:     2
    Global minimum: 0 at (3, 0.5)
    Domain          [-4.5, 4.5]
    Reference:      https://www.sfu.ca/~ssurjano/beale.html
    """

    def __init__(self, dimensions=2):
        super(Beale, self).__init__()
        self.dimensions = 2

        if dimensions != 2:
            Log.warning("Beale's function can only have two dimensions.")

    @staticmethod
    def get_tensorflow_function(data):
        x, y = tf.unstack(data)

        first_term = tf.square(tf.add(tf.subtract(tf.constant(1.5, dtype=tf.float64), x), tf.multiply(x, y)))
        second_term = tf.square(tf.add(tf.subtract(tf.constant(2.25, dtype=tf.float64), x),
                                       tf.multiply(x, tf.square(y))))
        third_term = tf.square(tf.add(tf.subtract(tf.constant(2.625, dtype=tf.float64), x),
                                      tf.multiply(x, tf.pow(y, tf.constant(3, dtype=tf.float64)))))

        return tf.add(first_term, tf.add(second_term, third_term))

    @staticmethod
    def get_numpy_function(vector):
        x = vector[0]
        y = vector[1]
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

    @staticmethod
    def get_symmetric_domain():
        return -4.5, 4.5

    @staticmethod
    def get_asymmetric_domain():
        return 2.25, 4.5


class Booth(OptimizationFunction):
    """
    Booth function.

    Type:           Multimodal
    Dimensions:     2
    Global minimum: 0 at (1, 3)
    Domain          [-10, 10]
    Reference:      https://www.sfu.ca/~ssurjano/booth.html
    """

    def __init__(self, dimensions=2):
        super(Booth, self).__init__()
        self.dimensions = 2

        if dimensions != 2:
            Log.warning("Booth's function can only have two dimensions.")

    @staticmethod
    def get_tensorflow_function(data):
        x, y = tf.unstack(data)
        first_term = tf.square(tf.subtract(tf.add(x, tf.multiply(tf.constant(2, dtype=tf.float64), y)),
                                           tf.constant(7, dtype=tf.float64)))
        second_term = tf.square(tf.subtract(tf.add(tf.multiply(tf.constant(2, dtype=tf.float64), x), y),
                                            tf.constant(5, dtype=tf.float64)))
        return tf.add(first_term, second_term)

    @staticmethod
    def get_numpy_function(vector):
        x = vector[0]
        y = vector[1]
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    @staticmethod
    def get_symmetric_domain():
        return -10, 10

    @staticmethod
    def get_asymmetric_domain():
        return 5, 10


class CrossTray(OptimizationFunction):
    """
    CrossTray function.

    Type:           Multimodal
    Dimensions:     n
    Global minimum: -2.06261 at (1.3491, 1.3491)
    Domain          [-10, 10]
    Reference:      https://www.sfu.ca/~ssurjano/crossit.html
    """

    def __init__(self, dimensions=2):
        super(CrossTray, self).__init__()
        self.dimensions = 2

        if dimensions != 2:
            Log.warning("CrossTray function can only have two dimensions.")

    @staticmethod
    def get_tensorflow_function(data):
        x, y = tf.unstack(data)
        return -0.0001 * (abs(
            tf.sin(x) * tf.sin(y) * tf.exp(abs(100 - (tf.sqrt(x ** 2 * y ** 2)) / math.pi))) + 1) ** 0.1

    @staticmethod
    def get_numpy_function(vector):
        x = vector[0]
        y = vector[1]
        return -0.0001 * (abs(math.sin(x) * math.sin(y) *
                              math.exp(abs(100 - (math.sqrt(x ** 2 * y ** 2)) / math.pi))) + 1) ** 0.1

    @staticmethod
    def get_symmetric_domain():
        return -10, 10

    @staticmethod
    def get_asymmetric_domain():
        return 5, 10


class Easom(OptimizationFunction):
    """
    Easom function.

    Type:           Unimodal
    Dimensions:     2
    Global minimum: 0 at (3.14, 3.14)
    Domain          [-100, 100]
    Reference:      https://www.sfu.ca/~ssurjano/easom.html
    """

    def __init__(self, dimensions=5):
        super(Easom, self).__init__()
        self.dimensions = 2

        if dimensions != 2:
            Log.warning("Easom's function can only have two dimensions.")

    @staticmethod
    def get_tensorflow_function(data):
        x, y = tf.unstack(data)
        return -tf.cos(x) * tf.cos(y) * tf.exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))

    @staticmethod
    def get_numpy_function(vector):
        x = vector[0]
        y = vector[1]
        return -math.cos(x) * math.cos(y) * math.exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))

    @staticmethod
    def get_symmetric_domain():
        return -100, 100

    @staticmethod
    def get_asymmetric_domain():
        return 50, 100


class Griewank(OptimizationFunction):
    """
    Griewank function.

    Type:           Multimodal
    Dimensions:     n
    Global minimum: 0 at (0, ..., 0)
    Domain          [-600, 600]
    Reference:      https://www.sfu.ca/~ssurjano/griewank.html
    """

    def __init__(self, dimensions=2):
        super(Griewank, self).__init__()
        self.dimensions = dimensions

    @staticmethod
    def get_tensorflow_function(data):
        columns = tf.unstack(data)

        summation = tf.reduce_sum(tf.divide(tf.square(data), tf.constant(4000, dtype=tf.float64)))

        product = 1
        for i in range(0, len(columns) - 1):
            product *= tf.cos(
                columns[i] / tf.sqrt(tf.add(tf.constant(i, dtype=tf.float64), tf.constant(1, dtype=tf.float64))))
        return tf.add(tf.subtract(summation, product), tf.constant(1, dtype=tf.float64))

    @staticmethod
    def get_numpy_function(vector):
        summation = 0
        for item in vector:
            summation += (item ** 2) / 4000

        product = 1
        for i in range(0, len(vector) - 1):
            product *= math.cos(vector[i] / math.sqrt(i + 1))
        return summation - product + 1

    @staticmethod
    def get_symmetric_domain():
        return -600, 600

    @staticmethod
    def get_asymmetric_domain():
        return 300, 600


class Matya(OptimizationFunction):
    """
    Matya function.

    Type:           Multimodal
    Dimensions:     2
    Global minimum: 0 at (0, 0)
    Domain          [-10, 10]
    Reference:      https://www.sfu.ca/~ssurjano/matya.html
    """

    def __init__(self, dimensions=2):
        super(Matya, self).__init__()
        self.dimensions = 2

        if dimensions != 2:
            Log.warning("Matya's function can only have two dimensions.")

    @staticmethod
    def get_tensorflow_function(data):
        x, y = tf.unstack(data)
        first_term = tf.multiply(tf.constant(0.26, dtype=tf.float64), tf.add(tf.square(x), tf.square(y)))
        second_term = tf.multiply(tf.constant(0.48, dtype=tf.float64), tf.multiply(x, y))
        return tf.subtract(first_term, second_term)

    @staticmethod
    def get_numpy_function(vector):
        x = vector[0]
        y = vector[1]
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    @staticmethod
    def get_symmetric_domain():
        return -10, 10

    @staticmethod
    def get_asymmetric_domain():
        return 5, 10


class Rastrigin(OptimizationFunction):
    """
    Rastrigin function.

    Type:           Multimodal
    Dimensions:     n
    Global minimum: 0 at (0, ..., 0)
    Domain          [-5.12, 5.12]
    Reference:      https://www.sfu.ca/~ssurjano/rastr.html
    """

    def __init__(self, dimensions=2):
        super(Rastrigin, self).__init__()
        self.dimensions = dimensions

    @staticmethod
    def get_tensorflow_function(data):
        pi_parameter = tf.constant(math.pi, dtype=tf.float64)
        cosine_parameter = tf.multiply(tf.multiply(tf.constant(2, dtype=tf.float64), pi_parameter), data)
        cosine_term = tf.multiply(tf.constant(10, dtype=tf.float64), tf.cos(cosine_parameter))
        summation_term = tf.subtract(tf.square(data), cosine_term)
        return tf.reduce_sum(tf.add(summation_term, 10))

    @staticmethod
    def get_numpy_function(vector):
        summation = 0
        for item in vector:
            summation += item ** 2 - 10 * math.cos(2 * math.pi * item) + 10
        return summation

    @staticmethod
    def get_symmetric_domain():
        return -5.12, 5.12

    @staticmethod
    def get_asymmetric_domain():
        return 2.56, 5.12


class MisplacedRastrigin(OptimizationFunction):
    """
    Rastrigin function.

    Type:           Multimodal
    Dimensions:     n
    Global minimum: 0 at (-15, ..., -15)
    Domain          [0, 20]
    Reference:      Custom
    """

    def __init__(self, dimensions=2):
        super(MisplacedRastrigin, self).__init__()
        self.dimensions = dimensions

    @staticmethod
    def get_tensorflow_function(data):
        fifteen_constant = tf.constant(15, dtype=tf.float64)
        added_data = tf.add(data, fifteen_constant)
        pi_parameter = tf.constant(math.pi, dtype=tf.float64)
        cosine_parameter = tf.multiply(tf.multiply(tf.constant(2, dtype=tf.float64), pi_parameter), added_data)
        cosine_term = tf.multiply(tf.constant(10, dtype=tf.float64), tf.cos(cosine_parameter))
        summation_term = tf.subtract(tf.square(added_data), cosine_term)
        return tf.reduce_sum(tf.add(summation_term, 10))

    @staticmethod
    def get_numpy_function(vector):
        summation = 0
        for item in vector:
            summation += (item - 15) ** 2 - 10 * math.cos(2 * math.pi * (item - 15)) + 10
        return summation

    @staticmethod
    def get_symmetric_domain():
        return 0, 20

    @staticmethod
    def get_asymmetric_domain():
        return 17.5, 20



class Rosenbrock(OptimizationFunction):
    """
    Rosenbrock function.

    Type:           Unimodal
    Dimensions:     n
    Global minimum: 0 at (1, ..., 1)
    Domain          [-30, 30]
    Reference:      https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(self, dimensions=2):
        super(Rosenbrock, self).__init__()
        self.dimensions = dimensions

    # @staticmethod
    # def get_tensorflow_function(data):
    #     columns = tf.unstack(data)
    #
    #     one = tf.constant(1, dtype=tf.float64)
    #
    #     index_summation = (tf.constant(1), tf.constant(0.0, dtype=tf.float64))
    #
    #     # noinspection PyUnusedLocal
    #     def condition(index, summation):
    #         return tf.less(tf.cast(index, tf.float64), tf.subtract(tf.cast(tf.shape(columns)[0], tf.float64), one))
    #
    #     def body(index, summation):
    #         x_i = tf.gather(columns, index)
    #         x_ip1 = tf.gather(columns, tf.add(index, 1))
    #
    #         first_term = tf.square(tf.subtract(x_ip1, tf.square(x_i)))
    #         second_term = tf.square(tf.subtract(x_i, one))
    #         summand = tf.add(tf.multiply(tf.constant(100.0, dtype=tf.float64), first_term), second_term)
    #
    #         return tf.add(index, 1), tf.add(summation, summand)
    #
    #     return tf.while_loop(condition, body, index_summation)[1]

    @staticmethod
    def get_tensorflow_function(data):
        vector = tf.unstack(data)

        summation = 0
        for i in range(1, len(vector) - 1):
            summation += (100 * (vector[i + 1] - vector[i] ** 2) ** 2 + (vector[i] - 1) ** 2)
        return summation

    @staticmethod
    def get_numpy_function(vector):
        summation = 0
        for i in range(1, len(vector) - 1):
            summation += (100 * (vector[i + 1] - vector[i] ** 2) ** 2 + (vector[i] - 1) ** 2)
        return summation

    @staticmethod
    def get_symmetric_domain():
        return -30, 30

    @staticmethod
    def get_asymmetric_domain():
        return 15, 30


class Sphere(OptimizationFunction):
    """
    Sphere function.

    Type:           Convex
    Dimensions:     n
    Global minimum: 0 at (0, ..., 0)
    Domain          [-5.12, 5.12]
    Reference:      https://www.sfu.ca/~ssurjano/spheref.html
    """

    def __init__(self, dimensions=5):
        super(Sphere, self).__init__()
        self.dimensions = dimensions

    @staticmethod
    def get_tensorflow_function(data):
        return tf.reduce_sum(tf.square(data))

    @staticmethod
    def get_numpy_function(vector):
        summation = 0
        for item in vector:
            summation += item ** 2
        return summation

    @staticmethod
    def get_symmetric_domain():
        return -10, 10

    @staticmethod
    def get_asymmetric_domain():
        return 5, 10


class StyblinskiTang(OptimizationFunction):
    """
    Styblinski Tang function.

    Type:           Multimodal
    Dimensions:     n
    Global minimum: -39.16599 * n at (-2.903534, ..., -2.903534)
    Domain          [-5, 5]
    Reference:      https://www.sfu.ca/~ssurjano/stybtang.html
    """

    def __init__(self, dimensions=2):
        super(StyblinskiTang, self).__init__()
        self.dimensions = dimensions

    @staticmethod
    def get_tensorflow_function(data):
        first_term = tf.pow(data, 4)
        second_term = tf.multiply(tf.constant(16, dtype=tf.float64), tf.square(data))
        third_term = tf.multiply(tf.constant(5, dtype=tf.float64), data)
        summation = tf.reduce_sum(tf.add(tf.subtract(first_term, second_term), third_term))
        result = tf.divide(tf.divide(summation, tf.constant(2, dtype=tf.float64)), tf.cast(tf.size(data), tf.float64))
        return tf.add(tf.constant(39.16599, dtype=tf.float64), result)

    @staticmethod
    def get_numpy_function(vector):
        summation = 0
        for item in vector:
            summation += (item ** 4) - (16 * item ** 2) + (5 * item)
        return ((summation / 2) / len(vector)) + 39.16599

    def get_global_minima(self):
        return [2.903534] * self.dimensions, -39.16617

    @staticmethod
    def get_symmetric_domain():
        return -5, 5

    @staticmethod
    def get_asymmetric_domain():
        return 2.5, 5


class ThreehumpCamel(OptimizationFunction):
    """
    ThreehumpCamel function.

    Type:           Multimodal
    Dimensions:     n
    Global minimum: 0 at (0, ..., 0)
    Domain          [-5, 5]
    Reference:      https://www.sfu.ca/~ssurjano/camel3.html
    """

    def __init__(self, dimensions=2):
        super(ThreehumpCamel, self).__init__()
        self.dimensions = 2

        if dimensions != 2:
            Log.warning("ThreehumpCamel function can only have two dimensions.")

    @staticmethod
    def get_tensorflow_function(data):
        x, y = tf.unstack(data)
        return 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6 / 6) + x * y + y ** 2

    @staticmethod
    def get_numpy_function(vector):
        x = vector[0]
        y = vector[1]
        return 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6 / 6) + x * y + y ** 2

    @staticmethod
    def get_symmetric_domain():
        return -5, 5

    @staticmethod
    def get_asymmetric_domain():
        return 2.5, 5
