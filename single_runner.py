from algorithms.gd.gd import GradientDescent
from algorithms.gpo.gpo import GPO
from algorithms.pso.np_pso import NPPSO
from algorithms.pso.tf_pso import TFPSO
from research.optimization_functions import Booth, Sphere, Rastrigin, Griewank, MisplacedRastrigin


def gradient_descent_result():
    hardware = GradientDescent.Hardware.GPU_1 if gpu is True else GradientDescent.Hardware.CPU

    return GradientDescent(
        epochs=epochs,
        spawn_range=spawn_range,
        learning_rate=learning_rate,
        plot=plot,
        hardware=hardware).optimize(cost_function)


def tf_pso_result():
    hardware = TFPSO.Hardware.GPU_1 if gpu is True else TFPSO.Hardware.CPU

    return TFPSO(
        swarm_size=tf_pso_swarm_size,
        epochs=epochs,
        spawn_range=spawn_range,
        constants=(tf_pso_cognitive_constant, tf_pso_social_constant),
        inertia=tf_pso_inertia,
        plot=plot,
        img_save=img_save,
        hardware=hardware).optimize(cost_function)


def np_pso_result():

    return NPPSO(
        swarm_size=np_pso_swarm_size,
        epochs=epochs,
        spawn_range=spawn_range,
        constants=(np_pso_cognitive_constant, np_pso_social_constant),
        inertia=(np_pso_inertia, np_pso_inertia),
        plot=plot).optimize(cost_function)


def tf_gpo_result():
    hardware = GPO.Hardware.GPU_1 if gpu is True else GPO.Hardware.CPU

    return GPO(
        population_size=tf_gpo_population_size,
        epochs=epochs,
        spawn_range=spawn_range,
        gradient_constant=tf_gpo_gradient_constant,
        social_constant=tf_gpo_social_constant,
        inertia=tf_gpo_inertia,
        plot=plot,
        hardware=hardware).optimize(cost_function)


# General parameters
cost_function = Sphere(30)
epochs = 2000
spawn_range = cost_function.get_asymmetric_domain()
plot = False
img_save = [50, 100, 1000, 2000]
gpu = False

# Gradient descent parameters
learning_rate = 0.07
# print(gradient_descent_result())

# Tensorflow PSO parameters
tf_pso_swarm_size = 40
tf_pso_cognitive_constant = 4
tf_pso_social_constant = 2
tf_pso_inertia = 0.4
print(tf_pso_result())

# Numpy PSO parameters
np_pso_swarm_size = 40
np_pso_cognitive_constant = 2
np_pso_social_constant = 2
np_pso_inertia = 0.4
# print(np_pso_result())

# Tensorflow GPO parameters
tf_gpo_population_size = 40
tf_gpo_gradient_constant = 2
tf_gpo_social_constant = 2
tf_gpo_inertia = 0.4
#print(tf_gpo_result())
