from research.optimization_functions import Sphere, Beale, CrossTray, Easom, Griewank, Matya, Rastrigin, Rosenbrock, \
    StyblinskiTang, ThreehumpCamel, Booth


# noinspection PyListCreation
def get_complete_optimization_list():

    research_samples = []

    # Sphere function
    research_samples.append((Sphere(10), 1000, 20))
    research_samples.append((Sphere(20), 1500, 20))
    research_samples.append((Sphere(30), 2000, 20))
    research_samples.append((Sphere(10), 1000, 40))
    research_samples.append((Sphere(20), 1500, 40))
    research_samples.append((Sphere(30), 2000, 40))
    research_samples.append((Sphere(10), 1000, 80))
    research_samples.append((Sphere(20), 1500, 80))
    research_samples.append((Sphere(30), 2000, 80))

    # Beale function
    research_samples.append((Beale(2), 1000, 20))
    research_samples.append((Beale(2), 1000, 40))
    research_samples.append((Beale(2), 1000, 80))

    # Booth function
    research_samples.append((Booth(2), 1000, 20))
    research_samples.append((Booth(2), 1000, 40))
    research_samples.append((Booth(2), 1000, 80))

    # Crosstray function
    research_samples.append((CrossTray(2), 1000, 20))
    research_samples.append((CrossTray(2), 1000, 40))
    research_samples.append((CrossTray(2), 1000, 80))

    # Easom function
    research_samples.append((Easom(2), 1000, 20))
    research_samples.append((Easom(2), 1000, 40))
    research_samples.append((Easom(2), 1000, 80))

    # Griewank function
    research_samples.append((Griewank(10), 1000, 20))
    research_samples.append((Griewank(20), 1500, 20))
    research_samples.append((Griewank(30), 2000, 20))
    research_samples.append((Griewank(10), 1000, 40))
    research_samples.append((Griewank(20), 1500, 40))
    research_samples.append((Griewank(30), 2000, 40))
    research_samples.append((Griewank(10), 1000, 80))
    research_samples.append((Griewank(20), 1500, 80))
    research_samples.append((Griewank(30), 2000, 80))

    # Matya function
    research_samples.append((Matya(2), 1000, 20))
    research_samples.append((Matya(2), 1000, 40))
    research_samples.append((Matya(2), 1000, 80))

    # Rastrigin function
    research_samples.append((Rastrigin(10), 1000, 20))
    research_samples.append((Rastrigin(20), 1500, 20))
    research_samples.append((Rastrigin(30), 2000, 20))
    research_samples.append((Rastrigin(10), 1000, 40))
    research_samples.append((Rastrigin(20), 1500, 40))
    research_samples.append((Rastrigin(30), 2000, 40))
    research_samples.append((Rastrigin(10), 1000, 80))
    research_samples.append((Rastrigin(20), 1500, 80))
    research_samples.append((Rastrigin(30), 2000, 80))

    # Rosenbrock function
    research_samples.append((Rosenbrock(10), 1000, 20))
    research_samples.append((Rosenbrock(20), 1500, 20))
    research_samples.append((Rosenbrock(30), 2000, 20))
    research_samples.append((Rosenbrock(10), 1000, 40))
    research_samples.append((Rosenbrock(20), 1500, 40))
    research_samples.append((Rosenbrock(30), 2000, 40))
    research_samples.append((Rosenbrock(10), 1000, 80))
    research_samples.append((Rosenbrock(20), 1500, 80))
    research_samples.append((Rosenbrock(30), 2000, 80))

    # StyblinskiTang function
    research_samples.append((StyblinskiTang(10), 1000, 20))
    research_samples.append((StyblinskiTang(20), 1500, 20))
    research_samples.append((StyblinskiTang(30), 2000, 20))
    research_samples.append((StyblinskiTang(10), 1000, 40))
    research_samples.append((StyblinskiTang(20), 1500, 40))
    research_samples.append((StyblinskiTang(30), 2000, 40))
    research_samples.append((StyblinskiTang(10), 1000, 80))
    research_samples.append((StyblinskiTang(20), 1500, 80))
    research_samples.append((StyblinskiTang(30), 2000, 80))

    # Threehump camel function
    research_samples.append((ThreehumpCamel(2), 1000, 20))
    research_samples.append((ThreehumpCamel(2), 1000, 40))
    research_samples.append((ThreehumpCamel(2), 1000, 80))

    return research_samples
