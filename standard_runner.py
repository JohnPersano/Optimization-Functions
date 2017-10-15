from research.research_data import ResearchData
from research.research_data import SpawnDomain
from utility.research_helper import get_complete_optimization_list

research_samples = []

sample_size = 50
velocity_max = 100
inertia = 0.4
gpu = False
spawn_domain = SpawnDomain.SYMMETRIC

for cost_function, epochs, population in get_complete_optimization_list():

    ResearchData(cost_function,
                 sample_size,
                 epochs,
                 velocity_max,
                 gpu,
                 spawn_domain,
                 gpo_population=population,
                 gpo_constants=(2, 2),
                 gpo_inertia=inertia,

                 pso_swarm=population,
                 pso_constants=(2, 2),
                 pso_inertia=inertia
                 ).run_samples()


