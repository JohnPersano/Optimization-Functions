from optimization_functions import Sphere
from research_data import ResearchData
from research_data import SpawnDomain

sample_size = 1
epochs = 100
velocity_max = 100
gpu = False
spawn_domain = SpawnDomain.ASYMMETRIC

ResearchData(Sphere(100),
             sample_size,
             epochs,
             velocity_max,
             gpu,
             spawn_domain,
             gpo_population=20,
             gpo_constants=(1, 2),
             gpo_inertia=0.0,

             pso_swarm=20,
             pso_constants=(1, 2),
             pso_inertia=.7
             ).run_samples()


