from algorithms.gpo.gpo import GPO
from optimization_functions import Rastrigin, Griewank, Sphere
from research_data import ResearchData, SpawnDomain

# ResearchData(Rastrigin(20),
#              sample_size=3,
#              spawn_range=SpawnRange.ASYMMETRIC,
#              epochs=1000,
#              points=(36, 40),
#              gpo_constants=(1.75, 2.25, 3, 5),
#              pso_constants=(2, 2),
#              inertia=(0.4, 0.4),
#              velocity_max=100).run_samples()



re = GPO(spawn_range=Rastrigin(0).get_asymmetric_domain(),
         social_linspace=(2, 2.3),
         gradient_linspace=(2, 4),
         inertia=(.4, .2),
         epochs=1000).optimize(Rastrigin(10))
print(re)
