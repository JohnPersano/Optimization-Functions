from optimization_functions import Rastrigin, Griewank, Matya, StyblinskiTang, Sphere
from research_data import ResearchData

ResearchData(Sphere(250),
             sample_size=50,
             epochs=500,

             gpo_population=36,
             gpo_constants=(2, 2, 2, 2),
             gpo_inertia=(0.4, 0.2),

             pso_swarm=100,
             pso_constants=(2, 2),
             # pso_inertia=(0.9, 0.4),
             pso_inertia=(0.4, 0.2),

             contrast_gd=True,

             velocity_max=5,
             gpu=False).run_samples()
