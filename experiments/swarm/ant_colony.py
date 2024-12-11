import numpy as np
from typing import Callable

class AntColony:

    def __init__(self,
                 fitness_function: Callable,
                 pheromones,
                 history
                ):
        self.fitness_function = fitness_function
        self.pheromones = pheromones
        self.history = history


    def generate_one_generation(self, size: int, sort_by_fitness: bool = True) -> tuple[list, list]:
        """Create one generation"""
        # maybe Generation class would be better, but lists are ok for now
        # usually priority and fitness values are needed in separate
        priorities_array = []
        fitness_array = []
        for _ in range(size):
            # get a path and fitness of this path
            priorities = self.pheromones.generate_one_path()
            fitness = self.fitness_function(priorities)
            # append to generation
            priorities_array.append(priorities)
            fitness_array.append(fitness)
        
        priorities_array, fitness_array = np.array(priorities_array), np.array(fitness_array)

        if sort_by_fitness:
            order = np.argsort(fitness_array)
            priorities_array, fitness_array = priorities_array[order], fitness_array[order]

        return priorities_array, fitness_array

    # maybe can be splitted, but let's keep all logic in one place for now
    def simple_evolution(self, 
                         n_generations: int = 1000,
                         n_per_generation: int = 100) -> None:
        
        for generation_number in range(n_generations):
            # create a generation, add it to history
            # size = (5 * n_per_generation) if (generation_number == 0) else n_per_generation
            priorities_array, fitness_array = self.generate_one_generation(size=n_per_generation, sort_by_fitness=False)
            self.history.add_generation(fitness_array, priorities_array)
            
            # get pheromone values for the generation
            # for more stable values, use last few generation for quantiles
            reference_array = self.history.get_last_k_generations_fitness(k=10)
            
            # updating pheromones
            self.pheromones.update_pheromones(priorities_array, fitness_array, reference_array)

            # break condition
            # -- todo




