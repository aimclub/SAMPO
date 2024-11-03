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


    def generate_one_generation(self, generation_size: int) -> tuple:
        """Create one generation"""
        # maybe Generation class would be better, but lists are ok for now
        # usually priority and fitness values are needed in separate
        priorities_array = []
        fitness_array = []
        for _ in range(generation_size):
            # get a path and fitness of this path
            priorities = self.pheromones.generate_one_path()
            fitness = self.fitness_function(priorities)
            # append to generation
            priorities_array.append(priorities)
            fitness_array.append(fitness)
        
        return (priorities_array, fitness_array)


    def simple_evolution(self, 
                         n_generations: int = 1000, 
                         n_per_generation: int = 25) -> None:
        for generation_number in range(n_generations):
            # create a generation, add it to history
            priorities_array, fitness_array = self.generate_one_generation(n_per_generation)
            self.history.add_generation(fitness_array, priorities_array)
            # get pheromone values for the generation
            # for more stable values, use last few generation for quantiles
            reference_array = self.history.get_last_k_generations_fitness(k=20)
            pheromone_amount = self.quantile_based_reward(fitness_array, reference_array)
            # updating pheromones
            self.pheromones.add_pheromones(priorities_array, pheromone_amount)
            self.pheromones.evaporate()

    @staticmethod
    def quantile_based_reward(fitness_array: list[float], 
                              reference_array: list[float], 
                              k_groups: int = 5) -> float:
        # get quantiles of fitness of last few generations
        bins = np.quantile(reference_array, np.linspace(0, 1, k_groups+1))
        # small fix for borders
        bins[0] = -np.inf
        bins[-1] = np.inf
        quantiles = np.digitize(fitness_array, bins=bins) - 1  # -1 to start from zero
        # reward = number of the group, squared
        # seems ok for now, balance between better rewards for top
        # but also not too harsh for others
        pheromone_amount = quantiles ** 2
        return pheromone_amount

