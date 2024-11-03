import numpy as np 

class BinsPheromones:
    
    def __init__(self, 
                 n_tasks: int, 
                 n_bins: int, 
                 n_sub_bins: int, 
                 random, 
                 start_amount: float = 100, 
                 evaporation_rate:float = 0.15):
        # number of choices to make
        self.n_tasks = n_tasks
        # number of bins and subbins for each choice
        self.n_bins = n_bins
        self.n_sub_bins = n_sub_bins

        # start from uniform (for now) 
        self.bins_pheromones = \
            start_amount * np.ones((n_tasks, n_bins))
        self.sub_bins_pheromones = \
            start_amount * np.ones((n_tasks, n_bins, n_sub_bins))

        # evaporation rate controls speed of changing old pheromones to new
        self.evaporation_rate = evaporation_rate
        # to make sure the seed is right everywhere, random is an attribute
        self.random = random

    def weighted_choice(self, array: list, weights: list):
        probabilities = weights / np.sum(weights)
        choice = self.random.choice(array, p=probabilities)
        return choice

    def generate_one_path(self) -> list[tuple[int, int]]:
        """Create one random path using current pheromones"""

        priority_list = []
        for i in range(0, self.n_tasks):
            # main priority value
            chosen_bin = self.weighted_choice(
                np.arange(self.n_bins), 
                self.bins_pheromones[i]
            )
            chosen_bin = int(chosen_bin)
            # sub priority value
            chosen_sub_bin = self.weighted_choice(
                np.arange(self.n_sub_bins), 
                self.sub_bins_pheromones[i][chosen_bin]
            )
            chosen_sub_bin = int(chosen_sub_bin)
            
            priority_list.append((chosen_bin, chosen_sub_bin))

        return priority_list

    def add_pheromones(self, 
                       path_list: list, 
                       pheromone_amount_list: list[float]) -> None:
        """
        Add new pheromones based on one generation
        path_list: list of paths 
        pheromone_amount_list: list of amounts to add
        """
        new_pheromone = (0 * self.bins_pheromones)
        new_sub_pheromone = (0 * self.sub_bins_pheromones)

        for path, pheromone_amount in zip(path_list, pheromone_amount_list):

            for node, value in enumerate(path):
                new_pheromone[node, value[0]] += pheromone_amount                
                new_sub_pheromone[node, value[0], value[1]] += pheromone_amount

        self.bins_pheromones += (new_pheromone * self.evaporation_rate)
        self.sub_bins_pheromones += (new_sub_pheromone * self.evaporation_rate)

    def evaporate(self) -> None:
        """Evaporate current pheromones"""
        self.bins_pheromones *= (1 - self.evaporation_rate)
        self.sub_bins_pheromones *= (1 - self.evaporation_rate)


