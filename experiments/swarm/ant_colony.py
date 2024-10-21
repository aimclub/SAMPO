import numpy as np

from swarm.evolution_history import EvolutionHistory
from swarm.ant_pheromones import AntPheromones

class AntColony:
    
    def __init__(self, fitness_function, n_tasks, 
                 n_per_iteration=25, n_bins=5, n_subbins=5, seed=0):

        self.fitness_function = fitness_function
        self.n_tasks = n_tasks
        self.n_per_iteration = n_per_iteration
        self.n_bins = n_bins
        self.n_subbins = n_subbins
        self.random = np.random.default_rng(seed=seed)
        
        self.pheromones = AntPheromones(n_tasks, n_bins, n_subbins)
        self.history = EvolutionHistory()

    def weighted_choice(self, array, weights):
        # TODO why would there be all zeros?
        if (weights == 0).all():
            print("[All weights for weighted choice are zeros]")
            return self.random.choice(array)
        else:
            probabilities = weights / np.sum(weights)
            return self.random.choice(array, p=probabilities)
    
    @staticmethod
    def split_to_bins(array, k=5):
        bins = np.quantile(array, np.linspace(1, 0, k+1))
        bins[0] = np.inf  # small fix for borders
        return np.digitize(array, bins=bins) - 1
    
    def get_one_solution(self):
        """Get one solution based on current pheromones"""
        
        pheromone_matrix, subpheromone_matrix = self.pheromones.get_pheromones()
        priority_pairs = []
        for i in range(0, self.n_tasks):
            chosen_bin = self.weighted_choice(
                np.arange(self.n_bins), 
                pheromone_matrix[i]
            )
            chosen_bin = int(chosen_bin)
            
            chosen_subbin = self.weighted_choice(
                np.arange(self.n_subbins), 
                subpheromone_matrix[i][chosen_bin]
            )
            chosen_subbin = int(chosen_subbin)
            
            priority_pairs.append((chosen_bin, chosen_subbin))
        
        fitness = self.fitness_function(priority_pairs)
        return dict(
            priority_pairs=priority_pairs, 
            fitness=fitness
        )
    
    def get_one_generation(self):
        """Get pheromones from one generation"""
        
        for i in range(self.n_per_iteration):
            self.history.add_solution(
                self.get_one_solution()
            )
                
        fitness_list, priority_pairs_list = self.history.get_generation_stats()
        total_pheromone, total_subpheromone = \
        self.calculate_pheromones(fitness_list, priority_pairs_list)
        
        self.history.end_generation()
        return (total_pheromone, total_subpheromone)
    
    def calculate_pheromones(self, fitness_list, priority_pairs_list):
        """Get amount of pheromones to add"""
        total_pheromone, total_subpheromone = \
            self.pheromones.get_zeros_pheromones()
        
        fitness_bins = self.split_to_bins(fitness_list, k=5)
        for quantile_n, priority_pairs in zip(fitness_bins, priority_pairs_list):
            pheromone_amount = (quantile_n)**2 * np.sign(quantile_n)

            for node, value in enumerate(priority_pairs):
                total_pheromone[node, value[0]] += pheromone_amount
                total_subpheromone[node, value[0], value[1]] += pheromone_amount
        
        return (total_pheromone, total_subpheromone)
    
    def update_pheromones(self, update_subpheromones=True):
        """Update pheromones"""
        new_pheromones, new_subpheromones = self.get_one_generation()
        if not update_subpheromones:
            new_subpheromones = None
        self.pheromones.add_pheromones(new_pheromones, new_subpheromones)




