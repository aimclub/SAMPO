import numpy as np

class EvolutionHistory:
    """Tracking history of evolution"""
    
    def __init__(self):
        self.history_fitness = []
        self.best_fitness = -np.inf
        self.best_solution = None
    
    def add_generation(self, fitness_array, solution_array):
        self.history_fitness.append(fitness_array)
        
        this_generation_best = max(fitness_array)
        if this_generation_best > self.best_fitness:
            self.best_fitness = this_generation_best
            self.best_solution = solution_array[np.argmax(fitness_array)]
        
    def get_stats(self):
        n_generations = len(self.history_fitness)
        generation_best = [
            np.max(g)
            for g in self.history_fitness
        ]
        generation_mean = [
            np.mean(g)
            for g in self.history_fitness
        ]
        
        best_so_far = np.maximum.accumulate(generation_best)
        best_overall = np.max(best_so_far)
        
        return dict(
            n_generations=n_generations, 
            generation_best=generation_best, 
            generation_mean=generation_mean, 
            best_so_far=best_so_far,
            best_overall=best_overall
        )

    @staticmethod
    def flatten_2d(array):
        return [
            element
            for sub_array in array
            for element in sub_array
        ]

    def get_last_k_generations_fitness(self, k=10):
        return self.flatten_2d(self.history_fitness[-k:])

