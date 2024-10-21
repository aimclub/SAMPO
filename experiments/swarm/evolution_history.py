import numpy as np

class EvolutionHistory:
    """Tracking history of evolution"""
    
    def __init__(self):
        self.generations = []
        self.current_generation = []
    
    def add_solution(self, solution):
        self.current_generation.append(solution)
        
    def end_generation(self):
        self.generations.append(self.current_generation)
        self.current_generation = []
        
    def get_stats(self):
        n_generations = len(self.generations)
        mins = [
            np.min([solution["fitness"] for solution in generation])
            for generation in self.generations
        ]
        mean = [
            np.mean([solution["fitness"] for solution in generation])
            for generation in self.generations
        ]
        cmins = np.minimum.accumulate(mins)
        return (n_generations, mins, mean, cmins)
    
    def get_generation_stats(self):
        return (
            [i["fitness"] for i in self.current_generation],
            [i["priority_pairs"] for i in self.current_generation]
        )

    def get_all_solutions_flat(self):
        flat_list = [
            solution
            for generation in self.generations
            for solution in generation
        ]
        return flat_list
        
