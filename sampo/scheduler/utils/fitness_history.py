import json
from typing import Any, Iterable
from itertools import pairwise


class FitnessHistory:
    """Class to track fitness values during evolution"""

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def update_history(self, population: Iterable = [], pareto_front: Iterable = [], comment: str = ""):
        self.history.append({
            "population_fitness": [i.fitness.values for i in population],

            # while you could get pareto-front from fitness later,
            # it is easier to save fitness values from hall-of-fame
            "pareto_front_fitness": [i.fitness.values for i in pareto_front],

            # comments about how current generation was created
            # or if there are any changes that need attention
            "comment": comment
        })

    def save_fitness_history(self, path: str):
        """Save current fitness history to JSON file"""
        try:
            with open(path, "w") as json_file:
                json.dump(self.history, json_file)
        except Exception as e:
            print(f"Error while saving history to {path}: {e}")


class FitnessHistorySummary:

    def __init__(self, path: str):
        with open(path, "r") as json_file:
            history = json.load(json_file)

        self.population_fitness_history = [generation["population_fitness"] for generation in history]
        self.pareto_front_fitness_history = [generation["pareto_front_fitness"] for generation in history]
        self.comments = [generation["comment"] for generation in history]

    def agg_population_fitness(self, agg_function):
        # List is used in case population size or number of objectives changes (np.array needs the same shape)
        return [
            agg_function(population_fitness, axis=0)
            for population_fitness in self.population_fitness_history
        ]

    def agg_pareto_front_fitness(self, agg_function):
        # List is used in case population size or number of objectives changes (np.array needs the same shape)
        return [
            agg_function(pareto_front_fitness, axis=0)
            for pareto_front_fitness in self.pareto_front_fitness_history
        ]

    def uniqueness_scores(self):
        """Calculate uniqueness of fitness values in population"""
        return [
            len(set(map(tuple, population_fitness))) / len(population_fitness)
            for population_fitness in self.population_fitness_history
        ]

    def pareto_front_ratios(self):
        """Calculate what ratio of the population is in pareto front"""
        return [
            len(pareto_front_fitness) / len(population_fitness)
            for population_fitness, pareto_front_fitness in zip(
                self.population_fitness_history,
                self.pareto_front_fitness_history
            )
        ]

    def population_shifts(self):
        """Calculate how much the population has changed compared to previous generation"""
        return [
            sum(1 for i in new_fitness if i not in old_fitness) / len(new_fitness)
            for old_fitness, new_fitness in pairwise(self.population_fitness_history)
        ]

    def pareto_front_shifts(self):
        """Calculate how much the pareto front has changed compared to previous generation"""
        return [
            sum(1 for i in new_fitness if i not in old_fitness) / len(new_fitness)
            for old_fitness, new_fitness in pairwise(self.pareto_front_fitness_history)
        ]
