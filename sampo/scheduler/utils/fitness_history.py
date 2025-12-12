import json
from typing import Any, Iterable
from itertools import pairwise
import numpy as np


class FitnessHistory:
    """Recording fitness values during evolution"""

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def update(self, population: Iterable = [], pareto_front: Iterable = [], comment: str = ""):
        self.history.append({
            "population_fitness": [i.fitness.values for i in population],

            # while you could get pareto-front from fitness later,
            # it is easier to save fitness values from hall-of-fame
            "pareto_front_fitness": [i.fitness.values for i in pareto_front],

            # comments about how current generation was created
            # or if there are any changes that need attention
            "comment": comment
        })

    def save_to_json(self, path: str):
        """Save current fitness history to JSON file"""
        # JSON format is a bit more flexible
        try:
            with open(path, "w") as json_file:
                json.dump(self.history, json_file)
        except Exception as e:
            print(f"Error while saving history to {path}: {e}")


class FitnessHistorySummary:
    """Functions for creating summary of evolution"""

    def __init__(self, path: str):
        with open(path, "r") as json_file:
            history = json.load(json_file)

        self.population_history = [generation["population_fitness"] for generation in history]
        self.pareto_front_history = [generation["pareto_front_fitness"] for generation in history]
        self.comments = [generation["comment"] for generation in history]

    def get_fitness_means(self, only_pareto: bool = False):
        """Average fitness in population for each generation and objective"""
        # mean might be better than median, outliers matter
        data = self.pareto_front_history if only_pareto else self.population_history
        return [
            np.mean(fitness_values, axis=0)
            for fitness_values in data
        ]

    def get_pareto_to_population_ratios(self):
        """What part of the population is in pareto front"""
        data = zip(self.population_history, self.pareto_front_history)
        return [
            len(pareto_front_fitness) / len(population_fitness)
            for population_fitness, pareto_front_fitness in data
        ]

    def get_generation_shifts(self, only_pareto: bool = False):
        """How much the population has changed compared to previous generation"""
        data = self.pareto_front_history if only_pareto else self.population_history
        data = pairwise(data)
        return [
            sum(1 for value in new_fitness if value not in old_fitness) / len(new_fitness)
            for old_fitness, new_fitness in data
        ]

    def get_uniqueness_scores(self, only_pareto: bool = False):
        """Uniqueness of fitness values in population"""
        data = self.pareto_front_history if only_pareto else self.population_history
        return [
            len(set(map(tuple, fitness_values))) / len(fitness_values)
            for fitness_values in data
        ]
