from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np
from deap import base, creator

from sampo.schemas import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec

ChromosomeType = tuple[np.ndarray, np.ndarray, np.ndarray, ScheduleSpec, np.ndarray]

class ScheduleGenerationScheme(Enum):
    Parallel = 'Parallel'
    Serial = 'Serial'


class FitnessFunction(ABC):
    """
    Base class for description of different fitness functions.
    """

    @abstractmethod
    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) \
            -> tuple[int | float]:
        """
        Calculate the value of fitness function of the chromosome.
        It is better when value is less.
        """
        ...


# create class FitnessMin, the weights = -1 means that fitness - is function for minimum

# creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
# creator.create('Individual', list, fitness=creator.FitnessMin)
# Individual = creator.Individual

class Individual(list):
    def __init__(self, individual_fitness_constructor: Callable[[], base.Fitness], chromosome: ChromosomeType):
        super().__init__(chromosome)
        self.fitness = individual_fitness_constructor()

    @staticmethod
    def prepare(individual_fitness_constructor: Callable[[], base.Fitness]) -> Callable[[ChromosomeType], list]:
        """
        Returns the constructor of Individual prepared to use in Genetic algorithm
        """
        return partial(Individual, individual_fitness_constructor)

