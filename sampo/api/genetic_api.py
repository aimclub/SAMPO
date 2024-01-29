from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from deap import base, creator

from sampo.schemas import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec

ChromosomeType = tuple[np.ndarray, np.ndarray, np.ndarray, ScheduleSpec, np.ndarray]

class FitnessFunction(ABC):
    """
    Base class for description of different fitness functions.
    """

    @abstractmethod
    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> float:
        """
        Calculate the value of fitness function of the chromosome.
        It is better when value is less.
        """
        ...


# create class FitnessMin, the weights = -1 means that fitness - is function for minimum

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)
Individual = creator.Individual

