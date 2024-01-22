from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

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

