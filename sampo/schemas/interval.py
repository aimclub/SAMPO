import random
from abc import abstractmethod, ABC
from dataclasses import dataclass
from random import Random
from typing import Optional

from sampo.schemas.serializable import AutoJSONSerializable

# to work with float and avoid errors due to inaccuracy
EPS = 1e5
 # TODO: describe the constant 
INF = float("inf")
 # TODO: describe the constant 
MINUS_INF = float("-inf")


# TODO: Take out common parts, deal with types, rethink intervals


class Interval(AutoJSONSerializable['BaseReq'], ABC):
    """
    A class for generating random numbers from a given boundary and distribution.
    """

    @abstractmethod
    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """
        Returns a random float in the interval boundary according to the distribution of the class
        :param rand: object for generating random numbers
        :return kind: the random float
        """
        ...

    @abstractmethod
    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """
        Returns a random int in the interval boundary according to the distribution of the class
        :param rand: object for generating random numbers
        :return kind: the random int
        """
        ...


@dataclass(frozen=True)
class IntervalUniform(Interval):
    """
    Implementation for uniform distribution
    :param min_val: left border for the interval
    :param max_val: right  border for the interval
    :param rand: object for generating random numbers with, if you want to use a randomizer with a determined seed
    """
    min_val: float
    max_val: float
    rand: Optional[Random] = Random()

    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """
        Returns a random float in the interval boundary according to the distribution of the class
        :param rand: object for generating random numbers, if you want to use a randomizer with a determined seed
        :return kind: the random float
        """
        rand = rand or self.rand
        return rand.uniform(self.min_val, self.max_val)

    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """
        Returns a random int in the interval boundary according to the distribution of the class
        :param rand: object for generating random numbers, if you want to use a randomizer with a determined seed
        :return kind: the random int
        """
        rand = rand or self.rand
        value = round(rand.uniform(self.min_val, self.max_val))
        # TODO fix cast math.inf to int
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value


@dataclass(frozen=True)
class IntervalGaussian(Interval):
    """
    Implementation for Gaussian distribution
    :param mean: mean for the distribution
    :param sigma: variance for the distribution
    :param min_val: left border for the interval
    :param max_val: right  border for the interval
    :param rand: object for generating random numbers with, if you want to use a randomizer with a determined seed
    """
    mean: float
    sigma: float
    min_val: Optional[float] = MINUS_INF
    max_val: Optional[float] = INF
    rand: Optional[Random] = Random()

    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """
        Returns a random float in the interval boundary according to the distribution of the class
        :param rand: object for generating random numbers
        :return kind: the random float
        """
        rand = rand or self.rand
        return rand.gauss(self.mean, self.sigma)

    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """
        Returns a random int in the interval boundary according to the distribution of the class
        :param rand: object for generating random numbers
        :return kind: the random int
        """
        rand = rand or self.rand
        value = round(rand.gauss(self.mean, self.sigma))
        # TODO fix cast math.inf to int
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value
