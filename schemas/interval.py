import random
from dataclasses import dataclass
from random import Random

from typing import Optional, Tuple, Dict, List

EPS = 1e5
INF = float("inf")
MINUS_INF = float("-inf")


@dataclass(frozen=True)
class Interval:
    min_val: float
    max_val: float
    rand: Optional[Random] = Random()

    def float(self, rand: Optional[random.Random] = None):
        rand = rand or self.rand
        return rand.uniform(self.min_val, self.max_val)

    def int(self, rand: Optional[random.Random] = None):
        rand = rand or self.rand
        value = round(rand.uniform(self.min_val, self.max_val))
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value


@dataclass(frozen=True)
class IntervalGaussian:
    mean: float
    sigma: float
    min_val: Optional[float] = MINUS_INF
    max_val: Optional[float] = INF
    rand: Optional[Random] = Random()

    def float(self, rand: Optional[random.Random] = None):
        rand = rand or self.rand
        return rand.gauss(self.mean, self.sigma)

    def int(self, rand: Optional[random.Random] = None):
        rand = rand or self.rand
        value = round(rand.gauss(self.mean, self.sigma))
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value


TimeInterval = Tuple[float, float]
TimeIntervalDict = Dict[str, List[TimeInterval]]
WorkerTimeIntervalDict = Dict[str, TimeIntervalDict]
