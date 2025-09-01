"""Random interval utilities.

Утилиты для работы со случайными интервалами.
"""

import random
from abc import abstractmethod, ABC
from dataclasses import dataclass
from random import Random
from typing import Optional

from sampo.schemas.serializable import AutoJSONSerializable

# to work with float and avoid errors due to inaccuracy
EPS = 1e5
# to work with distributions when certain start and finish value is undefined
INF = float('inf')
MINUS_INF = float('-inf')


# TODO: Take out common parts, deal with types, rethink intervals


class Interval(AutoJSONSerializable['BaseReq'], ABC):
    """Base class for random number generation from distributions.

    Базовый класс для генерации случайных чисел по распределениям.
    """

    @abstractmethod
    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """Return random float within interval.

        Возвращает случайное число с плавающей точкой внутри интервала.

        Args:
            rand (random.Random | None): random generator.
                Генератор случайных чисел.

        Returns:
            float: generated value.
                Сгенерированное значение.
        """
        ...

    @abstractmethod
    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """Return random integer within interval.

        Возвращает случайное целое число внутри интервала.

        Args:
            rand (random.Random | None): random generator.
                Генератор случайных чисел.

        Returns:
            int: generated integer.
                Сгенерированное целое число.
        """
        ...


@dataclass(frozen=True)
class IntervalUniform(Interval):
    """Uniform distribution interval.

    Интервал с равномерным распределением.

    Attributes:
        min_val (float): left boundary.
            Левая граница.
        max_val (float): right boundary.
            Правая граница.
        rand (Random | None): random generator with seed.
            Генератор случайных чисел с зерном.
    """
    min_val: float
    max_val: float
    rand: Optional[Random] = Random()

    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """Return random float within boundaries.

        Возвращает случайное число с плавающей точкой в пределах границ.

        Args:
            rand (random.Random | None): optional generator.
                Дополнительный генератор.

        Returns:
            float: random float value.
                Случайное число с плавающей точкой.
        """
        rand = rand or self.rand
        return rand.uniform(self.min_val, self.max_val)

    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """Return random integer within boundaries.

        Возвращает случайное целое число в пределах границ.

        Args:
            rand (random.Random | None): optional generator.
                Дополнительный генератор.

        Returns:
            int: random integer value.
                Случайное целое число.
        """
        rand = rand or self.rand
        value = round(rand.uniform(self.min_val, self.max_val))
        # TODO fix cast math.inf to int
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value


@dataclass(frozen=True)
class IntervalGaussian(Interval):
    """Gaussian distribution interval.

    Интервал с нормальным распределением.

    Attributes:
        mean (float): distribution mean.
            Среднее распределения.
        sigma (float): distribution variance.
            Дисперсия распределения.
        min_val (float | None): left boundary.
            Левая граница.
        max_val (float | None): right boundary.
            Правая граница.
        rand (Random | None): random generator with seed.
            Генератор случайных чисел с зерном.
    """

    mean: float
    sigma: float
    min_val: Optional[float] = MINUS_INF
    max_val: Optional[float] = INF
    rand: Optional[Random] = Random()

    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """Return random float following Gaussian distribution.

        Возвращает случайное число с плавающей точкой по нормальному распределению.

        Args:
            rand (random.Random | None): random generator.
                Генератор случайных чисел.

        Returns:
            float: random float value.
                Случайное число с плавающей точкой.
        """
        rand = rand or self.rand
        value = rand.gauss(self.mean, self.sigma)
        value = max(value, self.min_val)
        value = min(value, self.max_val)
        return value


    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """Return random integer following Gaussian distribution.

        Возвращает случайное целое число по нормальному распределению.

        Args:
            rand (random.Random | None): random generator.
                Генератор случайных чисел.

        Returns:
            int: random integer value.
                Случайное целое число.
        """
        rand = rand or self.rand
        value = round(rand.gauss(self.mean, self.sigma))
        # TODO fix cast math.inf to int
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value
