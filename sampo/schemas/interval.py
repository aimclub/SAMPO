"""Interval distributions for random number generation.

Интервальные распределения для генерации случайных чисел.
"""

import random
from abc import ABC, abstractmethod
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
    """Base interval with a probability distribution.

    Базовый интервал с заданным распределением вероятностей.
    """

    @abstractmethod
    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """Return a random float within the interval.

        Возвращает случайное число с плавающей точкой в пределах интервала.

        Args:
            rand (random.Random | None): Generator instance or ``None`` for the
                default. / Экземпляр генератора или ``None`` по умолчанию.

        Returns:
            float: Random value. / Случайное значение.
        """
        ...

    @abstractmethod
    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """Return a random integer within the interval.

        Возвращает случайное целое число в пределах интервала.

        Args:
            rand (random.Random | None): Generator instance or ``None`` for the
                default. / Экземпляр генератора или ``None`` по умолчанию.

        Returns:
            int: Random value. / Случайное значение.
        """
        ...


@dataclass(frozen=True)
class IntervalUniform(Interval):
    """Interval with a uniform distribution.

    Интервал с равномерным распределением.

    Attributes:
        min_val (float): Left boundary. / Левая граница.
        max_val (float): Right boundary. / Правая граница.
        rand (Random | None): Random generator. /
            Генератор случайных чисел.
    """

    min_val: float
    max_val: float
    rand: Optional[Random] = Random()

    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """Generate a random float with a uniform distribution.

        Сгенерировать случайное число с плавающей точкой при равномерном
        распределении.

        Args:
            rand (random.Random | None): Generator instance or ``None`` to use
                internal. / Экземпляр генератора или ``None`` для внутреннего
                использования.

        Returns:
            float: Random value. / Случайное значение.
        """
        rand = rand or self.rand
        return rand.uniform(self.min_val, self.max_val)

    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """Generate a random integer with a uniform distribution.

        Сгенерировать случайное целое число при равномерном распределении.

        Args:
            rand (random.Random | None): Generator instance or ``None`` to use
                internal. / Экземпляр генератора или ``None`` для внутреннего
                использования.

        Returns:
            int: Random value. / Случайное значение.
        """
        rand = rand or self.rand
        value = round(rand.uniform(self.min_val, self.max_val))
        # TODO fix cast math.inf to int
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value


@dataclass(frozen=True)
class IntervalGaussian(Interval):
    """Interval with a Gaussian distribution.

    Интервал с нормальным распределением.

    Attributes:
        mean (float): Distribution mean. / Матожидание распределения.
        sigma (float): Distribution variance. / Дисперсия распределения.
        min_val (float | None): Left boundary. / Левая граница.
        max_val (float | None): Right boundary. / Правая граница.
        rand (Random | None): Random generator. /
            Генератор случайных чисел.
    """

    mean: float
    sigma: float
    min_val: Optional[float] = MINUS_INF
    max_val: Optional[float] = INF
    rand: Optional[Random] = Random()

    def rand_float(self, rand: Optional[random.Random] = None) -> float:
        """Generate a Gaussian-distributed float within bounds.

        Сгенерировать число с плавающей точкой по нормальному распределению в
        заданных границах.

        Args:
            rand (random.Random | None): Generator instance or ``None`` to use
                internal. / Экземпляр генератора или ``None`` для внутреннего
                использования.

        Returns:
            float: Random value. / Случайное значение.
        """
        rand = rand or self.rand
        value = rand.gauss(self.mean, self.sigma)
        value = max(value, self.min_val)
        value = min(value, self.max_val)
        return value

    def rand_int(self, rand: Optional[random.Random] = None) -> int:
        """Generate a Gaussian-distributed integer within bounds.

        Сгенерировать целое число по нормальному распределению в заданных
        границах.

        Args:
            rand (random.Random | None): Generator instance or ``None`` to use
                internal. / Экземпляр генератора или ``None`` для внутреннего
                использования.

        Returns:
            int: Random value. / Случайное значение.
        """
        rand = rand or self.rand
        value = round(rand.gauss(self.mean, self.sigma))
        # TODO fix cast math.inf to int
        value = max(value, int(self.min_val - EPS))
        value = min(value, int(self.max_val + EPS))
        return value
