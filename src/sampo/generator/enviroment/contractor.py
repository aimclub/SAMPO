from random import Random
from typing import Optional, Dict, List

from sampo.generator.config.gen_counts import WORKER_PROPORTIONS
from sampo.schemas.contractor import Contractor
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.resources import Worker
from sampo.schemas.utils import uuid_str


def get_stochastic_counts(pack_count: float, sigma_scaler: float, proportions: Optional[Dict[str, float]],
                          available_types: Optional[List] = None, rand: Optional[Random] = None) -> Dict[str, int]:
    available_types = available_types or list(proportions.keys())
    counts = {name: prop * pack_count for name, prop in proportions.items() if name in available_types}
    stochastic_counts = {
        name: IntervalGaussian(count, sigma_scaler * count / 2, proportions[name], count).rand_int(rand)
        for name, count in counts.items()
    }
    return stochastic_counts


def dict_subtract(d: Dict[str, float], subtractor: float) -> Dict[str, float]:
    for name in d.keys():
        d[name] -= subtractor
    return d


def get_contractor_with_equal_proportions(number_of_workers_in_contractors: int or List[int],
                                          number_of_contractors: int = 1) \
        -> List[Contractor]:
    """
    Generates a contractors list of specified length with specified capacities
    :param number_of_workers_in_contractors: How many workers of all each contractor contains in itself.
    One int for all or List[int] for each contractor. If List, its length should be equal to number_of_contractors.
    :param number_of_contractors: Number of generated contractors.
    :return: List with contractors
    """
    assert isinstance(number_of_workers_in_contractors, int) \
           or len(number_of_workers_in_contractors) == number_of_contractors, \
        'Wrong arguments. Length of number_of_workers_in_contractors should equal to number_of_contractors,' + \
        ' or its value should be int,'
    if isinstance(number_of_workers_in_contractors, int):
        number_of_workers_in_contractors = [number_of_workers_in_contractors] * number_of_contractors

    return [get_contractor(workers_count) for workers_count in number_of_workers_in_contractors]


def get_contractor(pack_worker_count: float,
                   sigma_scaler: Optional[float] = 0.1,
                   index: int = 0,
                   worker_proportions: Optional[Dict[str, float]] = WORKER_PROPORTIONS,
                   available_worker_types: Optional[List] = None, rand: Optional[Random] = None) -> Contractor:
    contractor_id = uuid_str(rand)

    worker_counts = get_stochastic_counts(pack_worker_count, sigma_scaler, worker_proportions,
                                          available_worker_types, rand)

    workers = dict()
    for name in worker_counts.keys():
        # print(counts_classes, worker_counts[name])
        # counts_classes[START_BASIC_CLASS] += worker_proportions[name]
        # counts_classes[START_BASIC_CLASS] += max((worker_counts[name] - sum(counts_classes)), 0)
        workers[name] = Worker(uuid_str(rand), name,
                               contractor_id=contractor_id,
                               count=worker_counts[name])

    return Contractor(contractor_id, f"Подрядчик {index}", workers, {})
