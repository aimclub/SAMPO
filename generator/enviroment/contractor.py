from random import Random
from typing import Optional, Dict, List

from generator.config.gen_counts import WORKER_PROPORTIONS, WORKER_CLASSES_PROPORTIONS, WORKER_CLASSES
from schemas.contractor import Contractor
from schemas.interval import IntervalGaussian
from schemas.resources import Worker
from schemas.utils import uuid_str


def get_stochastic_counts(pack_count: float, sigma_scaler: float, proportions: Optional[Dict[str, float]],
                          available_types: Optional[List] = None, rand: Optional[Random] = None) -> Dict[str, int]:
    available_types = available_types or list(proportions.keys())
    counts = {name: prop * pack_count for name, prop in proportions.items() if name in available_types}
    stochastic_counts = {
        name: IntervalGaussian(count, sigma_scaler * count / 2, proportions[name], count).int(rand)
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
    One int for all or List[int] for each contractor. If List, its length should be equal to number_of_contractors
    :param number_of_contractors: Number of generated contractors.
    :return: List with contractors
    """
    assert isinstance(number_of_workers_in_contractors, int) \
           or len(number_of_workers_in_contractors) == number_of_contractors, \
           'Wrong arguments. Length of number_of_workers_in_contractors should equal to number_of_contractors,' + \
           ' or its value should be int,'
    if isinstance(number_of_workers_in_contractors, int):
        number_of_workers_in_contractors = [number_of_workers_in_contractors] * number_of_contractors

    worker_classes_proportions = {
        "driver": [1],
        "fitter": [1],
        "handyman": [1],
        "electrician": [1],
        "manager": [1],
        "engineer": [1]
    }
    return [get_contractor(workers_count, worker_classes_proportions=worker_classes_proportions)
            for workers_count in number_of_workers_in_contractors]


def get_contractor(pack_worker_count: float,
                   sigma_scaler: Optional[float] = 0.1,
                   worker_proportions: Optional[Dict[str, float]] = WORKER_PROPORTIONS,
                   worker_classes_proportions: Optional[Dict[str, List[float]]] = WORKER_CLASSES_PROPORTIONS,
                   available_worker_types: Optional[List] = None, rand: Optional[Random] = None) -> Contractor:
    contractor_id = uuid_str(rand)

    worker_counts = get_stochastic_counts(pack_worker_count, sigma_scaler, worker_proportions,
                                          available_worker_types, rand)

    workers = dict()
    for name in worker_counts.keys():
        # counts = max(worker_counts[name] - worker_proportions[name], 0)
        counts = worker_counts[name]
        classes_proportions = worker_classes_proportions[name]
        sum_prop = sum(classes_proportions)
        counts_classes = [int(counts / sum_prop * prop)
                          for i_class, prop in enumerate(classes_proportions)]
        # print(counts_classes, worker_counts[name]) # TODO add min command in one of productivity classes
        # counts_classes[START_BASIC_CLASS] += worker_proportions[name]
        # counts_classes[START_BASIC_CLASS] += max((worker_counts[name] - sum(counts_classes)), 0)
        for i_class in range(len(classes_proportions)):
            if counts_classes[i_class] == 0:
                continue
            workers[(name, i_class)] = Worker(uuid_str(rand), name,
                                              contractor_id=contractor_id,
                                              productivity_class=i_class,
                                              productivity=WORKER_CLASSES[name][i_class],
                                              count=counts_classes[i_class])

    return Contractor(contractor_id, "", list(worker_counts.keys()), [], workers, {})
