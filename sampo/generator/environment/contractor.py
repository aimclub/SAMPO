from random import Random

from sampo.generator.config.gen_counts import WORKER_PROPORTIONS
from sampo.schemas.contractor import Contractor
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.resources import Worker
from sampo.schemas.utils import uuid_str


def _get_stochastic_counts(pack_count: float, sigma_scaler: float, proportions: dict[str, float] | None,
                           available_types: list | None = None, rand: Random | None = None) -> dict[str, int]:
    """
    Return random quantity of each type of resources. Random value is gotten from Gaussian distribution

    :param pack_count: The number of resource sets
    :param sigma_scaler: parameter to calculate the scatter by Gaussian distribution with mean=0 amount from the
    transferred proportions
    :param proportions: proportions of quantity for contractor resources to be scaled by pack_worker_count
    :param available_types: Worker types for generation,
    if a subset of worker_proportions is used, if None, all worker_proportions are used
    :param rand: Number generator with a fixed seed, or None for no fixed seed

    """
    available_types = available_types or list(proportions.keys())
    counts = {name: prop * pack_count for name, prop in proportions.items() if name in available_types}
    stochastic_counts = {
        name: IntervalGaussian(count, sigma_scaler * count / 2, proportions[name], count).rand_int(rand)
        for name, count in counts.items()
    }
    return stochastic_counts


def _dict_subtract(d: dict[str, float], subtractor: float) -> dict[str, float]:
    """

    :param d: dict[str:
    :param float]:
    :param subtractor: float:

    """
    for name in d.keys():
        d[name] -= subtractor
    return d


def get_contractor_with_equal_proportions(number_of_workers_in_contractors: int or list[int],
                                          number_of_contractors: int = 1) \
        -> list[Contractor]:
    """Generates a contractors list of specified length with specified capacities

    :param number_of_workers_in_contractors: How many workers of all each contractor contains in itself.
    One int for all or list[int] for each contractor. If list, its length should be equal to number_of_contractors
    :param number_of_contractors: Number of generated contractors.
    :returns: list with contractors

    """
    assert isinstance(number_of_workers_in_contractors, int) \
           or len(number_of_workers_in_contractors) == number_of_contractors, \
        'Wrong arguments. Length of number_of_workers_in_contractors should equal to number_of_contractors,' + \
        ' or its value should be int,'
    if isinstance(number_of_workers_in_contractors, int):
        number_of_workers_in_contractors = [number_of_workers_in_contractors] * number_of_contractors

    return [get_contractor(workers_count) for workers_count in number_of_workers_in_contractors]


def get_contractor(pack_worker_count: float,
                   sigma_scaler: float | None = 0.1,
                   index: int = 0,
                   worker_proportions: dict[str, int] | None = WORKER_PROPORTIONS,
                   available_worker_types: list | None = None, rand: Random | None = None) -> Contractor:
    """Generates a contractor for a synthetic graph for a given resource scalar and generation parameters

    :param pack_worker_count: The number of resource sets
    :param sigma_scaler: parameter to calculate the scatter by Gaussian distribution with mean=0 amount from the
    transferred proportions
    :param index: a parameter for naming a contractor
    :param worker_proportions: proportions of quantity for contractor resources to be scaled by pack_worker_count
    :param available_worker_types: Worker types for generation,
    if a subset of worker_proportions is used, if None, all worker_proportions are used
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :returns: the contractor

    """
    contractor_id = uuid_str(rand)

    worker_counts = _get_stochastic_counts(pack_worker_count, sigma_scaler, worker_proportions,
                                           available_worker_types, rand)

    workers = dict()
    for name in worker_counts.keys():
        # print(counts_classes, worker_counts[name])
        # counts_classes[START_BASIC_CLASS] += worker_proportions[name]
        # counts_classes[START_BASIC_CLASS] += max((worker_counts[name] - sum(counts_classes)), 0)
        workers[name] = Worker(uuid_str(rand), name,
                               contractor_id=contractor_id,
                               count=worker_counts[name])

    return Contractor(contractor_id, f'Contractor {index}', workers, {})
