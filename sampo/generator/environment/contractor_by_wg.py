from _operator import itemgetter
from enum import Enum
from itertools import chain, groupby

from sampo.generator.environment.contractor import get_contractor
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.requirements import WorkerReq


class ContractorGenerationMethod(Enum):
    MIN = 'min'
    MAX = 'max'
    AVG = 'avg'


def _value_by_req(method: ContractorGenerationMethod, req: WorkerReq) -> int:
    """
    Sets the function by which the number for the function of searching for a contractor by the graph of works
    is determined by the given parameter

    :param method: type the specified parameter: min ~ min_count, max ~ max_count, avg ~ (min_count + max_count) / 2
    :param req: the Worker Req
    :return:
    """
    val = 0
    match method:
        case ContractorGenerationMethod.MIN: val = req.min_count
        case ContractorGenerationMethod.AVG: val = (req.min_count + req.max_count) / 2
        case ContractorGenerationMethod.MAX: val = req.max_count
    return int(val)


def get_contractor_by_wg(wg: WorkGraph,
                         scaler: float | None = 1,
                         method: ContractorGenerationMethod = ContractorGenerationMethod.AVG) -> Contractor:
    """
    Creates a pool of contractor resources based on job requirements, selecting the maximum specified parameter

    :param wg: The graph of works for which it is necessary to find a set of resources
    :param scaler: Multiplier for the number of resources in the contractor
    :param method: type the specified parameter: min ~ min_count, max ~ max_count, avg ~ (min_count + max_count) / 2
    :return: Contractor capable of completing the work schedule
    """
    if scaler < 1:
        raise ValueError('scaler should be greater than 1')
    wg_reqs = list(chain(*[n.work_unit.worker_reqs for n in wg.nodes]))
    min_wg_reqs = [(req.kind, _value_by_req(method, req)) for req in wg_reqs]
    maximal_min_req: dict[str, int] = \
        dict(list(group[1])[-1] for group in groupby(sorted(min_wg_reqs), itemgetter(0)))
    return get_contractor(scaler, sigma_scaler=0, worker_proportions=maximal_min_req)
