from collections import defaultdict
from typing import Iterable

from sampo.schemas import Worker, Contractor
from sampo.schemas.types import WorkerName, ContractorName

WorkerContractorPool = dict[WorkerName, dict[ContractorName, Worker]]

def get_worker_contractor_pool(contractors: Iterable[Contractor]) -> WorkerContractorPool:
    """
    Gets worker-contractor dictionary from contractors list.
    Alias for frequently used functionality.

    :param contractors: list of all the considered contractors
    :return: dictionary of workers by worker name, next by contractor id
    """
    worker_pool = defaultdict(dict)
    for contractor in contractors:
        for name, worker in contractor.workers.items():
            worker_pool[name][contractor.id] = worker.copy()
    return worker_pool
