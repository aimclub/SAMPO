import random
from typing import Optional, List

from sampo.schemas.requirements import WorkerReq
from sampo.utilities.sampler.resources import WORKER_TYPES, WorkerSpecialization
from sampo.utilities.sampler.types import MinMax


def get_worker_req(rand: random.Random,
                   name: str,
                   volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                   worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                   ) -> WorkerReq:
    assert type(rand) == random.Random
    assert type(name) == str
    count = rand.randint(volume.min, volume.max)
    return WorkerReq(name, count, worker_count.min, worker_count.max)


def get_worker_reqs_list(rand: random.Random,
                         volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                         worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                         ) -> List[WorkerReq]:
    assert type(rand) == random.Random
    names: List[WorkerSpecialization] = list(WORKER_TYPES)
    rand.shuffle(names)
    req_count = rand.randint(1, len(names))
    names = names[:req_count]
    return get_worker_specific_reqs_list(rand, names, volume, worker_count)


def get_worker_specific_reqs_list(rand: random.Random,
                                  worker_names: List[WorkerSpecialization],
                                  volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                                  worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                                  ) -> List[WorkerReq]:
    assert type(rand) == random.Random
    reqs = [get_worker_req(rand, name, volume, worker_count) for name in worker_names]
    return reqs
