import random
from typing import Optional

from sampo.schemas.utils import uuid_str
from sampo.schemas.works import WorkUnit
from sampo.utilities.sampler.requirements import get_worker_reqs_list
from sampo.utilities.sampler.types import MinMax


def get_work_unit(rand: random.Random, name: str,
                  work_id: Optional[str] = '',
                  volume_type: Optional[str] = 'unit',
                  group: Optional[str] = "default",
                  work_volume: Optional[MinMax[float]] = MinMax[float](0.1, 100.0),
                  req_volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                  req_worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                  ) -> WorkUnit:
    reqs = get_worker_reqs_list(rand, req_volume, req_worker_count)
    work_id = work_id or uuid_str(rand)
    volume = rand.random() * (work_volume.max - work_volume.min) + work_volume.min
    return WorkUnit(work_id, name, worker_reqs=reqs, volume=volume, volume_type=volume_type, group=group)


def get_similar_work_unit(rand: random.Random,
                          exemplar: WorkUnit,
                          scalar: Optional[float] = 1.0,
                          name: Optional[str] = '',
                          work_id: Optional[str] = ''
                          ) -> WorkUnit:
    reqs = [req.scale_all(scalar) for req in exemplar.worker_reqs]
    work_id = work_id or uuid_str(rand)
    name = name or exemplar.name
    return WorkUnit(work_id, name, worker_reqs=reqs, group=exemplar.group,
                    volume=exemplar.volume * scalar, volume_type=exemplar.volume_type)
