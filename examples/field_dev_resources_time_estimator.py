import logging
from itertools import chain
from operator import attrgetter
from random import Random
from sampo.schemas.time import Time

from typing import Type

from sampo.utilities.collections_util import build_index
from sampo.schemas import WorkTimeEstimator, WorkUnit, Worker, WorkerReq, WorkEstimationMode, WorkerProductivityMode
from idbadapter import MschmAdapter
from stairsres.res_time_model import ResTimeModel

SERVICE_WORKS = ["Начало работ по марке", "Окончание работ по марке", "NaN", "start of project", "finish of project"]

URL = "test"
model = ResTimeModel(MschmAdapter(url=URL))

logger = logging.getLogger('field-dev-estimator-log')

class FieldDevWorkEstimator(WorkTimeEstimator):
    def __init__(self, rand: Random = Random()):
        self._url = URL
        self._model = model
        self._use_idle = True
        self._estimation_mode = WorkEstimationMode.Realistic
        self.rand = rand
        self._productivity_mode = WorkerProductivityMode.Static

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]):
        w_u = {
            "name": work_unit.name.split("_stage_")[0],
            "volume": work_unit.volume,
            "measurement": work_unit.volume_type,
        }
        w_l = [{"name": w.name, "_count": w.count} for w in worker_list]
        name2worker = build_index(worker_list, attrgetter("name"))
        match self._estimation_mode:
            case WorkEstimationMode.Optimistic:
                mode_str = "0.1"
            case WorkEstimationMode.Realistic:
                mode_str = "0.5"
            case _:
                mode_str = "0.9"

        for res_req in work_unit.worker_reqs:
            if name2worker.get(res_req.kind, None) is None:
                w_l.append({"name": res_req.kind, "_count": 0})
        if w_u["name"] in SERVICE_WORKS:
            return Time(0)
        try:
            time = Time(int(self._model.estimate_time(work_unit=w_u, worker_list=w_l, mode=mode_str)))
            return time
        except Exception as e:
            logger.warning(f"Couldn't estimate time for work unit with name='{w_u['name']}': {e}")

    def find_work_resources(
        self, work_name: str, work_volume: float, resource_name: list[str] | None = None, measurement: str | None = None
    ) -> list[WorkerReq]:
        if work_name in SERVICE_WORKS:
            return []
        worker_req_dict = self._model.get_resources_volumes(
            work_name=work_name, work_volume=work_volume, measurement=measurement
        ) or {}

        worker_reqs = [
            [
                WorkerReq(
                    kind=req["kind"], volume=Time(req["volume"]), min_count=req["min_count"], max_count=req["max_count"]
                )
                for req in worker_req
            ]
            for worker_req in worker_req_dict.values()
        ]
        return list(chain.from_iterable(worker_reqs))

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._estimation_mode = mode

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        self._productivity_mode = mode

    def get_recreate_info(self) -> tuple[Type, tuple]:
        return FieldDevWorkEstimator, tuple(self._url)