from collections import defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import List, Optional, Callable

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.requirements import WorkerReq, EquipmentReq, MaterialReq, ConstructionObjectReq
from sampo.schemas.resources import Worker
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.serializers import custom_serializer


@dataclass
class WorkUnit(AutoJSONSerializable['WorkUnit'], Identifiable):
    """
    Class that describe vertex in graph (one work/task)
    """
    def __init__(self, id: str, name: str, worker_reqs: list[WorkerReq] = [], equipment_reqs: list[EquipmentReq] = [],
                 material_reqs: list[MaterialReq] = [], object_reqs: list[ConstructionObjectReq] = [],
                 group: str = 'default', is_service_unit=False, volume: float = 0,
                 volume_type: str = "unit", display_name: str = ""):
        """
        :param worker_reqs: list of required professions (i.e. workers)
        :param equipment_reqs: list of required equipment
        :param material_reqs: list of required materials (e.g. logs, stones, gravel etc.)
        :param object_reqs: list of required objects (e.g. electricity, pipelines, roads)
        :param group: union block of works
        :param is_service_unit: service units are additional vertexes
        :param volume: scope of work
        :param volume_type: unit of scope of work
        :param display_name: name of work
        """
        super(WorkUnit, self).__init__(id, name)
        self.worker_reqs = worker_reqs
        self.equipment_reqs = equipment_reqs
        self.object_reqs = object_reqs
        self.material_reqs = material_reqs
        self.group = group
        self.is_service_unit = is_service_unit
        self.volume = volume
        self.volume_type = volume_type
        self.display_name = display_name

    @custom_serializer('worker_reqs')
    def worker_reqs_serializer(self, value: list[WorkerReq]):
        """
        Return serialized list of worker requirements

        :param value: list of worker requirements
        :return: list of worker requirements
        """
        return [wr._serialize() for wr in value]

    @classmethod
    @custom_serializer('worker_reqs', deserializer=True)
    def worker_reqs_deserializer(cls, value):
        """
        Get list of worker requirements

        :param value: serialized list of work requirements
        :return: list of worker requirements
        """
        return [WorkerReq._deserialize(wr) for wr in value]

    # TODO: move this logit to WorkTimeEstimator
    def estimate_static(self, worker_list: List[Worker], work_estimator: WorkTimeEstimator = None) -> Time:
        """
        Calculate summary time of task execution (without stochastic part)

        :param worker_list:
        :param work_estimator:
        :return: time of task execution
        """
        if work_estimator:
            # TODO What?? We are loosing much performance
            #  when processing worker names EACH estimation
            workers = {w.name.replace("_res_fact", ""): w.count for w in worker_list}
            work_time = work_estimator.estimate_time(self.name.split("_stage_")[0], self.volume, workers)
            if work_time > 0:
                return work_time

        return self._abstract_estimate(worker_list, get_static_by_worker)

    def estimate_stochastic(self, worker_list: list[Worker], rand: Random = None) -> Time:
        """
        Calculate summary time of task execution (considering stochastic part)

        :param worker_list:
        :param rand: random number
        :return: time of task execution
        """
        return self._abstract_estimate(worker_list, get_stochastic_by_worker, rand)

    def _abstract_estimate(self, worker_list: list[Worker],
                           get_productivity: Callable[[Worker, Random], float],
                           rand: Random = None) -> Time:
        """
        Abstract method that can estimate time of task execution using certain function get_productivity()

        :param worker_list: list of workers
        :param get_productivity: function, that calculate time of task productivity
        :param rand: stochastic part
        :return: maximum time of task execution
        """
        groups = defaultdict(list)
        for w in worker_list:
            groups[w.name].append(w)
        times = [Time(0)]  # if there are no requirements for the work, it is done instantly
        for req in self.worker_reqs:
            if req.min_count == 0:
                continue
            name = req.kind
            command = groups[name]
            worker_count = sum([worker.count for worker in command], 0)
            if worker_count < req.min_count:
                return Time.inf()
            productivity = sum([get_productivity(c, rand) for c in command], 0) / worker_count
            productivity *= communication_coefficient(worker_count, req.max_count)
            if productivity == 0:
                return Time.inf()
            times.append(Time(req.volume // productivity))
        return max(times)

    def __getstate__(self):
        # custom method to avoid calling __hash__() on GraphNode objects
        return self._serialize()

    def __setstate__(self, state):
        new_work_unit = self._deserialize(state)
        self.worker_reqs = new_work_unit.worker_reqs
        self.equipment_reqs = new_work_unit.equipment_reqs
        self.object_reqs = new_work_unit.object_reqs
        self.material_reqs = new_work_unit.material_reqs
        self.id = new_work_unit.id
        self.name = new_work_unit.name
        self.is_service_unit = new_work_unit.is_service_unit
        self.volume = new_work_unit.volume
        self.volume_type = new_work_unit.volume_type
        self.group = new_work_unit.group
        self.display_name = new_work_unit.display_name


def get_static_by_worker(w: Worker, _: Optional[Random] = None):
    """
    Receive productivity of certain worker

    :param w: certain worker
    :param _: parameter for stochastic part
    :return: result of method get_static_productivity()
    """
    return w.get_static_productivity()


def get_stochastic_by_worker(w: Worker, rand: Optional[Random] = None):
    """Return the stochastic productivity of worker team"""
    return w.get_stochastic_productivity(rand)


# Function is chosen because it has a quadratic decrease in efficiency as the number of commands on the object
# increases, after the maximum number of commands begins to decrease in efficiency, and its growth rate depends on
# the maximum number of commands.
# sum(1 - ((x-1)^2 / max_groups^2), where x from 1 to groups_count
# TODO: describe the function (description, parameters, return type)
def communication_coefficient(groups_count: int, max_groups: int) -> float:
    n = groups_count
    m = max_groups
    return 1 / (6 * m ** 2) * (-2 * n ** 3 + 3 * n ** 2 + (6 * m ** 2 - 1) * n)
