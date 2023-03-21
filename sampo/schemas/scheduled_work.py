from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Equipment, ConstructionObject, Worker
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit
from sampo.utilities.serializers import custom_serializer


# TODO: describe the class (description, parameters)
@dataclass
class ScheduledWork(AutoJSONSerializable['ScheduledWork']):

    ignored_fields = ['equipments', 'materials', 'object']

    def __init__(self,
                 work_unit: WorkUnit,
                 start_end_time: Tuple[Time, Time],
                 workers: List[Worker],
                 contractor: Contractor | str,
                 equipments: Optional[List[Equipment]] = None,
                 materials: Optional[List[Equipment]] = None,
                 object: Optional[ConstructionObject] = None):
        self.work_unit = work_unit
        self.start_end_time = start_end_time
        self.workers = workers
        self.equipments = equipments
        self.materials = materials
        self.object = object
        if contractor is not None:
            if isinstance(contractor, str):
                self.contractor = contractor
            else:
                self.contractor = contractor.name if contractor.name else contractor.id
        else:
            self.contractor = ""

        self.cost = 0
        for worker in self.workers:
            self.cost += worker.get_cost() * self.duration.value

    def __str__(self):
        return f'ScheduledWork[work_unit={self.work_unit}, start_end_time={self.start_end_time}, ' \
               f'workers={self.workers}, contractor={self.contractor}]'

    def __repr__(self):
        return self.__str__()

    @custom_serializer('workers')
    @custom_serializer('start_end_time')
    def serialize_serializable_list(self, value):
        return [t._serialize() for t in value]

    @classmethod
    @custom_serializer('start_end_time', deserializer=True)
    def deserialize_time(cls, value):
        return [Time._deserialize(t) for t in value]

    @classmethod
    @custom_serializer('workers', deserializer=True)
    def deserialize_workers(cls, value):
        return [Worker._deserialize(t) for t in value]

    # TODO: describe the function (description, parameters, return type)
    def get_actual_duration(self, work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        return self.work_unit.estimate_static(self.workers, work_estimator)

    # TODO: describe the function (description return type)
    @property
    def start_time(self) -> Time:
        return self.start_end_time[0]

    # TODO: describe the function (description, parameters, return type)
    @start_time.setter
    def start_time(self, val: Time):
        self.start_end_time = (val, self.start_end_time[1])

    # TODO: describe the function (description, return type)
    @property
    def finish_time(self) -> Time:
        return self.start_end_time[1]

    # TODO: describe the function (description, parameters, return type)
    @finish_time.setter
    def finish_time(self, val: Time):
        self.start_end_time = (self.start_end_time[0], val)

    # TODO: describe the function (description, return type)
    @staticmethod
    def start_time_getter():
        return lambda x: x.start_end_time[0]

    # TODO: describe the function (description, return type)
    @staticmethod
    def finish_time_getter():
        return lambda x: x.start_end_time[1]

    # TODO: describe the function (description, return type)
    @property
    def duration(self) -> Time:
        start, end = self.start_end_time
        return end - start
    
    # TODO: describe the function (description, parameters, return type)
    def is_overlapped(self, time: int) -> bool:
        start, end = self.start_end_time
        return start <= time < end

    # TODO: describe the function (description, return type)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.work_unit.id,
            "task_name": self.work_unit.name,
            "start": self.start_time.value,
            "finish": self.finish_time.value,
            "contractor_id": self.contractor,
            "workers": {worker.name: worker.count for worker in self.workers},
        }

    def __deepcopy__(self, memodict={}):
        return ScheduledWork(deepcopy(self.work_unit, memodict),
                             deepcopy(self.start_end_time, memodict),
                             deepcopy(self.workers, memodict),
                             self.contractor)
