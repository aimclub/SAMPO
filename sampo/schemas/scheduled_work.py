from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sampo.schemas.contractor import Contractor
from sampo.schemas.landscape import MaterialDelivery
from sampo.schemas.resources import Equipment, ConstructionObject, Worker
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit
from sampo.schemas.zones import ZoneTransition
from sampo.utilities.serializers import custom_serializer


@dataclass
class ScheduledWork(AutoJSONSerializable['ScheduledWork']):
    """
    Contains all necessary info to represent WorkUnit in schedule:

    * WorkUnit
    * list of workers, that are required to complete task
    * start and end time
    * contractor, that complete task
    * list of equipment, that is needed to complete the task
    * list of materials - set of non-renewable resources
    * object - variable, that is used in landscape
    """

    ignored_fields = ['equipments', 'materials', 'object', 'work_unit']

    def __init__(self,
                 work_unit: WorkUnit,
                 start_end_time: tuple[Time, Time],
                 workers: list[Worker],
                 contractor: Contractor | str,
                 equipments: list[Equipment] | None = None,
                 zones_pre: list[ZoneTransition] | None = None,
                 zones_post: list[ZoneTransition] | None = None,
                 materials: list[MaterialDelivery] | None = None,
                 c_object: ConstructionObject | None = None):
        self.id = work_unit.id
        self.name = work_unit.name
        self.display_name = work_unit.display_name
        self.is_service_unit = work_unit.is_service_unit
        self.volume = work_unit.volume
        self.volume_type = work_unit.volume_type
        self.start_end_time = start_end_time
        self.workers = workers if workers is not None else []
        self.equipments = equipments if equipments is not None else []
        self.zones_pre = zones_pre if zones_pre is not None else []
        self.zones_post = zones_post if zones_post is not None else []
        self.materials = materials if materials is not None else []
        self.object = c_object if c_object is not None else []

        if contractor is not None:
            if isinstance(contractor, str):
                self.contractor = contractor
            else:
                self.contractor = contractor.name if contractor.name else contractor.id
        else:
            self.contractor = ""

        self.cost = sum([worker.get_cost() * self.duration.value for worker in self.workers])

    def __str__(self):
        return f'ScheduledWork[work_unit={self.id}, start_end_time={self.start_end_time}, ' \
               f'workers={self.workers}, contractor={self.contractor}]'

    def __repr__(self):
        return self.__str__()

    @custom_serializer('workers')
    @custom_serializer('zones_pre')
    @custom_serializer('zones_post')
    @custom_serializer('start_end_time')
    def serialize_serializable_list(self, value):
        return [t._serialize() for t in value]

    @classmethod
    @custom_serializer('start_end_time', deserializer=True)
    def deserialize_time(cls, value):
        return [Time._deserialize(t) for t in value]

    @classmethod
    @custom_serializer('workers', deserializer=True)
    @custom_serializer('zones_pre', deserializer=True)
    @custom_serializer('zones_post', deserializer=True)
    def deserialize_workers(cls, value):
        return [Worker._deserialize(t) for t in value]

    @property
    def start_time(self) -> Time:
        return self.start_end_time[0]

    @start_time.setter
    def start_time(self, val: Time):
        self.start_end_time = (val, self.start_end_time[1])

    @property
    def finish_time(self) -> Time:
        return self.start_end_time[1]

    @finish_time.setter
    def finish_time(self, val: Time):
        self.start_end_time = (self.start_end_time[0], val)

    @property
    def min_child_start_time(self) -> Time:
        return self.finish_time if self.is_service_unit else self.finish_time + 1

    @staticmethod
    def start_time_getter():
        return lambda x: x.start_end_time[0]

    @staticmethod
    def finish_time_getter():
        return lambda x: x.start_end_time[1]

    @property
    def duration(self) -> Time:
        start, end = self.start_end_time
        return end - start

    def is_overlapped(self, time: int) -> bool:
        start, end = self.start_end_time
        return start <= time < end

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_id': self.id,
            'task_name': self.name,
            'start': self.start_time.value,
            'finish': self.finish_time.value,
            'contractor_id': self.contractor,
            'workers': {worker.name: worker.count for worker in self.workers},
        }
