from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union, Iterable
from uuid import uuid4

import numpy as np
from pandas import DataFrame, Series

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.resources import Worker, Equipment
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.types import WorkerName, ContractorName
from sampo.utilities.serializers import custom_serializer

DEFAULT_CONTRACTOR_CAPACITY = 25


@dataclass
class Contractor(AutoJSONSerializable['Contractor'], Identifiable):
    """
    Used to store information about the contractor and its resources
    :param workers: dictionary, where the key is the employee's specialty, and the value is the pool of employees of
    this specialty
    :param equipments: dictionary, where the key is the type of technique, and the value is the pool of techniques of
    that type
    """
    workers: dict[str, Worker] = field(default_factory=dict)
    equipments: dict[str, Equipment] = field(default_factory=dict)

    def __post_init__(self):
        for w in self.workers.values():
            w.contractor_id = self.id

    @property
    def worker_list(self) -> list[Worker]:
        return list(self.workers.values())

    def __hash__(self) -> int:
        return hash(self.id)

    @custom_serializer('workers')
    def serialize_workers(self, value) -> list[dict]:
        return [{'key': k, 'val': v._serialize()} for k, v in value.items()]

    @custom_serializer('equipments')
    def serialize_equipment(self, value) -> []:
        return {k: v._serialize() for k, v in value.items()}

    @classmethod
    @custom_serializer('workers', deserializer=True)
    def deserialize_workers(cls, value) -> dict[str, Worker]:
        return {i['key']: Worker._deserialize(i['val']) for i in value}

    @classmethod
    @custom_serializer('equipments', deserializer=True)
    def deserialize_equipment(cls, value) -> dict[str,Equipment]:
        return {k: Equipment._deserialize(v) for k, v in value.items()}
