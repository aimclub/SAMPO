from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Union
from uuid import uuid4

import numpy as np
from pandas import DataFrame, Series

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.resources import Worker, Equipment
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.types import WorkerName, ContractorName
from sampo.utilities.serializers import custom_serializer

WorkerContractorPool = Dict[WorkerName, Dict[ContractorName, Worker]]
DefaultContractorCapacity = 25


@dataclass
class Contractor(AutoJSONSerializable['Contractor'], Identifiable):
    """
    Used to store information about the contractor and its resources
    :param workers: dictionary, where the key is the employee's specialty, and the value is the pool of employees of
    this specialty
    :param equipments: dictionary, where the key is the type of technique, and the value is the pool of techniques of
    that type
    """
    workers: Dict[str, Worker]
    equipments: Dict[str, Equipment]

    def __post_init__(self):
        for w in self.workers.values():
            w.contractor_id = self.id

    def __hash__(self):
        return hash(self.id)

    @custom_serializer('workers')
    def serialize_workers(self, value):
        return [{'key': k, 'val': v._serialize()} for k, v in value.items()]

    @custom_serializer('equipments')
    def serialize_equipment(self, value):
        return {k: v._serialize() for k, v in value.items()}

    @classmethod
    @custom_serializer('workers', deserializer=True)
    def deserialize_workers(cls, value):
        return {tuple(i['key']): Worker._deserialize(i['val']) for i in value}

    @classmethod
    @custom_serializer('equipments', deserializer=True)
    def deserialize_equipment(cls, value):
        return {k: Equipment._deserialize(v) for k, v in value.items()}


# TODO move from schemas
def get_worker_contractor_pool(contractors: Union[List['Contractor'], 'Contractor']) -> WorkerContractorPool:
    """
    Gets agent dictionary from contractors list.
    Alias for frequently used functionality.
    :param contractors: List of all the considered contractors
    :return: Dictionary of workers by worker name, next by contractor id
    """
    agents = defaultdict(dict)
    for contractor in contractors:
        for name, worker in contractor.workers.items():
            agents[name][contractor.id] = worker.copy()
    return agents


# TODO move from schemas
def get_contractor_for_resources_schedule(resources: Union[DataFrame, List[Dict[str, int]]],
                                          contractor_capacity: int = DefaultContractorCapacity,
                                          contractor_id: str = None,
                                          contractor_name: str = "") \
        -> 'Contractor':
    """
    Generates a contractor, which satisfies the provided resources
    :param resources: List of resource requirements for each task
    :param contractor_id: ID of the new contractor. If None, a new UUID4 is generated
    :param contractor_name: Name of the new contractor. If None, an empty string should be passed
    :param contractor_capacity: Capacity of the generated contractor. It influences scale of resource capacities
    :return: A new contractor of the given type
    """

    def aggregate_resources(resources_in: DataFrame, contractor_capacity_in: int) -> Series:
        """
        Aggregates resources based on the contractor's capacity
        :param resources_in: DataFrame with the (possibly sparse) resource grid
        :param contractor_capacity_in: Capacity of the generated contractor
        :return: Series with resources count
        """
        min_contractor = resources_in.max(axis=0, skipna=True, numeric_only=True)

        return np.ceil(min_contractor * contractor_capacity_in / 10)

    resources_df = resources if isinstance(resources, DataFrame) else DataFrame(resources)
    contractor_id = contractor_id if contractor_id else str(uuid4().hex)

    # TODO: process equipment
    return Contractor(id=contractor_id,
                      name=contractor_name,
                      workers={item[0]: Worker(str(i), item[0], int(item[1]), contractor_id)
                               for i, item in enumerate(aggregate_resources(resources_df, contractor_capacity).items())},
                      equipments=dict())
