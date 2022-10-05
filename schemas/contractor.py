import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Union
from uuid import uuid4

import numpy as np
from pandas import DataFrame, Series

from schemas.identifiable import Identifiable
from schemas.resources import Worker, Equipment
from schemas.serializable import AutoJSONSerializable
from utils.collections import flatten
from utils.serializers import custom_serializer

AgentsDict = Dict[str, int]


class ContractorType(Enum):
    Minimal = 15
    Average = 25
    Maximal = 35

    def command_capacity(self):
        return self.value


@dataclass
class Contractor(AutoJSONSerializable['Contractor'], Identifiable):
    available_worker_types: List[str]
    available_equipment_types: List[str]
    workers: Dict[Tuple[str, int], Worker]
    equipments: Dict[str, Equipment]
    worker_types: Dict[str, List[Worker]] = field(init=False)

    ignored_fields = ['worker_types']

    def __post_init__(self):
        def key_func(x: Worker):
            return x.name

        w_type2workers = {
            w_type: list(ws)
            for w_type, ws in itertools.groupby(sorted(self.workers.values(), key=key_func), key=key_func)
        }

        self. worker_types = w_type2workers

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


def get_agents_from_contractors(contractors: Union[List['Contractor'], 'Contractor']) -> AgentsDict:
    """
    Gets agent dictionary from contractors list.
    Alias for frequently used functionality.
    :param contractors: List of all the considered contractors
    :return: Dictionary of workers by worker name, next by contractor id
    """
    agents = {}
    for contractor in flatten(contractors):
        for name, val in contractor.workers.items():
            contractor2worker = agents.get(name[0], {})
            if not contractor2worker:
                agents[name[0]] = contractor2worker
            contractor2worker[val.contractor_id] = val
    return agents


def get_contractor_for_resources_schedule(resources: Union[DataFrame, List[Dict[str, int]]],
                                          contractor_type: ContractorType = ContractorType.Minimal,
                                          contractor_id: str = None) \
        -> 'Contractor':
    """
    Generates a contractor, which satisfies the provided resources
    :param resources: List of resource requirements for each task
    :param contractor_id: ID of the new contractor. If None, a new UUID4 is generated
    :param contractor_type: Type of the generated contractor. It influences scale of resource capacities
    :return: A new contractor of the given type
    """

    def aggregate_resources(resources_in: DataFrame, contractor_type_in: ContractorType) -> Series:
        """
        Aggregates resources based on the contractor's type
        :param resources_in: DataFrame with the (possibly sparse) resource grid
        :param contractor_type_in: Type of the generated contractor
        :return:
        """
        min_contractor = resources_in.max(axis=0, skipna=True, numeric_only=True)

        if contractor_type_in is ContractorType.Minimal:
            return min_contractor
        if contractor_type_in is ContractorType.Average:
            return np.ceil(min_contractor) * 1.5
        return np.ceil(min_contractor) * 2.5

        # if contractor_type == MAX_CONTRACTOR:
        #     return resources.sum(axis=0, skipna=True, numeric_only=True)
        # elif contractor_type == AVG_CONTRACTOR:
        #     return np.ceil(DataFrame.from_records([resources.sum(axis=0, skipna=True, numeric_only=True),
        #                                            resources.max(axis=0, skipna=True, numeric_only=True)])
        #                    .mean(axis=0, skipna=True, numeric_only=True))
        # elif contractor_type == MIN_CONTRACTOR:
        #     return resources.max(axis=0, skipna=True, numeric_only=True)

    resources_df = resources if isinstance(resources, DataFrame) else DataFrame(resources)
    contractor_id = contractor_id if contractor_id else str(uuid4().hex)
    # TODO: process equipment
    return Contractor(id=contractor_id,
                      name=contractor_type.name,
                      available_worker_types=list(resources_df.columns),
                      available_equipment_types=list(),
                      workers={(item[0], i): Worker(str(i), item[0], int(item[1]), contractor_id)
                               for i, item in enumerate(aggregate_resources(resources_df, contractor_type).items())},
                      equipments=dict())
