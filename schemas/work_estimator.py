import pickle
from abc import abstractmethod, ABC
from dataclasses import dataclass, InitVar, field
from enum import auto, Enum
from typing import Dict, Any, Optional

from schemas.contractor import AgentsDict
from schemas.time import Time


class WorkTimeEstimationMode(Enum):
    Optimistic = auto()
    Realistic = auto()
    Pessimistic = auto()


class WorkResourceEstimator(ABC):
    def __init__(self, path: str):
        with open(path, 'rb') as read_pickle:
            self._models = pickle.load(read_pickle)

    @abstractmethod
    def find_work_resources(self, work_name: str, work_volume: float) -> 'ResourceWorkDuration':
        """
        Estimates all the needed resources and execution duration variants for a work with the known volume
        :param work_name: Unique name of the estimated work
        :param work_volume: Volume of the work
        :return: Model of the work execution info
        """
        ...

    @abstractmethod
    def estimate_work_duration(self, work_name: str, work_volume: float, resources: AgentsDict) \
            -> 'ResourceWorkDuration':
        """
        Calculates three variants of the work execution with the given resources
        :param work_name: Unique name oh the estimated work
        :param work_volume: Volume of the work
        :param resources: Dictionary with the resources mapped on this work
        :return: Model of the work execution info
        """
        ...

    @staticmethod
    def __null_prediction() -> Dict[str, Any]:
        return {
            'work_scope': 0,
            'resources': {},
            'pure_vvr_optimistic': 0, 'pauses_optimistic': 0,
            'pure_vvr_realistic': 0, 'pauses_realistic': 0,
            'pure_vvr_pessimistic': 0, 'pauses_pessimistic': 0,
        }


@dataclass()
class ResourceWorkDuration:
    init_name: InitVar[str]
    init_data: InitVar[Dict[str, Any]]
    work_name: str = field(init=False)
    work_volume: float = field(init=False)
    resources: Dict[str, int] = field(init=False)
    min_duration: 'WorkDurationPrediction' = field(init=False)
    avg_duration: 'WorkDurationPrediction' = field(init=False)
    max_duration: 'WorkDurationPrediction' = field(init=False)

    def __post_init__(self, init_name: str, init_data: Dict[str, Any]):
        self.work_name = init_name
        self.work_volume = init_data['work_scope']
        self.resources = init_data['resources'] if 'resources' in init_data else init_data['resources_set']
        self.min_duration = WorkDurationPrediction(
            Time(init_data['pure_vvr_optimistic']),
            Time(init_data['pauses_optimistic']))
        self.avg_duration = WorkDurationPrediction(
            Time(init_data['pure_vvr_realistic']),
            Time(init_data['pauses_realistic']))
        self.max_duration = WorkDurationPrediction(
            Time(init_data['pure_vvr_pessimistic']),
            Time(init_data['pauses_pessimistic']))

        self.resources = self._grammar_check_resources(self.resources)

    @staticmethod
    def _grammar_check_resources(res: Dict[str, int]):
        return {k.replace('омошник', 'омощник'): v for k, v in res.items()}


@dataclass(frozen=True)
class WorkDurationPrediction:
    working_time: Time
    idle_time: Time
    unit: str = 'days'


def get_estimation_mode(mode: WorkTimeEstimationMode):
    def estimate(duration: ResourceWorkDuration):
        if mode is WorkTimeEstimationMode.Optimistic:
            return duration.min_duration
        if mode is WorkTimeEstimationMode.Realistic:
            return duration.avg_duration
        return duration.max_duration

    return estimate


@dataclass()
class WorkTimeEstimator:
    def __init__(self, work_resource_estimator: WorkResourceEstimator,
                 use_idle: Optional[bool] = True, mode: Optional[str] = 'realistic'):
        self._get_duration = None
        self._use_idle = None
        self._work_resources_estimator = work_resource_estimator
        self.set_mode(use_idle, mode)

    def set_mode(self, use_idle: Optional[bool] = True,
                 mode: Optional[WorkTimeEstimationMode] = WorkTimeEstimationMode.Realistic):
        self._get_duration = get_estimation_mode(mode)
        self._use_idle = use_idle

    def estimate_time(self, work_name: str, work_volume: float, resources: AgentsDict) -> Time:
        result = self._work_resources_estimator.estimate_work_duration(work_name, work_volume, resources)
        duration: WorkDurationPrediction = self._get_duration(result)
        estimated_time = duration.working_time + int(self._use_idle) * duration.idle_time
        return estimated_time
