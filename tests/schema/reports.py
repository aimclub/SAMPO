import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar

from tests.schema.exceptions import TestException, ExceptionLevel


@dataclass
class TestReport(ABC):
    test_name: str
    test_start_time: datetime
    full_test_info: Dict[str, Any]

    exception: Optional[TestException] = field(init=False)
    info: Dict[str, Any] = field(init=False)
    iter_info: Dict[str, Any] = field(init=False)

    @property
    def is_error(self) -> bool:
        return bool(self.exception) and self.exception.level == ExceptionLevel.Error

    @property
    def is_warning(self) -> bool:
        return bool(self.exception) and self.exception.level == ExceptionLevel.Warning

    @abstractmethod
    def dump(self) -> dict:
        ...

    def dumps(self) -> str:
        return json.dumps(self.dump(), default=str)

    def __post_init__(self):
        assert self.full_test_info
        attributes = 'iter_info', 'info'

        attr_getter = (lambda a: self.full_test_info[a]) \
            if set(attributes) <= self.full_test_info.keys() \
            else (lambda a: {k.split(a)[-1][1:]: v
                             for k, v in self.full_test_info.items() if k.startswith(a)})

        for attr in attributes:
            self.__setattr__(attr, attr_getter(attr))

    def with_exception(self, ex: TestException):
        self.exception = ex
        return self


TComparedObject = TypeVar('TComparedObject')


@dataclass
class ComparisonTestReport(TestReport, Generic[TComparedObject]):
    reference_result: TComparedObject
    compared_result: TComparedObject

    def dump(self) -> dict:
        return {'experiment_info': dict(self.info, **self.iter_info),
                'exception': self.exception and self.exception.dumps(),
                'comparison': {
                    'ref': self.reference_result,
                    'cmp': self.compared_result
                }}
