from dataclasses import dataclass
from enum import auto, Enum


class ExceptionLevel(Enum):
    Warning = auto()
    Error = auto()


@dataclass
class TestException:
    level: ExceptionLevel
    message: str

    @staticmethod
    def warning(msg: str) -> 'TestException':
        return TestException(ExceptionLevel.Warning, msg)

    @staticmethod
    def error(msg: str) -> 'TestException':
        return TestException(ExceptionLevel.Error, msg)

    def dumps(self):
        return f'[{self.level.name}]: {self.message}'
