from enum import Enum

class PipelineError(Exception):
    """
    Raised when any pipeline error occurred.

    This is a kind of 'IllegalStateException', e.g. raising this
    indicates that the corresponding pipeline come to incorrect internal state.
    """
    def __init__(self, message: str):
        super().__init__(message)

from sampo.pipeline.base import InputPipeline
from sampo.pipeline.default import DefaultInputPipeline


class PipelineType(Enum):
    DEFAULT = 0


class SchedulingPipeline:
    @staticmethod
    def create(pipeline_type: PipelineType = PipelineType.DEFAULT) -> InputPipeline:
        match pipeline_type:
            case PipelineType.DEFAULT:
                return DefaultInputPipeline()
            case _:
                raise PipelineError('Unknown pipeline type')
