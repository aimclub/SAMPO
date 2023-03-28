from enum import Enum

from sampo.pipeline.base import InputPipeline
from sampo.pipeline.default import DefaultInputPipeline
from sampo.pipeline.exception import SchedulingPipelineError


class PipelineType(Enum):
    DEFAULT = 0


class SchedulingPipeline:
    @staticmethod
    def create(pipeline_type: PipelineType = PipelineType.DEFAULT) -> InputPipeline:
        match pipeline_type:
            case PipelineType.DEFAULT:
                return DefaultInputPipeline()
            case _:
                raise SchedulingPipelineError('Unknown pipeline type')
