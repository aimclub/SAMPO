from logging import Logger

import sampo.scheduler
from sampo.backend.default import DefaultComputationalBackend


class SAMPO:
    backend = DefaultComputationalBackend()
    logger = Logger('SAMPO')
