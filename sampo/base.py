import logging

import sampo.scheduler

from sampo.backend.default import DefaultComputationalBackend

logging.basicConfig(format='[%(name)s] [%(levelname)s] %(message)s', level=logging.NOTSET)


class SAMPO:
    backend = DefaultComputationalBackend()
    logger = logging.getLogger('SAMPO')
