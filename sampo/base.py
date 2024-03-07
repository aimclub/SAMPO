import logging

from sampo.backend.default import DefaultComputationalBackend

logging.basicConfig(format='[%(name)s] [%(levelname)s] %(message)s', level=logging.INFO)


class SAMPO:
    backend = DefaultComputationalBackend()
    logger = logging.getLogger('SAMPO')
