from typing import FrozenSet

WorkerSpecialization = str

DRIVER: WorkerSpecialization = "driver"
FITTER: WorkerSpecialization = "fitter"
HANDYMAN: WorkerSpecialization = "handyman"
ELECTRICIAN: WorkerSpecialization = "electrician"
MANAGER: WorkerSpecialization = "manager"
ENGINEER: WorkerSpecialization = "engineer"

WORKER_TYPES: FrozenSet[WorkerSpecialization] = frozenset({DRIVER, FITTER, HANDYMAN, ELECTRICIAN, MANAGER, ENGINEER})