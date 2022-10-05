from schemas.interval import Interval, IntervalGaussian

ONE_SECTION_PIPE = Interval(0.5, 3)
DIST_TO_PARENT = Interval(5, 15)
DIST_BETWEEN_BOREHOLES = Interval(0.05, 0.3)
BRANCHING_PROBABILITY = 0.15


# minimal few pack, optimal for parallel test 20-40
# can be divided into few collaborators? for example 15:15:10 for 40x pack
WORKER_PROPORTIONS = {
    "driver": 6,
    "fitter": 6,
    "handyman": 8,
    "electrician": 2,
    "manager": 3,
    "engineer": 2
}

EQUIPMENTS_PROPORTIONS = {}

BASIC_CLASSES = [
    IntervalGaussian(1.2, 0.1, 1, 1.5),
    IntervalGaussian(1.05, 0.2, 1.2, 0.8),
    IntervalGaussian(0.9, 0.3, 0.5, 1)
]

CLASSES_COUNT = 3
START_BASIC_CLASS = 1
BASIC_CLASSES_PROPORTIONS = [1, 2, 2]

WORKER_CLASSES = {
    "driver": BASIC_CLASSES,
    "fitter": BASIC_CLASSES,
    "handyman": BASIC_CLASSES,
    "electrician": BASIC_CLASSES,
    "manager": BASIC_CLASSES,
    "engineer": BASIC_CLASSES
}

WORKER_CLASSES_PROPORTIONS = {
    "driver": BASIC_CLASSES_PROPORTIONS,
    "fitter": BASIC_CLASSES_PROPORTIONS,
    "handyman": BASIC_CLASSES_PROPORTIONS,
    "electrician": BASIC_CLASSES_PROPORTIONS,
    "manager": BASIC_CLASSES_PROPORTIONS,
    "engineer": BASIC_CLASSES_PROPORTIONS
}


MIN_GRAPH_COUNTS = {
    'light_masts': Interval(1, 3),
    'borehole': Interval(4, 12),
    'pipe_nodes': Interval(1, 3),
    'pipe_net': Interval(2, 8)
}

GRAPH_COUNTS = {
    'light_masts': Interval(2, 4),
    'borehole': Interval(8, 28),
    'pipe_nodes': Interval(1, 5),
    'pipe_net': Interval(6, 20)
}

ADDITION_CLUSTER_PROBABILITY = 0.25
MAX_BOREHOLES_PER_BLOCK = 14
