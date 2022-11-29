from sampo.schemas.interval import IntervalUniform

ONE_SECTION_PIPE = IntervalUniform(0.5, 3)
DIST_TO_PARENT = IntervalUniform(5, 15)
DIST_BETWEEN_BOREHOLES = IntervalUniform(0.05, 0.3)
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

MIN_GRAPH_COUNTS = {
    'light_masts': IntervalUniform(1, 3),
    'borehole': IntervalUniform(4, 12),
    'pipe_nodes': IntervalUniform(1, 3),
    'pipe_net': IntervalUniform(2, 8)
}

GRAPH_COUNTS = {
    'light_masts': IntervalUniform(2, 4),
    'borehole': IntervalUniform(8, 28),
    'pipe_nodes': IntervalUniform(1, 5),
    'pipe_net': IntervalUniform(6, 20)
}

ADDITION_CLUSTER_PROBABILITY = 0.25
MAX_BOREHOLES_PER_BLOCK = 14
