from fixtures import *
from sampo.scheduler.genetic.converter import ChromosomeType


TEST_ITERATIONS = 10


def test_generate_individual(setup_toolbox):
    (tb, _), _, _, _, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        chromosome: ChromosomeType = tb.generate_chromosome()
        assert tb.validate(chromosome)


def test_mutate_order(setup_toolbox):
    (tb, _), _, _, _, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        individual = tb.generate_chromosome()
        mutant = tb.mutate(individual[0])
        order = mutant[0]

        # check there are no duplications
        assert len(order) == len(set(order))


def test_mutate_resources(setup_toolbox):
    (tb, resources_border), _, _, _, _ = setup_toolbox

    rand = Random()

    for i in range(TEST_ITERATIONS):
        individual = tb.generate_chromosome()
        type_of_resource = rand.randint(0, len(resources_border[0]) - 1)
        mutant = tb.mutate_resources(individual,
                                     resources_border[0][type_of_resource],
                                     resources_border[1][type_of_resource],
                                     type_of_resource)

        assert tb.validate(mutant)


def test_mate_order(setup_toolbox, setup_wg):
    (tb, resources_border), _, _, _, _ = setup_toolbox
    _, _, _, population_size = get_params(setup_wg.vertex_count)

    population = tb.population(n=population_size)

    for i in range(TEST_ITERATIONS):
        individual1, individual2 = tb.select(population, 2)

        individual1, individual2 = tb.mate(individual1[0], individual2[0])
        order1 = individual1[0]
        order2 = individual2[0]

        # check there are no duplications
        assert len(order1) == len(set(order1))
        assert len(order2) == len(set(order2))


def test_mate_resources(setup_toolbox, setup_wg):
    (tb, resources_border), _, _, _, _ = setup_toolbox
    _, _, _, population_size = get_params(setup_wg.vertex_count)

    population = tb.population(n=population_size)
    rand = Random()

    for i in range(TEST_ITERATIONS):
        individual1, individual2 = tb.select(population, 2)

        worker = rand.sample(list(range(len(resources_border) + 1)), 1)[0]
        individual1, individual2 = tb.mate_resources(individual1[0], individual2[0], worker)

        # check there are correct resources at mate positions
        assert (resources_border[0][worker] <= individual1[1][:, worker]).all() and \
               (individual1[1][:, worker] <= resources_border[1][worker]).all()
        assert (resources_border[0][worker] <= individual1[1][:, worker]).all() and \
               (individual1[1][:, worker] <= resources_border[1][worker]).all()

        # check the whole chromosomes
        assert tb.validate(individual1)
        assert tb.validate(individual2)


