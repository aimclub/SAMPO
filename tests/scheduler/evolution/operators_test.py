from fixtures import *
from sampo.scheduler.genetic.converter import ChromosomeType


TEST_ITERATIONS = 1000


def test_generate_individual(setup_toolbox):
    tb, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        chromosome: ChromosomeType = tb.n_per_product()
        assert tb.validate(chromosome)


def test_mutate_order(setup_toolbox):
    tb, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        individual = tb.n_per_product()
        mutant = tb.mutate(individual[0])
        order = mutant[0]

        # check there are no duplications
        assert len(order) == len(set(order))


def test_mutate_resources(setup_toolbox):
    tb, resources_border = setup_toolbox

    rand = Random()

    for i in range(TEST_ITERATIONS):
        individual = tb.n_per_product()
        type_of_resource = rand.randint(0, len(resources_border[0]) - 1)
        mutant = tb.mutate_resources(individual,
                                     resources_border[0][type_of_resource],
                                     resources_border[1][type_of_resource],
                                     type_of_resource)

        assert tb.validate(mutant)


def test_mate_order(setup_toolbox, setup_wg):
    tb, resources_border = setup_toolbox
    _, _, _, population_size = get_params(setup_wg.vertex_count)

    population = tb.population(n=population_size)

    for i in range(TEST_ITERATIONS):
        individual1, individual2 = [tb.clone(ind) for ind in tb.select(population, 2)]

        tb.mate(individual1[0][0], individual2[0][0])
        order1 = individual1[0][0]
        order2 = individual2[0][0]

        # check there are no duplications
        assert len(order1) == len(set(order1))
        assert len(order2) == len(set(order2))


def test_mate_resources(setup_toolbox, setup_wg):
    tb, resources_border = setup_toolbox
    _, _, _, population_size = get_params(setup_wg.vertex_count)

    population = tb.population(n=population_size)
    rand = Random()

    for i in range(TEST_ITERATIONS):
        individual1, individual2 = [tb.clone(ind) for ind in tb.select(population, 2)]

        workers_for_mate = rand.sample(list(range(len(resources_border) + 1)), 2)

        for worker in workers_for_mate:
            tb.mate_resources(individual1[0][1][workers_for_mate], individual2[0][1][workers_for_mate])

            # check there are correct resources at mate positions
            assert (resources_border[0][worker] <= individual1[0][1][worker]).all() and \
                   (individual1[0][1][worker] <= resources_border[1][worker]).all()
            assert (resources_border[0][worker] <= individual1[0][1][worker]).all() and \
                   (individual1[0][1][worker] <= resources_border[1][worker]).all()

            # check the whole chromosomes
            assert tb.validate(individual1[0])
            assert tb.validate(individual2[0])


