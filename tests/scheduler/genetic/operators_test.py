from tests.scheduler.genetic.fixtures import *
from sampo.scheduler.genetic.converter import ChromosomeType
import random


TEST_ITERATIONS = 10


def test_generate_individual(setup_toolbox):
    tb, _, _, _, _, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        chromosome = tb.generate_chromosome()
        assert tb.validate(chromosome)


def test_mutate_order(setup_toolbox):
    tb, _, _, _, _, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        individual = tb.generate_chromosome()
        mutant = tb.mutate_order(individual)
        order = mutant[0]

        # check there are no duplications
        assert len(order) == len(set(order))
        assert tb.validate(mutant)


def test_mutate_resources(setup_toolbox):
    tb, _, _, _, _, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        individual = tb.generate_chromosome()
        mutant = tb.mutate_resources(individual)

        assert tb.validate(mutant)


def test_mutate_resource_borders(setup_toolbox):
    tb, _, _, _, _, _ = setup_toolbox

    for i in range(TEST_ITERATIONS):
        individual = tb.generate_chromosome()
        mutant = tb.mutate_resource_borders(individual)

        assert tb.validate(mutant)


def test_mate_order(setup_toolbox, setup_wg):
    tb, _, _, _, _, _ = setup_toolbox
    _, _, _, population_size = get_params(setup_wg.vertex_count)

    population = tb.population_chromosomes(n=population_size)

    for i in range(TEST_ITERATIONS):
        individual1, individual2 = population[:2]

        child1, child2 = tb.mate_order(individual1, individual2)
        order1 = child1[0]
        order2 = child2[0]

        # check there are no duplications
        assert len(order1) == len(set(order1))
        assert len(order2) == len(set(order2))
        assert tb.validate(child1)
        assert tb.validate(child2)


def test_mate_resources(setup_toolbox, setup_wg):
    tb, resources_border, _, _, _, _ = setup_toolbox
    _, _, _, population_size = get_params(setup_wg.vertex_count)

    population = tb.population_chromosomes(n=population_size)

    for i in range(TEST_ITERATIONS):
        individual1, individual2 = random.sample(population, 2)
        individual1, individual2 = tb.mate_resources(individual1, individual2, optimize_resources=False)

        # check there are correct resources at mate positions
        assert (resources_border[0] <= individual1[1].T[:-1]).all() and \
               (individual1[1].T[:-1] <= resources_border[1]).all()
        assert (resources_border[0] <= individual1[1].T[:-1]).all() and \
               (individual1[1].T[:-1] <= resources_border[1]).all()

        # check the whole chromosomes
        assert tb.validate(individual1)
        assert tb.validate(individual2)


