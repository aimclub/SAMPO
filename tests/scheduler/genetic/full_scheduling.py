from sampo.scheduler.genetic.base import GeneticScheduler


def test_multiprocessing(setup_wg, setup_contractors):

    genetic = GeneticScheduler(number_of_generation=500,
                               mutate_order=1.0,
                               mutate_resources=1.0,
                               size_of_population=100,
                               size_selection=500)

    genetic.schedule(setup_wg, setup_contractors)
