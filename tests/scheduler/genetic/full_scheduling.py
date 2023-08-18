from sampo.scheduler.genetic.base import GeneticScheduler


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    genetic = GeneticScheduler(number_of_generation=50,
                               mutate_order=0.05,
                               mutate_resources=0.01,
                               size_of_population=50)

    genetic.schedule(setup_wg, setup_contractors)
