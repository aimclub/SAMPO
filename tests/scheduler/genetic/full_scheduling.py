from sampo.scheduler.genetic.base import GeneticScheduler


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    genetic = GeneticScheduler(number_of_generation=50,
                               mutate_order=0.0,
                               mutate_resources=0.0,
                               size_of_population=50,
                               size_selection=100)

    genetic.schedule(setup_wg, setup_contractors)
