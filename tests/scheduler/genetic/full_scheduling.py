from sampo.scheduler import GeneticScheduler


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    genetic = GeneticScheduler(number_of_generation=10,
                               mutate_order=0.05,
                               mutate_resources=0.005,
                               size_of_population=50)

    genetic.schedule(setup_wg, setup_contractors, landscape=setup_landscape)
