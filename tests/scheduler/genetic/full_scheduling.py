from sampo.scheduler.genetic.base import GeneticScheduler


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    genetic = GeneticScheduler(number_of_generation=50,
                               mutate_order=3/len(setup_wg.nodes),
                               mutate_resources=1/len(setup_wg.nodes)/len(setup_contractors[0].workers),
                               size_of_population=50)

    genetic.schedule(setup_wg, setup_contractors)
