import random

from sampo.api.genetic_api import ScheduleGenerationScheme
from sampo.generator import SimpleSynthetic, SyntheticGraphType
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import DefaultInputPipeline
from sampo.scheduler import GeneticScheduler
from sampo.schemas import MaterialReq
from sampo.schemas.time_estimator import DefaultWorkEstimator

work_time_estimator = DefaultWorkEstimator()


def run_test(args):
    graph_size, iterations = args
    global seed

    result = []
    for i in range(iterations):
        rand = random.Random(seed)
        ss = SimpleSynthetic(rand=rand)
        wg = ss.work_graph(top_border=graph_size, mode=SyntheticGraphType.SEQUENTIAL)
        print(wg.vertex_count)

        materials_name = ['stone', 'brick', 'sand', 'rubble', 'concrete', 'metal']
        for node in wg.nodes:
            if not node.work_unit.is_service_unit:
                work_materials = rand.choices(materials_name, k=rand.randint(2, 6))
                node.work_unit.material_reqs = [MaterialReq(name, rand.randint(52, 345), name) for name in
                                                work_materials]
        contractors = [get_contractor_by_wg(wg, contractor_id=str(i), contractor_name='Contractor' + ' ' + str(i + 1))
                       for i in range(1)]

        landscape = ss.simple_synthetic_landscape(wg)
        scheduler = GeneticScheduler(number_of_generation=1,
                                     mutate_order=0.05,
                                     mutate_resources=0.005,
                                     size_of_population=1,
                                     work_estimator=work_time_estimator,
                                     rand=rand,
                                     sgs_type=ScheduleGenerationScheme.Serial)
        # schedule = DefaultInputPipeline() \
        #     .wg(wg) \
        #     .contractors(contractors) \
        #     .work_estimator(work_time_estimator) \
        #     .landscape(landscape) \
        #     .schedule(scheduler) \
        #     .finish()
        schedule = DefaultInputPipeline() \
            .wg(wg) \
            .contractors(contractors) \
            .work_estimator(work_time_estimator) \
            .landscape(landscape) \
            .schedule(scheduler) \
            .visualization('2001-01-01')[0] \
            .show_gant_chart()

        # result.append(schedule[0].schedule.execution_time)

        seed += 1

    return result


total_iters = 1
graphs = 1
sizes = [30 * i for i in range(1, graphs + 1)]
total_results = []
seed = 1

for size in sizes:
    results_by_size = run_test((size, total_iters))
    total_results.append(results_by_size)
    print(size)

result_df = {'size': [], 'makespan': []}
for i, results_by_size in enumerate(total_results):
    result = results_by_size[0]

    result_df['size'].append(sizes[i])
    result_df['makespan'].append(result)

