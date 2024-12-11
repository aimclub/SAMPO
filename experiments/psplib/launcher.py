
import json
from pathlib import Path

from psplib.psplib_time_estimator import ConstantWorkTimeEstimator

from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.graph import WorkGraph, GraphNode, EdgeType
from sampo.schemas.resources import Worker
from sampo.schemas.contractor import Contractor
from sampo.schemas.works import WorkUnit
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.time_estimator import WorkTimeEstimator

from sampo.scheduler.genetic.utils import prepare_optimized_data_structures
from sampo.scheduler.genetic.operators import is_chromosome_correct
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome
from sampo.scheduler.genetic.converter import convert_chromosome_to_schedule

from sampo.scheduler.topological.base import TopologicalScheduler



def load_psplib_data(
        path_to_json: Path|str
    ) -> tuple[WorkGraph, list[Contractor], WorkTimeEstimator, dict]:
    

    # 1. Load data
    with open(path_to_json, "r") as json_data:
        psplib_json_data = json.load(json_data)

    # 
    n_contractors = psplib_json_data["n_resources"]
    workers_names = [f"Resource{i+1}" for i in range(n_contractors)]

        
    # 2. Create WorkGraph
    nodes = {}
    for job in psplib_json_data["jobs"]:
        node_id = job["name"]

        worker_reqs = [
            WorkerReq(
                workers_names[resource_id], 
                Time(0),
                job["resources"][resource_id],
                job["resources"][resource_id],
                workers_names[resource_id]
            ) 
            for resource_id in range(n_contractors)
        ]
                
        work_unit = WorkUnit(
            str(node_id), str(node_id), worker_reqs 
        )
        
        node = GraphNode(work_unit, parent_works=[])
        nodes[node_id] = node

    # 3. Adding parents
    for job in psplib_json_data["jobs"]:
        parents_tuples = [
            (nodes[predecessor_name], 0, EdgeType.FinishStart)
            for predecessor_name in job["predecessors"]
        ]
        nodes[job["name"]].add_parents(parents_tuples)

    # Theoretically node[0] could not be the starting node
    start_nodes = [node for node in nodes.values() if not node.parents]
    finish_nodes = [node for node in nodes.values() if not node.children]
    assert len(start_nodes) == 1
    assert len(finish_nodes) == 1
    start_node = start_nodes[0]
    finish_node = finish_nodes[0]
    wg = WorkGraph(start_node, finish_node)

    # 4. Create contractors
    contractor_name="Contractor1"
    workers = {
        name: Worker(name, name, count, contractor_id=contractor_name)
        for name, count in zip(workers_names, psplib_json_data["resource_pool"])
    }
    contractors = [Contractor(contractor_name, contractor_name, workers)]

    # 5. Create WorkTimeEstimator
    job_durations = {job["name"]: job["duration"] for job in psplib_json_data["jobs"]}
    work_estimator = ConstantWorkTimeEstimator(job_durations)

    # 6. Get best known solution
    best_known_makespan = psplib_json_data.get("best_known_makespan", -1)

    return (wg, contractors, work_estimator, best_known_makespan)


# todo change
def get_dict_of_optimized_data_structures(wg, contractors, landscape=LandscapeConfiguration()):
    ods_tuple = prepare_optimized_data_structures(wg, contractors, landscape)
    ods_names = [
        "worker_pool", "index2node", "index2zone", "work_id2index",
        "worker_name2index", "index2contractor", "worker_pool_indices",
        "contractor2index", "contractor_borders", "node_indices",
        "parents", "children", "resources_border"
    ]
    ods = dict(zip(ods_names, ods_tuple))
    return ods
    

class Launcher:

    
    def __init__(
        self, 
        wg: WorkGraph, 
        contractors: list[Contractor], 
        work_estimator: WorkTimeEstimator, 
        ods:dict
    ):
        self.wg = wg
        self.contractors = contractors
        self.work_estimator = work_estimator
        self.ods = ods
        self.chromosome = self.create_random_chromosome()

    def create_random_chromosome(self):
        # We want a valid chromosome so we can change some parts of it.
        # To do this, we create a random chromosome via random topological sort.
        schedule = TopologicalScheduler(work_estimator=self.work_estimator)
        schedule = schedule.schedule(
            self.wg, self.contractors, 
            spec=ScheduleSpec(), 
            landscape=LandscapeConfiguration()
        )
        
        chromosome = self.schedule_to_chromosome(schedule[0], self.ods)
        return chromosome

        
    def get_schedule(self, activity_list: list[int]):
        # set order of chromosome
        self.chromosome[0][:] = activity_list
        # check if it is valid
        if not self.is_chromosome_valid(self.chromosome, self.ods):
            raise Exception("Chromosome seems to be invalid")
        
        # transform to correct format, idk
        node2work, *other_info = self.chromosome_to_schedule(self.chromosome, self.ods, self.work_estimator)
        scheduled_works = Schedule.from_scheduled_works(node2work.values(), self.wg)

        return (scheduled_works, *other_info)

    def __call__(self, chromosome_order):
        scheduled_works = self.get_schedule(chromosome_order)[0]
        makespan = scheduled_works.full_schedule_df["finish"].max()
        return makespan

    # next few methods are just for correctly calling other functions 

    @staticmethod
    def is_chromosome_valid(chromosome, ods):
        return is_chromosome_correct(
            chromosome, 
            ods["node_indices"], ods["parents"],
            ods["contractor_borders"]
        )
    
    @staticmethod
    def schedule_to_chromosome(schedule, ods):
        return convert_schedule_to_chromosome(
            ods["work_id2index"], ods["worker_name2index"],
            ods["contractor2index"], ods["contractor_borders"], 
            schedule, ScheduleSpec(), LandscapeConfiguration()
        )

    
    @staticmethod
    def chromosome_to_schedule(chromosome, ods, work_estimator):
        return convert_chromosome_to_schedule(
            chromosome,
            ods["worker_pool"], ods["index2node"],
            ods["index2contractor"], ods["index2zone"],
            ods["worker_pool_indices"], ods["worker_name2index"],
            ods["contractor2index"], LandscapeConfiguration(),
            work_estimator=work_estimator
        )



import functools
def get_earliest_finish_time(job_durations, predecessors_dict):

    # recursion is easier to implement
    @functools.cache
    def eft(job_id):
        return job_durations[job_id] + max(
            (eft(p) for p in predecessors_dict[job_id]),
            default=0  # if there are no predecessors, EFT for job is zero
        )

    # EFT for the whole project is the max from each job
    project_eft = max(eft(job_id) for job_id in job_durations.keys())
    return project_eft

