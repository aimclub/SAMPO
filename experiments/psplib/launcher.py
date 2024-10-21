
import json

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

from sampo.scheduler.genetic.utils import prepare_optimized_data_structures
from sampo.scheduler.genetic.operators import is_chromosome_correct
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome
from sampo.scheduler.genetic.converter import convert_chromosome_to_schedule

from sampo.scheduler.topological.base import RandomizedTopologicalScheduler


class Launcher:

    
    def __init__(self, psplib_json_data: dict):
        
        self.project_data = self.load_data(psplib_json_data)
        
        self.wg = self.create_workgraph(self.project_data)
        self.contractors = self.create_contractors(self.project_data)
        self.work_estimator = self.get_work_estimator(self.project_data)

        self.ods = self.get_optimized_data_structures(self.wg, self.contractors)
        
        self.chromosome = self.get_random_chromosome()

    
    @classmethod
    def from_path_to_json(cls, path_to_json):
        with open(path_to_json, "r") as json_data:
            psplib_json_data = json.load(json_data)
        return cls(psplib_json_data)

    
    @staticmethod
    def load_data(psplib_json_data: dict):

        n_tasks_full = psplib_json_data["n_jobs_full"]
        resource_pool = psplib_json_data["resource_pool"]
        
        n_contractors = len(resource_pool)
        workers = [f"Resource{i+1}" for i in range(n_contractors)]

        job_durations = {}
        job_resources = {}
        predecessors_dict = {}
        for job in psplib_json_data["jobs"]:
            job_id = job["id_"]
            job_durations[job_id] = job["duration"]
            job_resources[job_id] = job["resources"]
            predecessors_dict[job_id] = job["predecessors"]

        return dict(
            n_tasks_full=n_tasks_full,
            n_contractors=n_contractors,
            workers=workers,
            resource_pool=resource_pool,
            job_resources=job_resources,
            predecessors_dict=predecessors_dict,
            job_durations=job_durations
        )
        
    
    @staticmethod
    def create_workgraph(project_data: dict):
        
        nodes = []
        for node_id in range(project_data["n_tasks_full"]):
            worker_reqs = [
                WorkerReq(
                    project_data["workers"][resource_id], 
                    Time(0),
                    project_data["job_resources"][node_id][resource_id],
                    project_data["job_resources"][node_id][resource_id],
                    project_data["workers"][resource_id]
                ) 
                for resource_id in range(project_data["n_contractors"])
            ]
        
            work_unit_id = str(node_id)
            
            work_unit = WorkUnit(
                work_unit_id, str(node_id), worker_reqs 
            )
        
            parents_tuples = [
                (nodes[i], 0, EdgeType.FinishStart)
                for i in project_data["predecessors_dict"][node_id]
            ]
            
            node = GraphNode(work_unit, parents_tuples)
            nodes.append(node)

        wg = WorkGraph(nodes[0], nodes[-1])
        return wg

    
    @staticmethod
    def create_contractors(project_data: dict, contractor_name="Contractor1"):
        workers = {
            name: Worker(name, name, count, contractor_id=contractor_name)
            for name, count in zip(
                project_data["workers"], 
                project_data["resource_pool"]
            )
        }
        contractors = [Contractor(contractor_name, contractor_name, workers)]

        return contractors

    
    @staticmethod
    def get_work_estimator(project_data):
        durations = project_data["job_durations"]
        work_estimator = ConstantWorkTimeEstimator(durations)
        return work_estimator

        
    @staticmethod
    def get_optimized_data_structures(wg, contractors):
        
        ods_tuple = prepare_optimized_data_structures(
            wg, contractors, LandscapeConfiguration()
        )
        ods_names = [
            "worker_pool", "index2node", "index2zone", "work_id2index",
            "worker_name2index", "index2contractor", "worker_pool_indices",
            "contractor2index", "contractor_borders", "node_indices",
            "parents", "children", "resources_border"
        ]
        ods = dict(zip(ods_names, ods_tuple))

        return ods


    def get_random_chromosome(self):
        """
        We want a valid chromosome so we can change some parts of it
        So we create a random chromosome via random topological sort
        """
        schedule = RandomizedTopologicalScheduler(self.work_estimator, 0)
        schedule = schedule.schedule(
            self.wg, self.contractors, 
            spec=ScheduleSpec(), 
            landscape=LandscapeConfiguration()
        )
        
        chromosome = self.schedule_to_chromosome(schedule[0], self.ods)
        return chromosome

        
    def get_schedule_df(self, chromosome_order):
        self.chromosome[0][:] = chromosome_order
    
        if not self.is_chromosome_valid(self.chromosome, self.ods):
            raise Exception("Chromosome seems to be invalid")
        
        schedule = self.chromosome_to_schedule(
            self.chromosome, self.ods, self.work_estimator
        )
        schedule = Schedule.from_scheduled_works(schedule[0].values(), self.wg)
    
        return schedule.full_schedule_df

    
    def convert_job_name_to_index(self, jobs_ids):
        converter = self.ods["work_id2index"]
        jobs_indices = [converter[str(i)] for i in jobs_ids]
        return jobs_indices

    
    def get_makespan(self, jobs_ids):
        jobs_indices = self.convert_job_name_to_index(jobs_ids)
        df = self.get_schedule_df(jobs_indices)
        makespan = df["finish"].max()
        return makespan

        
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


