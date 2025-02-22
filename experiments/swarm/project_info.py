from sampo.scheduler.genetic.utils import prepare_optimized_data_structures
from sampo.scheduler.genetic.operators import is_chromosome_correct
from sampo.schemas.landscape import LandscapeConfiguration

# main function is Validation?

class ProjectInfo:

    def __init__(self, wg, contractors, landscape=LandscapeConfiguration()):

        # !! index2contractor_obj _obj?
        self.worker_pool, self.index2node, self.index2zone, self.work_id2index, self.worker_name2index, self.index2contractor_obj, \
        self.worker_pool_indices, self.contractor2index, self.contractor_borders, self.node_indices, self.parents, self.children, \
        self.resources_border = prepare_optimized_data_structures(wg, contractors, landscape)

        _, self.n_workers, self.n_works = self.resources_border.shape

        self.nonzero_resources_indices = [
            (i, j)
            for i in range(self.n_works)
            for j in range(self.n_workers)
            if self.get_resource_borders(work_id=i, worker_id=j) != (0, 0)
        ]

    def get_resource_borders(self, work_id, worker_id):
        min_amount = self.resources_border[0, worker_id, work_id]
        max_amount = min(
            self.resources_border[1, worker_id, work_id],
            self.contractor_borders.max(axis=0)[worker_id]
        )
        return (min_amount, max_amount)

    def assert_chromosome_correct(self, chromosome):
        is_chromosome_valid = is_chromosome_correct(chromosome, self.node_indices, self.parents, self.contractor_borders)
        if not is_chromosome_valid:
            raise Exception("Chromosome seems to be invalid")
            