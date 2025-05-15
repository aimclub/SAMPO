from sampo.generator import SimpleSynthetic, SyntheticGraphType
from random import Random

rand = Random(123)
ss = SimpleSynthetic(rand)

fixed_wg = ss.work_graph(mode=SyntheticGraphType.GENERAL, cluster_counts=10, bottom_border=100, top_border=100)

for node in fixed_wg.nodes:
    assert node.edges_to is not None
    assert node.edges_from is not None

print(fixed_wg.vertex_count)

from sampo.schemas.stochastic_graph import ProbabilisticFollowingStochasticGraphScheme

# construct the stochastic graph scheme
graph_scheme = ProbabilisticFollowingStochasticGraphScheme(rand=rand, wg=fixed_wg)

for i, node in enumerate(fixed_wg.nodes):
    defect = ss.graph_nodes(top_border=20)

    defect_prob = rand.random() * 0.1
    graph_scheme.add_part(node=node.id, nodes=defect, prob=defect_prob)

perfect_wg = graph_scheme.prepare_graph().to_work_graph()
print(perfect_wg.vertex_count)

perfect_wg_1 = graph_scheme.prepare_graph().to_work_graph()
print(perfect_wg_1.vertex_count)

