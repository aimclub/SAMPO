import heapq
import random

import numpy as np


def test_building_routes(setup_landscape_many_holders):
    def dijkstra(node_ind, n, target_node_ind):
        visited = [False] * n
        visited[node_ind] = True
        distances = np.full((n, n), MAX_WEIGHT)
        heap = [(0, node_ind)]

        while heap:
            curr_dist, neighbour = heapq.heappop(heap)
            if curr_dist > distances[node_ind][neighbour]:
                continue

            for road in landscape.lg.nodes[neighbour].roads:
                dist = curr_dist + road.weight
                if dist < distances[node_ind][landscape.lg.node2ind[road.finish]]:
                    distances[node_ind][landscape.lg.node2ind[road.finish]] = dist
                    heapq.heappush(heap, (dist, landscape.lg.node2ind[road.finish]))

        return distances[node_ind][target_node_ind]

    MAX_WEIGHT = np.Inf
    landscape = setup_landscape_many_holders
    landscape.build_landscape()

    launches = 1000
    correct = 0
    count_nodes = landscape.lg.vertex_count
    for i in range(launches):
        # use random uniform because of the least entropy
        from_node_ind = random.randint(0, count_nodes - 1)
        to_node_ind = random.randint(0, count_nodes - 1)
        landscape_route_len = landscape.dist_mx[from_node_ind][to_node_ind]
        correct += 1 if landscape_route_len == dijkstra(from_node_ind, count_nodes, to_node_ind) else 0

    assert correct / launches == 1.0


def test_holder_sorting(setup_lg, setup_landscape_many_holders):
    landscape = setup_landscape_many_holders
    landscape.build_landscape()
    lg, holders = setup_lg

    correct = 0
    for node in lg.nodes:
        target_holder = landscape.get_sorted_holders(lg.node2ind[node])[0][1]
        min_dist = np.Inf
        received_holder = 0
        for holder in holders:
            dist = landscape.dist_mx[lg.node2ind[node]][lg.node2ind[holder]]
            if dist < min_dist:
                min_dist = dist
                received_holder = holder
        correct += 1 if landscape.holder_node_id2resource_holder[target_holder].node.id == received_holder.id else 0

    assert correct / lg.vertex_count == 1.0