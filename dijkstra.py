from priority_queue import *
from graph import Graph
import numpy as np

def dijkstra(graph, start_node: Node):
    for node in graph.nodes.values():
        node.dist = np.inf
        node.prev = None

    start_node.dist = 0

    Q = BinaryHeapPriorityQueue()
    for node in graph:
        Q.insert(node)

    while not Q.is_empty():
        u = Q.extract_min()

        for neighbor_node, weight in u.neighbors.items():
            distance = u.dist + weight
            if distance < neighbor_node.dist:
                neighbor_node.dist = distance
                neighbor_node.prev = u
                Q.decrease_key(neighbor_node, distance)

    return graph