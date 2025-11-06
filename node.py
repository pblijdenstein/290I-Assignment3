import numpy as np

class Node:
    def __init__(self, id, dist = np.inf):
        self.id = id            # node index
        self.idx = -1           # index in the priority queue
        self.dist = dist        # initialize distance to infinity, the key for priority queue
        self.prev = None        # previous node in the shortest path
        self.neighbors = {}     # dictionary store of neighbor nodes (key) and edge weights (value)

    def add_neighbor(self, neighbor_node, weight):
        self.neighbors[neighbor_node] = weight # the key is the neighbor node, the value is the weight of the edge