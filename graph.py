class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, from_node, to_node, weight, bidirectional=True):
        from_node.add_neighbor(to_node, weight)
        if bidirectional:
            to_node.add_neighbor(from_node, weight)

    def print(self):
        for node in self.nodes.values():
            print(f"Node {node.id}:")
            for neighbor, weight in node.neighbors.items():
                print(f"  -> Neighbor {neighbor.id} with weight {weight}")

    def __iter__(self):
        return iter(self.nodes.values())