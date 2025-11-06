import codecs
from fastapi import FastAPI, File, UploadFile
import csv
import json
from graph import Graph
from node import Node
import numpy as np

def create_graph_from_json(file: UploadFile):
    """
    Create a graph representation from the json file.

    Args:
       json file containing conections between nodes. 
        assume in the form of:
       [
           {"source":<str>, "target":<str>, "weight":<float>, "bidirectional":<bool>},
           ...
       ]
    """
    content = file.file.read()
    data = json.loads(content)
    graph = Graph()

    for row in data:
        source_id = str(row["source"])
        target_id = str(row["target"])
        weight = float(row["weight"])
        bidirectional = bool(row["bidirectional"])

        # add nodes if they don't exist
        if source_id not in graph.nodes:
            graph.add_node(Node(source_id, np.inf))
        if target_id not in graph.nodes:
            graph.add_node(Node(target_id, np.inf))

        # add edge
        graph.add_edge(graph.nodes[source_id], graph.nodes[target_id], weight, bidirectional)

    # debug print
    #graph.print()

    return graph



def create_graph_from_csv(file: UploadFile):
    """
    Create a graph representation from the csv file.

    Args:
        file (UploadFile): The uploaded file containing adjacency matrix.
        assume in the form of:
        node_id   node_id1, node_id2, ..., node_idn
        node_id1  inf     , dist12    ..., dist1n
        node_id2  dist21  , inf     , ..., dist2n
        ...       ...     , ...     , ..., ...
        node_idn  distn1  ,   distn2, ..., inf
    """
    reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    headers = reader.fieldnames # get the node ids
    n = len(headers)
    # graph construction
    graph = Graph()

    # add nodes
    for i in range(1,n):
        node_id = headers[i]
        graph.add_node(Node(node_id, np.inf))

    # get the distance values and add edges
    for row in reader:
        from_node_id = row[headers[0]]
        for i in range(1,n):
            to_node_id = headers[i]
            if from_node_id != to_node_id:
                weight = float(row[to_node_id]) if row[to_node_id] != 'inf' else np.inf
                graph.add_edge(graph.nodes[from_node_id], graph.nodes[to_node_id], weight, bidirectional=True)
    
    # degug print
    #graph.print()

    return graph