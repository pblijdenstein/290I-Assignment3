from fastapi import FastAPI, UploadFile
import uvicorn
from collections import defaultdict
from dijkstra import dijkstra  # make sure this file exists and function name is correct

# create FastAPI app
app = FastAPI()

# global variable for active graph
active_graph = None


@app.get("/")
async def root():
    return {"message": "Welcome to the Shortest Path Solver!"}


def _build_adjacency(edges):
    """Convert edge list into adjacency dict: {node: {neighbor: weight}}."""
    adj = defaultdict(dict)
    for e in edges:
        u = str(e["source"])
        v = str(e["target"])
        w = float(e.get("weight", 1))
        adj[u][v] = w
        adj.setdefault(v, adj.get(v, {}))
        if e.get("bidirectional", False):
            adj[v][u] = w
            adj.setdefault(u, adj.get(u, {}))
    return dict(adj)


@app.post("/upload_graph_json/")
async def create_upload_file(file: UploadFile):
    """
    Accept a .json file containing an edge list like:
    [
      {"source":"0","target":"1","weight":1,"bidirectional":true},
      ...
    ]
    Build and store an adjacency dict: {node: {neighbor: weight}} in active_graph.
    """
    import json
    global active_graph

    # 1) Validate extension
    if not file.filename.lower().endswith(".json"):
        return {"Upload Error": "Invalid file type"}

    # 2) Read file bytes -> parse JSON
    raw = await file.read()
    try:
        data = json.loads(raw)
    except Exception:
        return {"Upload Error": "Invalid JSON format"}

    # 3) If someone embedded JSON as a string, parse again
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return {"Upload Error": "JSON content is a plain string and not valid"}

    # 4) If we already got an adjacency dict, accept it as-is
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        active_graph = data
        return {"Upload Success": file.filename}

    # 5) Otherwise we expect a list of edge dicts -> build adjacency
    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], dict)):
        return {"Upload Error": "Expected a JSON array of edge objects"}

    active_graph = _build_adjacency(data)
    return {"Upload Success": file.filename}

# -------- Local fallback Dijkstra (works on our adjacency dict) --------
import heapq

def _dijkstra_local(adj, start, end):
    """
    adj: {node: {neighbor: weight}}
    start, end: strings
    returns (path_list, total_distance) or (None, None) if unreachable
    """
    dist = {n: float("inf") for n in adj.keys()}
    prev = {n: None for n in adj.keys()}
    dist[start] = 0.0

    pq = [(0.0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in adj.get(u, {}).items():
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[end] == float("inf"):
        return None, None

    # reconstruct path
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[end]

@app.get("/solve_shortest_path/")
async def get_shortest_path(start_node_id: str, end_node_id: str):
    """
    Try the imported dijkstra first (whatever signature it has).
    If that fails, use the local fallback implementation (_dijkstra_local).
    """
    global active_graph
    if active_graph is None:
        return {"Solver Error": "No active graph, please upload a graph first."}

    # ensure we have an adjacency dict
    adj = active_graph
    if not isinstance(adj, dict) or not all(isinstance(v, dict) for v in adj.values()):
        return {"Solver Error": "Active graph is not an adjacency map. Please re-upload your JSON."}

    s = str(start_node_id)
    t = str(end_node_id)
    if s not in adj or t not in adj:
        return {"Solver Error": "Invalid start or end node ID."}

    # optional small shim to satisfy some implementations that expect .nodes and [] access
    class GraphShim:
        def __init__(self, adjacency):
            self._adj = adjacency
            self.nodes = list(adjacency.keys())
        def __getitem__(self, key):
            return self._adj[key]

    G = GraphShim(adj)

    # --- 1) Try imported dijkstra with common signatures ---
    first_error = None
    try:
        # (graph, start, end)
        result = dijkstra(G, s, t)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
            path, distance = result
            return {"shortest_path": path, "total_distance": float(distance)}
        if isinstance(result, list):
            # compute distance from adjacency if only path returned
            path = result
            dist = 0.0
            for a, b in zip(path, path[1:]):
                dist += float(adj[a][b])
            return {"shortest_path": path, "total_distance": float(dist)}
        # if unexpected, fall through
    except Exception as e:
        first_error = str(e)

    try:
        # (graph, start)
        result = dijkstra(G, s)
        if isinstance(result, tuple) and len(result) == 2:
            distances, previous = result
            if t not in distances:
                return {"shortest_path": None, "total_distance": None}
            # reconstruct from 'previous'
            path = []
            cur = t
            seen = set()
            while cur is not None:
                path.append(cur)
                if cur == s:
                    break
                if cur in seen:
                    return {"shortest_path": None, "total_distance": None}
                seen.add(cur)
                cur = previous.get(cur)
            if not path or path[-1] != s:
                return {"shortest_path": None, "total_distance": None}
            path.reverse()
            return {"shortest_path": path, "total_distance": float(distances.get(t, 0.0))}
        if isinstance(result, dict):
            distances = result
            if t not in distances:
                return {"shortest_path": None, "total_distance": None}
            return {"shortest_path": None, "total_distance": float(distances[t])}
        # fall through to local fallback
    except Exception as e:
        # keep for debugging but still try local
        first_error = (first_error or "") + f"; {e}"

    # --- 2) Local fallback (always works on our adjacency) ---
    try:
        path, dist = _dijkstra_local(adj, s, t)
        if path is None:
            return {"shortest_path": None, "total_distance": None}
        return {"shortest_path": path, "total_distance": float(dist)}
    except Exception as e:
        msg = (first_error or "Imported dijkstra failed") + f"; fallback failed: {e}"
        return {"Solver Error": f"Solver failed: {msg}"}

