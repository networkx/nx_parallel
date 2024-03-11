from joblib import Parallel, delayed
from networkx.algorithms.shortest_paths.weighted import (
    _weight_function,
    _dijkstra,
    _bellman_ford,
    single_source_bellman_ford_path,
)
import nx_parallel as nxp

__all__ = [
    "all_pairs_bellman_ford_path",
    "johnson",
]


def all_pairs_bellman_ford_path(G, weight="weight", get_chunks="chunks"):
    """The parallel implementation first divides the nodes into chunks and then
    creates a generator to lazily compute shortest paths for each node_chunk, and
    then employs joblib's `Parallel` function to execute these computations in
    parallel across all available CPU cores.

    Parameters
    ------------
    get_chunks : str, function (default = "chunks")
        A function that takes in an iterable of all the nodes as input and returns
        an iterable `node_chunks`. The default chunking is done by slicing the
        `G.nodes` into `n` chunks, where `n` is the number of CPU cores.

    networkx.all_pairs_bellman_ford_path : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path.html#all-pairs-bellman-ford-path
    """

    def _process_node_chunk(node_chunk):
        return [
            (node, single_source_bellman_ford_path(G, node, weight=weight))
            for node in node_chunk
        ]

    if hasattr(G, "graph_object"):
        G = G.graph_object

    nodes = G.nodes
    total_cores = nxp.cpu_count()

    if get_chunks == "chunks":
        num_in_chunk = max(len(nodes) // total_cores, 1)
        node_chunks = nxp.chunks(nodes, num_in_chunk)
    else:
        node_chunks = get_chunks(nodes)

    paths_chunk_generator = (
        delayed(_process_node_chunk)(node_chunk) for node_chunk in node_chunks
    )

    for path_chunk in Parallel(n_jobs=nxp.cpu_count())(paths_chunk_generator):
        for path in path_chunk:
            yield path


def johnson(G, weight="weight", get_chunks="chunks"):
    """The parallel computation is implemented by dividing the
    nodes into chunks and computing the shortest paths using Johnson's Algorithm
    for each chunk in parallel.

    Parameters
    ------------
    get_chunks : str, function (default = "chunks")
        A function that takes in an iterable of all the nodes as input and returns
        an iterable `node_chunks`. The default chunking is done by slicing the
        `G.nodes` into `n` chunks, where `n` is the number of CPU cores.

    networkx.johnson : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.johnson.html#johnson
    """
    if hasattr(G, "graph_object"):
        G = G.graph_object

    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)

    # Calculate distance of shortest paths
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    # Update the weight function to take into account the Bellman--Ford
    # relaxation distances.
    def new_weight(u, v, d):
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    def dist_path(v):
        paths = {v: [v]}
        _dijkstra(G, v, new_weight, paths=paths)
        return paths

    def _johnson_subset(chunk):
        return {node: dist_path(node) for node in chunk}

    total_cores = nxp.cpu_count()
    if get_chunks == "chunks":
        num_in_chunk = max(len(G.nodes) // total_cores, 1)
        node_chunks = nxp.chunks(G.nodes, num_in_chunk)
    else:
        node_chunks = get_chunks(G.nodes)

    results = Parallel(n_jobs=total_cores)(
        delayed(_johnson_subset)(chunk) for chunk in node_chunks
    )
    return {v: d_path for result_chunk in results for v, d_path in result_chunk.items()}
