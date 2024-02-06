from joblib import Parallel, delayed
from networkx.algorithms.shortest_paths.weighted import single_source_bellman_ford_path
import nx_parallel as nxp

__all__ = ["all_pairs_bellman_ford_path_chunk", "all_pairs_bellman_ford_path_no_chunk"]


def all_pairs_bellman_ford_path_no_chunk(G, weight="weight"):
    """The parallel computation is implemented by computing the
    shortest paths for each node concurrently.

    networkx.all_pairs_bellman_ford_path : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path.html#all-pairs-bellman-ford-path
    """

    def _calculate_shortest_paths_subset(source):
        return (source, single_source_bellman_ford_path(G, source, weight=weight))

    if hasattr(G, "graph_object"):
        G = G.graph_object

    cpu_count = nxp.cpu_count()

    nodes = G.nodes

    paths = Parallel(n_jobs=cpu_count, return_as="generator")(
        delayed(_calculate_shortest_paths_subset)(source) for source in nodes
    )
    return paths

def all_pairs_bellman_ford_path_chunk(G, weight="weight"):
    def _calculate_shortest_paths_subset(nodes):
        for source in nodes:
            yield (source, single_source_bellman_ford_path(G, source, weight=weight))

    if hasattr(G, "graph_object"):
        G = G.graph_object

    nodes = G.nodes

    cpu_count = nxp.cpu_count()
    num_in_chunk = max(len(nodes) // cpu_count, 1)
    node_chunks = nxp.chunks(nodes, num_in_chunk)

    paths = Parallel(n_jobs=cpu_count, return_as="generator")(
        delayed(_calculate_shortest_paths_subset)(nodes) for nodes in node_chunks
    )
    return paths
