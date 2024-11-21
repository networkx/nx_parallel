from operator import attrgetter
import networkx as nx
from nx_parallel import algorithms

__all__ = ["BackendInterface", "ParallelGraph"]


ALGORITHMS = [
    # Bipartite
    "node_redundancy",
    # Isolates
    "number_of_isolates",
    # Vitality
    "closeness_vitality",
    # Tournament
    "is_reachable",
    "tournament_is_strongly_connected",
    # Centrality
    "betweenness_centrality",
    "edge_betweenness_centrality",
    "closeness_centrality",
    # Efficiency
    "local_efficiency",
    # Shortest Paths : generic
    "all_pairs_all_shortest_paths",
    # Shortest Paths : weighted graphs
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path_length",
    "all_pairs_dijkstra_path",
    "all_pairs_bellman_ford_path_length",
    "all_pairs_bellman_ford_path",
    "johnson",
    # Clustering
    "square_clustering",
    # Shortest Paths : unweighted graphs
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    # Approximation
    "approximate_all_pairs_node_connectivity",
    # Connectivity
    "connectivity.all_pairs_node_connectivity",
    "floyd_warshall",
]


class ParallelGraph:
    """A wrapper class for networkx.Graph, networkx.DiGraph, networkx.MultiGraph,
    and networkx.MultiDiGraph.
    """

    __networkx_backend__ = "parallel"

    def __init__(self, graph_object=None):
        if graph_object is None:
            self.graph_object = nx.Graph()
        elif isinstance(graph_object, nx.Graph):
            self.graph_object = graph_object
        else:
            self.graph_object = nx.Graph(graph_object)

    def is_multigraph(self):
        return self.graph_object.is_multigraph()

    def is_directed(self):
        return self.graph_object.is_directed()

    def __str__(self):
        return f"Parallel{self.graph_object}"


def assign_algorithms(cls):
    """Class decorator to assign algorithms to the class attributes."""
    for attr in ALGORITHMS:
        # get the function name by parsing the module hierarchy
        func_name = attr.rsplit(".", 1)[-1]
        setattr(cls, func_name, attrgetter(attr)(algorithms))
    return cls


@assign_algorithms
class BackendInterface:
    """BackendInterface class for parallel algorithms."""

    @staticmethod
    def convert_from_nx(graph, *args, **kwargs):
        """Convert a networkx.Graph, networkx.DiGraph, networkx.MultiGraph,
        or networkx.MultiDiGraph to a ParallelGraph.
        """
        if isinstance(graph, ParallelGraph):
            return graph
        return ParallelGraph(graph)

    @staticmethod
    def convert_to_nx(result, *, name=None):
        """Convert a ParallelGraph to a networkx.Graph, networkx.DiGraph,
        networkx.MultiGraph, or networkx.MultiDiGraph.
        """
        if isinstance(result, ParallelGraph):
            return result.graph_object
        return result
