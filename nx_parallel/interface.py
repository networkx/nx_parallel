import networkx as nx
from typing import Optional, Union
from operator import attrgetter

from nx_parallel import algorithms

__all__ = ["BackendInterface", "ParallelGraph"]

NX_GTYPES = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]

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
]


class ParallelGraph:
    """
    A wrapper class for networkx.Graph, networkx.DiGraph, networkx.MultiGraph,
    and networkx.MultiDiGraph.
    """

    __networkx_backend__ = "parallel"

    def __init__(
        self,
        graph_object: Optional[NX_GTYPES] = None,
    ):
        if graph_object is None:
            self.graph_object = nx.Graph()
        elif isinstance(graph_object, NX_GTYPES):
            self.graph_object = graph_object
        else:
            self.graph_object = nx.Graph(graph_object)

    @property
    def is_multigraph(self):
        return self.graph_object.is_multigraph()

    @property
    def is_directed(self):
        return self.graph_object.is_directed()

    def __str__(self):
        return f"Parallel{self.graph_object}"


class BackendInterface:
    """BackendInterface class for parallel algorithms."""

    # assign the imported functions to class attributes
    for attr in ALGORITHMS:
        if "." in attr:
            module_name, func_name = attr.rsplit(".", 1)
            setattr(
                locals()["BackendInterface"],
                func_name,
                attrgetter(attr)(algorithms),
            )
        else:
            setattr(
                locals()["BackendInterface"],
                attr,
                getattr(algorithms, attr),
            )

    @staticmethod
    def convert_from_nx(graph: Optional[NX_GTYPES], *args, **kwargs) -> ParallelGraph:
        """
        Convert a networkx.Graph, networkx.DiGraph, networkx.MultiGraph,
        or networkx.MultiDiGraph to a ParallelGraph.
        """
        if isinstance(graph, ParallelGraph):
            return graph
        return ParallelGraph(graph)

    @staticmethod
    def convert_to_nx(result, *, name: Optional[str] = None) -> Optional[NX_GTYPES]:
        """
        Convert a ParallelGraph to a networkx.Graph, networkx.DiGraph,
        networkx.MultiGraph, or networkx.MultiDiGraph.
        """
        if isinstance(result, ParallelGraph):
            return result.graph_object
        return result
