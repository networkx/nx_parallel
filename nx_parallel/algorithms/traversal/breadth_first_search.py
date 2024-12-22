from joblib import Parallel, delayed
from networkx.utils import py_random_state
import nx_parallel as nxp

__all__ = ["parallel_bfs"]


@nxp._configure_if_nx_active()
@py_random_state(3)
def parallel_bfs(G, source=None, get_chunks="chunks", n_jobs=None):
    """
    Perform a parallelized Breadth-First Search (BFS) on the graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    source : node, optional
        Starting node for the BFS traversal. If None, BFS is performed for all nodes.
    get_chunks : str or function (default="chunks")
        A function to divide nodes into chunks for parallel processing.
        If "chunks", the nodes are split into `n_jobs` chunks automatically.
    n_jobs : int, optional
        Number of jobs to run in parallel. If None, defaults to the number of CPUs.

    Returns
    -------
    bfs_result : dict
        A dictionary where keys are nodes and values are their BFS traversal order.
    """
    if hasattr(G, "graph_object"):
        G = G.graph_object

    if source is None:
        nodes = G.nodes
    else:
        nodes = [source]

    if n_jobs is None:
        n_jobs = nxp.get_n_jobs()

    # Create node chunks
    if get_chunks == "chunks":
        node_chunks = nxp.create_iterables(G, "node", n_jobs, nodes)
    else:
        node_chunks = get_chunks(nodes)

    # Run BFS on each chunk in parallel
    bfs_results = Parallel(n_jobs=n_jobs)(
        delayed(_bfs_chunk)(G, chunk) for chunk in node_chunks
    )

    # Combine results from all chunks
    combined_result = {}
    for result in bfs_results:
        combined_result.update(result)

    return combined_result


def _bfs_chunk(G, nodes):
    """
    Perform BFS for a subset of nodes.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    nodes : list
        A list of nodes to start BFS from.

    Returns
    -------
    bfs_result : dict
        BFS traversal order for the given subset of nodes.
    """
    bfs_result = {}
    for node in nodes:
        if node not in bfs_result:
            visited = set()
            queue = [node]
            order = 0

            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    bfs_result[current] = order
                    order += 1
                    queue.extend(
                        neighbor
                        for neighbor in G.neighbors(current)
                        if neighbor not in visited
                    )

    return bfs_result
