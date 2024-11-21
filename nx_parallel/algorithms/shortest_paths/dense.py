"""Floyd-Warshall algorithm for shortest paths."""

from joblib import Parallel, delayed
import nx_parallel as nxp
import math

__all__ = [
    "floyd_warshall",
]


def floyd_warshall(G, weight="weight", blocking_factor=None):
    """
    Parallel implementation of Floyd warshall using the tiled floyd warshall algorithm  [1]_.


    networkx.floyd_warshall_numpy : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.dense.floyd_warshall_numpy.html

    Parameters
    ----------
    blocking_factor : number
        The number used for divinding the adjacency matrix in sub-matrix.
        The default blocking factor is get by finding the optimal value
        for the core available

    Returns
    -------
    A : 2D array
        All pairs shortest paths Graph adjacency matrix

    References
    ----------
    .. [1] Gary J. Katz and Joseph T. Kider Jr:
    All-Pairs Shortest-Paths for Large Graphs on the GPU, 2008.

    """
    if hasattr(G, "graph_object"):
        G = G.graph_object
    undirected = not G.is_directed()
    nodelist = list(G)
    A = _adjacency_matrix(G, weight, nodelist, undirected)
    n = G.number_of_nodes()

    total_cores = nxp.get_n_jobs()

    if blocking_factor is None:
        blocking_factor, is_prime = _find_nearest_divisor(n, total_cores)
    no_of_primary = n // blocking_factor

    for primary_block in range(no_of_primary):
        k_start = (primary_block * n) // no_of_primary
        k_end = k_start + (n // no_of_primary) - 1
        if is_prime and primary_block == no_of_primary - 1:
            k_end = k_end + (n % no_of_primary)
        k = (k_start, k_end)
        # Phase 1: Compute Primary block
        # Execute Normal floyd warshall for the primary block submatrix
        _partial_floyd_warshall_numpy(A, k, k, k)

        # Phase 2: Compute Cross block

        params = []
        for block in range(no_of_primary):
            # skip the primary block computed in phase 1
            if block != primary_block:
                # append the actual indices of the matrix by multiply the block number with the blocking factor
                if is_prime and block == no_of_primary - 1:
                    block_coord = (block * blocking_factor, n - 1)
                else:
                    block_coord = _block_range(blocking_factor, block)
                params.append((block_coord, k))
                params.append((k, block_coord))

        Parallel(n_jobs=(no_of_primary - 1) * 2, require="sharedmem")(
            delayed(_partial_floyd_warshall_numpy)(A, k, i, j) for (i, j) in params
        )

        # Phase 3: Compute remaining
        params.clear()
        for block_i in range(no_of_primary):
            for block_j in range(no_of_primary):
                # skip all block previously computed, so skip every block with primary block value
                if block_i != primary_block or block_j != primary_block:
                    i_range = _block_range(blocking_factor, block_i)
                    j_range = _block_range(blocking_factor, block_j)
                    if is_prime:
                        if block_i == no_of_primary - 1:
                            i_range = (block_i * blocking_factor, n - 1)
                        if block_j == no_of_primary - 1:
                            j_range = (block_j * blocking_factor, n - 1)
                    params.append((i_range, j_range))
        Parallel(n_jobs=(no_of_primary - 1) ** 2, require="sharedmem")(
            delayed(_partial_floyd_warshall_numpy)(A, k, i, j) for (i, j) in params
        )
    dist = _matrix_to_dict(A, nodelist)

    return dist


def _partial_floyd_warshall_numpy(A, k_iteration, i_iteration, j_iteration):
    """
    Compute partial FW in the determined sub-block for the execution of
    parallel tiled FW.

    Parameters
    ----------
    A : 2D numpy.ndarray
        matrix that reppresent the adjacency matrix of the graph

    k_iteration : tuple
        range (start-end) of the primary block in the current iteration

    i_iteration : tuple
        range (start-end) of the rows in the current block computed

    j_iteration : tuple
        range (start-end) of the columns in the current block computed

    Returns
    -------
    A : 2D numpy.ndarray
        adjacency matrix updated after floyd warshall
    """
    for k in range(k_iteration[0], k_iteration[1] + 1):
        for i in range(i_iteration[0], i_iteration[1] + 1):
            for j in range(j_iteration[0], j_iteration[1] + 1):
                A[i][j] = min(A[i][j], (A[i][k] + A[k][j]))


def _block_range(blocking_factor, block):
    return (block * blocking_factor, (block + 1) * blocking_factor - 1)


def _calculate_divisor(i, x, y):
    if x % i == 0:
        divisor1 = result2 = i
        result1 = divisor2 = x // i
        # difference1 = abs((result1 - 1) ** 2 - y)
        difference1 = abs(result1 - y)
        # difference2 = abs((result2 - 1) ** 2 - y)
        difference2 = abs(result2 - y)

        if difference1 < difference2:
            return divisor1, result1, difference1
        else:
            return divisor2, result2, difference2
    return None


def _find_nearest_divisor(x, y):
    """
    find the optimal value for the blocking factor parameter

    Parameters
    ----------
    x : node number

    y : cpu core available
    """
    if x < y:
        return 1, False
    # Find the square root of x
    sqrt_x = int(math.sqrt(x)) + 1

    # Execute the calculation in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_divisor)(i, x, y) for i in range(2, sqrt_x)
    )

    # Filter out None results
    results = [r for r in results if r is not None]

    if len(results) <= 0:
        # This check if a number is prime, although repeat process with a non prime number
        best_divisor, _ = _find_nearest_divisor(x - 1, y)
        return best_divisor, True
    # Find the best divisor
    best_divisor, _, _ = min(results, key=lambda x: x[2])

    return best_divisor, False


def _adjacency_matrix(G, weight, nodelist, undirected):
    """
    Generate an adjacency python matrix

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the array.

    weight : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for
        the edge weight. If an edge does not have that attribute, then the
        value 1 is used instead.

    Returns
    -------
    A : 2D array
        Graph adjacency matrix
    """

    n = len(nodelist)
    # Initialize the adjacency matrix with infinity values
    A = [[float("inf") for _ in range(n)] for _ in range(n)]

    # Set diagonal elements to 0 (distance from node to itself)
    for i in range(n):
        A[i][i] = 0

    def process_edge(src, dest, attribute, undirected):
        src_idx = nodelist.index(src)
        dest_idx = nodelist.index(dest)
        A[src_idx][dest_idx] = attribute.get(weight, 1.0)
        if undirected:
            A[dest_idx][src_idx] = attribute.get(weight, 1.0)

    # Parallel processing of edges, modifying A directly
    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(process_edge)(src, dest, attribute, undirected)
        for src, dest, attribute in G.edges(data=True)
    )
    return A


def _matrix_to_dict(A, nodelist):
    """
    Convert a matrix (list of lists) to a dictionary of dictionaries.

    Parameters
    ----------
    A : list of lists
        The adjacency matrix to be converted.

    Returns
    -------
    dist : dict
        The resulting dictionary of distance.
    """
    dist = {i: {} for i in nodelist}

    def process_row(row, i):
        for column, j in enumerate(nodelist):
            dist[i][j] = A[row][column]

    # Parallel processing of rows, modifying dist directly
    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(process_row)(row, i) for row, i in enumerate(nodelist)
    )

    return dist
