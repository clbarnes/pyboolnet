import networkx as nx


def are_d_separated(g, query_var1, query_var2, *given_vars):
    return all(is_path_blocked(g, path, given_vars)
               for path in nx.all_simple_paths(g.to_undirected(), query_var1, query_var2))


def is_path_blocked(graph, path, given_vars):
    for i, node in enumerate(path[1:-1], 1):
        if node in given_vars:
            if graph.has_edge(node, path[i-1]) and graph.has_edge(node, path[i+1]):  # tail to tail
                return True
            if graph.has_edge(path[i-1], node) and graph.has_edge(node, path[i+1]):  # right right
                return True
            if graph.has_edge(path[i+1], node) and graph.has_edge(node, path[i-1]):
                return True
        elif graph.has_edge(path[i-1], node) and graph.has_edge(path[i+1], node):  # head to head
            return True

    return False


def shave_leaves(g):
    """
    Successively return all leaf nodes, pruning existing leaves from the graph with each iteration

    :param g: nx.DiGraph
    :return: iterator of lists of leaf nodes
    """
    assert g.is_directed()
    g = g.copy()
    while len(g.nodes()) > 0:
        leaves = [node for node in g.nodes_iter() if g.out_degree()[node] == 0]
        g.remove_nodes_from(leaves)
        yield leaves