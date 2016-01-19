import networkx as nx


def are_d_separated(g, query_var1, query_var2, *given_vars):
    """
    Algorithm from http://www.autonlab.org/tutorials/bayesinf05.pdf, p6-7

    :param g: Directed graph
    :param query_var1:
    :param query_var2:
    :param given_vars:
    :return:
    """
    return all(is_path_blocked(g, path, given_vars)
               for path in nx.all_simple_paths(g.to_undirected(), query_var1, query_var2))


def is_path_blocked(graph, path, given_vars):
    for i, node in enumerate(path[1:-1], 1):
        if node in given_vars:
            if graph.has_edge(node, path[i-1]) and graph.has_edge(node, path[i+1]):  # tail to tail
                return True
            if graph.has_edge(path[i-1], node) and graph.has_edge(node, path[i+1]):  # right right
                return True
            if graph.has_edge(path[i+1], node) and graph.has_edge(node, path[i-1]):  # left left
                return True
        elif graph.has_edge(path[i-1], node) and graph.has_edge(path[i+1], node):  # head to head
            return True

    return False
