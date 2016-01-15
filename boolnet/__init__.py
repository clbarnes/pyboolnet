import networkx as nx
import json
import re
import itertools


VAR_REGEX = re.compile('(?<=^p\()\w+(?=\||\))')
PARENT_REGEX = re.compile('(?<=\|)(\w|\s|,)+(?=\)$)')


class BooleanBeliefNetwork:
    def __init__(self, data):
        g = nx.DiGraph()
        for key, prob_dict in data.items():
            var, parents = parse_var_parents(key)
            truth_table = parse_truth_dict(prob_dict)

            g.add_node(var, parents=tuple(parents), init_probs=truth_table)
            for parent in parents:
                g.add_edge(parent, var)

        assert nx.is_directed_acyclic_graph(g)
        self._g = g

    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            data = json.load(f)

        return cls(data)

    def draw(self, out_path, args=None):
        agraph = nx.to_agraph(self._g)
        agraph.layout('dot', args=args)
        agraph.draw(out_path)

    def get_prob(self, var_name, var_state, **parent_states):
        if set(parent_states) == set(self._g.predecessors(var_name)):
            return self._get_prob_simple(var_name, var_state, **parent_states)
        elif len(self._g.successors(var_name)) == 0 and len(parent_states) == 0:
            return self.joint(**{var_name: var_state})
        else:
            raise NotImplementedError('Inference too complicated')

    def _get_prob_simple(self, var_name, var_state, **parent_states):
        parents = self._g.node[var_name]['parents']
        parent_state_tuple = tuple(parent_states[parent] for parent in parents)
        p = self._g.node[var_name]['init_probs'][var_state][parent_state_tuple]
        return p

    def joint(self, **var_states):
        if set(var_states).issuperset(self._g.nodes_iter()):
            return self._complete_joint(**var_states)
        else:
            return self._marginal_joint(**var_states)

    def _complete_joint(self, **var_states):
        prod = 1
        for var in self._g.nodes_iter():
            prod *= self.get_prob(var, var_states[var], **var_states)

        return prod

    def _marginal_joint(self, **var_states):
        total = 0
        marginal_vars = [var for var in self._g.nodes_iter() if var not in var_states]
        for marginal_states in itertools.product([True, False], repeat=len(marginal_vars)):
            marginal_var_states = dict(zip(marginal_vars, marginal_states))
            total += self._complete_joint(**var_states, **marginal_var_states)

        return total


def parse_var_parents(s):
    var = VAR_REGEX.search(s).group()
    try:
        parents = PARENT_REGEX.search(s).group().split(',')
    except AttributeError:
        parents = []

    return var, parents


def parse_truth_dict(d):
    """
    :param d: dict (if variable has parents) or int (if not) for probability of variable being True given parent
    states, keyed on "TF"-style strings
    :return: dict {self_state: {tuple(*parent_states): p}}
    """
    try:
        p_true = {tuple(c.lower() == 't' for c in key) : val for key, val in d.items()}
        p_false = {key: 1-val for key, val in p_true.items()}
        return {True: p_true, False: p_false}
    except AttributeError:
        return {True: {(): d}, False: {(): 1-d}}


if __name__ == "__main__":
    path = 'example_data/data.json'
    net = BooleanBeliefNetwork.from_json(path)
