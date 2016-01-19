import itertools
import json
import networkx as nx

from .str_tools import parse_var_parents, parse_truth_dict
from .network_tools import are_d_separated
from .misc import prod


class BooleanBeliefNetwork:
    """
    A Bayesian/ belief network where all variables are boolean and the joint probability distribution is known.
    """
    def __init__(self, data):
        g = nx.DiGraph()
        for key, prob_dict in data.items():
            var, parents = parse_var_parents(key)
            prob_fn = parse_truth_dict(prob_dict)

            g.add_node(var, parents=tuple(parents), init_probs=prob_fn)
            for parent in parents:
                g.add_edge(parent, var)

        assert nx.is_directed_acyclic_graph(g)
        self._g = g

    @classmethod
    def from_json(cls, json_path):
        """
        Generate boolean belief network from a JSON file

        :param json_path:
        :return:
        """
        with open(json_path) as f:
            data = json.load(f)

        return cls(data)

    def draw(self, out_path, args=None):
        """
        Use graphviz' dot layout to generate an image of the belief network

        :param out_path:
        :param args:
        :return:
        """
        agraph = nx.to_agraph(self._g)
        agraph.layout('dot', args=args)
        agraph.draw(out_path)

    # def _get_prob_dep(self, var_name, var_state, **known_states):
    #     """
    #     Master method for getting any single conditional probability for any conditions (WIP)
    #
    #     :param var_name: Name of query variable
    #     :param var_state: State of query variable
    #     :param known_states: keyword arguments of name=state pairs for all known variables
    #     :return: float of conditional probability
    #     """
    #
    #     if var_name in known_states:
    #         return 1 if known_states[var_name] == var_state else 0
    #
    #     # neighbours = self._g.predecessors(var_name) + self._g.successors(var_name)
    #     # known_states = {key: value for key, value in known_states.items() if key in neighbours}
    #     if set(known_states) == set(self._g.predecessors(var_name)):
    #         return self._get_prob_simple(var_name, var_state, **known_states)
    #     elif len(known_states) == 0:
    #         return self.joint(**{var_name: var_state})
    #     elif set(known_states) == set(self._g.successors(var_name)):
    #         return self._get_prob_inverse(var_name, var_state, **known_states)
    #     else:
    #         raise NotImplementedError('Inference too complicated')

    def get_prob(self, var_name, var_state, **known_states):
        p_this_state = self.joint(**{var_name: var_state}, **known_states)
        p_other_state = self.joint(**{var_name: not var_state}, **known_states)

        return p_this_state / (p_this_state + p_other_state)

    # def get_neighbours(self, var_name):
    #     """
    #     Get all neighbours of a variable.
    #
    #     :param var_name: str, name of variable
    #     :return: list of names of parent and child variables
    #     """
    #     return self._g.predecessors(var_name) + self._g.successors(var_name)

    def _get_prob_simple(self, var_name, var_state, **parent_states):
        """
        Probability of a variable taking a given state when its parents' and only its parents' states are known.

        :param var_name: Name of query variable
        :param var_state: State of query variable
        :param known_states: keyword arguments of name=state pairs for all known variables
        :return: float of conditional probability
        """
        parents = self._g.node[var_name]['parents']
        parent_state_tuple = tuple(parent_states[parent] for parent in parents)
        p = self._g.node[var_name]['init_probs'](var_state, parent_state_tuple)
        return p

    def get_all_predecessors(self, var):
        """
        Recursively the set of nodes which are predecessors of the given variable.

        :param var: node
        :return: set of nodes
        """
        preds = set()
        preds.update(self._g.predecessors(var))
        for item in preds.copy():
            preds.update(self.get_all_predecessors(item))

        return preds

    def _are_pair_conditionally_independent(self, query_var1, query_var2, *given_vars):
        return are_d_separated(self._g, query_var1, query_var2, given_vars)

    def are_conditionally_independent(self, query_vars, *given_vars):
        """
        Return whether the set of query variables are conditionally independent given the states of some other variables

        :param query_vars: sequence of query variables
        :param given_vars: optional arguments of given variables
        :return: bool
        """
        ret = True
        for query_var1, query_var2 in itertools.combinations(query_vars, 2):
            ret = ret and self._are_pair_conditionally_independent(query_var1, query_var2, *given_vars)

        return ret

    def joint(self, **var_states):
        """
        Compute the joint probability of given variable states. Any unknown states are marginalised over.

        :param var_states: keyword arguments of name=state pairs for all known variables
        :return: float of probability
        """
        if set(var_states).issuperset(self._g.nodes_iter()):
            return self._complete_joint(**var_states)
        else:
            return self._marginal_joint(**var_states)

    def _complete_joint(self, **var_states):
        """
        Compute the joint probability of given variable states when all states are known.

        :param var_states: keyword arguments of name=state pairs for all variables
        :return: float of probability
        """
        return prod(self._get_prob_simple(var, var_states[var], **var_states) for var in self._g.nodes_iter())

    def _marginal_joint(self, **var_states):
        """
        Compute the joint probability of given variable states where some states need marginalising over.

        :param var_states: keyword arguments of name=state pairs for all known variables
        :return: float of probability
        """
        total = 0
        marginal_vars = [var for var in self._g.nodes_iter() if var not in var_states]
        for marginal_states in itertools.product([True, False], repeat=len(marginal_vars)):
            marginal_var_states = dict(zip(marginal_vars, marginal_states))
            total += self._complete_joint(**var_states, **marginal_var_states)

        return total

    # def _get_prob_inverse(self, var_name, var_state, **known_states):
    #     children = self._g.successors(var_name)
    #     known_states = {var: state for var, state in known_states.items() if var in children}
    #
    #     if not self.are_conditionally_independent(children, var_name):
    #         raise NotImplementedError('Inference too complicated')
    #
    #     numerator = self.get_prob(var_name, var_state)
    #     for var, state in known_states.items():
    #         numerator *= self.get_prob(var, state, **{var_name: var_state})
    #
    #     denominator = self.joint(**known_states)
    #
    #     return numerator/denominator
