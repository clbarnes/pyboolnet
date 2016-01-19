import re

VAR_REGEX = re.compile('(?<=^p\()\w+(?=\||\))')
PARENT_REGEX = re.compile('(?<=\|)(\w|\s|,)+(?=\)$)')


def parse_var_parents(s):
    """
    Parse a probability string and return the variable name and the names of its conditions.

    :param s: "p(A|B,C)"-style string
    :return: tuple(variable_name, [list, of, parent, variables])
    """
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