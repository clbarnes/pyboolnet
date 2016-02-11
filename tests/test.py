from pyboolnet.boolnet import BooleanBeliefNetwork
import json


def default_settings():
    anmr = BooleanBeliefNetwork.from_json('../example_data/anmr.json')
    wanmr = BooleanBeliefNetwork.from_json('../example_data/wanmr.json')
    wanmr2 = BooleanBeliefNetwork.from_json('../example_data/wanmr_pouget.json')

    return anmr, wanmr, wanmr2


def rare_aggressive(p_W=0.05, p_AgW=0.9):
    nets = []
    for net_name in ['anmr', 'wanmr', 'wanmr_pouget']:
        with open('../example_data/{}.json'.format(net_name)) as f:
            data = json.load(f)

        if 'w' in net_name:
            data['p(W)'] = p_W
            data['p(A|W)'] = {'T': p_AgW, 'F': 0}
        else:
            data['p(A)'] = p_W * p_AgW

        nets.append(BooleanBeliefNetwork(data))

    assert len(nets) == 3
    return nets


def print_rolling(anmr, wanmr, wanmr2, R_state, **kwargs):
    anmr_roll = anmr.get_prob('R', R_state, **kwargs)
    wanmr_roll = wanmr.get_prob('R', R_state, **kwargs)
    wanmr2_roll = wanmr2.get_prob('R', R_state, **kwargs)

    information_str = ' , '.join('{}={}'.format(var, state) for var, state in sorted(kwargs.items())) + ' , '

    print('\tPr( R={} | {}ANMR model )  = {}'.format(R_state, information_str, anmr_roll))
    print('\tPr( R={} | {}WANMR model ) = {}'.format(R_state, information_str, wanmr_roll))
    print('\tPr( R={} | {}WANMR_Pouget model ) = {}'.format(R_state, information_str, wanmr2_roll))


def print_all(anmr, wanmr, wanmr2):
    partial_print = lambda R_state, **kwargs: print_rolling(anmr, wanmr, wanmr2, R_state, **kwargs)

    print('Correct rolling (avoid death)')
    partial_print(True, A=True)

    print('Errant rolling (waste energy)')
    partial_print(True, A=False)

    print('Pre-emptive rolling')
    partial_print(True, A=False, W=True)


if __name__ == '__main__':
    print('DEFAULT SETTINGS')
    anmr, wanmr, wanmr2 = default_settings()
    print_all(anmr, wanmr, wanmr2)

    print('\n\nRARE, AGGRESSIVE WASPS')
    anmr_RA, wanmr_RA, wanmr2_RA = rare_aggressive()
    print_all(anmr_RA, wanmr_RA, wanmr2_RA)
