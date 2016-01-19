from pyboolnet.boolnet import BooleanBeliefNetwork

anmr = BooleanBeliefNetwork.from_json('../example_data/anmr.json')
wanmr = BooleanBeliefNetwork.from_json('../example_data/wanmr.json')

print(anmr.get_prob('R', True, A=False))
print(wanmr.get_prob('R', True, A=False))
