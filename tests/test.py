from pyboolnet.boolnet import BooleanBeliefNetwork

anmr_path = '../example_data/wanmr.json'
net = BooleanBeliefNetwork.from_json(anmr_path)
print(net.get_prob('R', True))