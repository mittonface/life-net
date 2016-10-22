from gol_generator import Life
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection

network = FeedForwardNetwork()

inputLayer = LinearLayer(25*25)
hiddenLayer = SigmoidLayer(25*25)
outputLayer = LinearLayer(25*25)

network.addInputModule(inputLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outputLayer)

in_to_hidden = FullConnection(inputLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outputLayer)

network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()

