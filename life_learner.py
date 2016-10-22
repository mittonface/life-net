from gol_generator import Life
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def build_life_nn(size= (25, 25), num_samples=100, random_density=0.3, epochs=100):

    l = Life()

    # our input and output will both be a full lifescape
    dataset = SupervisedDataSet(size[0]*size[1], size[0]*size[1])

    # add a bunch of random samples
    for i in range(num_samples):
        # start with a random lifescape
        l.randomize_lifescape(percent_activated=random_density)

        # save the current state as input. Ravel will turn it into a 1D list
        input = l.lifescape.ravel()

        # do a step of life
        l.do_step()

        # save the next state as the output
        output = l.lifescape.ravel()

        # add this as a training example
        dataset.addSample(input, output)


    # build a simple network
    network = FeedForwardNetwork()

    # input layer has one node per cell
    inputLayer = LinearLayer(size[0]*size[1])

    # in theory, there should be a minimum number of nodes here that would be able to capture the complexity
    # needed. It probably would be reasonable easy to figure out. Each node only needs to know about 8 other
    # input nodes and the weight function for these would be reasonable simple. I'm just going to make it a
    # full 625, which I *think* is overkill
    hiddenLayer = SigmoidLayer(size[0]*size[1])  # in theory there is a mim

    # output layer is an entire lifescape
    outputLayer = LinearLayer(size[0]*size[1])

    # let the network know about these layers
    network.addInputModule(inputLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outputLayer)

    # define the connections between the layers.
    in_to_hidden = FullConnection(inputLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outputLayer)

    # add connections to the network
    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)

    # this probaly does something
    network.sortModules()

    #
    trainer = BackpropTrainer(network, dataset)

    for i in range(epochs):
        training_error = trainer.train()

    return (network, training_error)
