# A neural network composed by multiple layers and neurons

import tensorflow as tf
from neuron import Neuron

class NeuralNetwork:
    _layers = None

    def __init__(self):
        tf.random.set_seed(5)
        self._layers = []

    # Add a new layer of neurons to the current layers
    def addLayer(self, numberOfNeurons, inputSize):
        newLayer = []

        for i in range(numberOfNeurons):
            newLayer.append(Neuron(tf.random.uniform(shape=[inputSize]), tf.random.uniform(shape=[1])))
        
        self._layers.append(newLayer)

    def getWeightsForLayer(self, layer):
        return [x.weights for x in self._layers[layer]]

    def getBiasesForLayer(self, layer):
        return [x.bias for x in self._layers[layer]]

    # Fast forward pass
    def predict(self, X, weights, biases):
        return self._predictWithSoftMax(X, self._layers[0])

    def _predictWithSoftMax(self, X, neurons):
        output = []

        for i in range(len(neurons)):
            output.append(tf.math.exp(neurons[i].predict(X)))

        #divisors = tf.reduce_sum(output, axis=0)
        #print(divisors)
        print(output)

        return output
