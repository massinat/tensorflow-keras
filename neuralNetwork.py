# A neural network composed by multiple layers and neurons

import tensorflow as tf
from neuron import Neuron

class NeuralNetwork:
    _layers = None

    def __init__(self):
        tf.random.set_seed(5)
        self._layers = []

    # Add a new layer of neurons to the current layers
    def addLayer(self, numberOfNeurons, inputSize, activationFunction):
        newLayer = []

        for i in range(numberOfNeurons):
            newLayer.append(Neuron(tf.random.uniform(shape=[inputSize]), tf.random.uniform(shape=[1]), activationFunction))
        
        self._layers.append(newLayer)

    def getWeightsForLayer(self, layer):
        return [x.weights for x in self._layers[layer]]

    def getBiasesForLayer(self, layer):
        return [x.bias for x in self._layers[layer]]

    # Fast forward pass
    def predict(self, X, weights, biases):
        currentInput = X

        for i in range(len(self._layers)):
            lastLayerOutput = []

            for j in range(len(self._layers[i])):
                self._layers[i][j].updateWeights(weights[i][j])
                self._layers[i][j].updateBias(biases[i][j])
                lastLayerOutput.append(self._layers[i][j].predict(currentInput))

            currentInput = lastLayerOutput

        return tf.Variable(lastLayerOutput, dtype=tf.dtypes.float32)