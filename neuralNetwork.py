# A neural network composed by multiple layers and neurons

import numpy as np
from neuron import Neuron

class NeuralNetwork:
    _layers = None

    def __init__(self):
        self._layers = []

    # Add a new layer of neurons to the current layers
    def addLayer(self, numberOfNeurons, inputSize, activationFunction):
        newLayer = []

        for i in range(numberOfNeurons):
            newLayer.append(Neuron(np.zeros(inputSize), 0, activationFunction))
        
        self._layers.append(newLayer)

    # Fast forward pass
    def output(self, X):
        currentInput = X

        for i in range(len(self._layers)):
            lastLayerOutput = []

            for j in range(len(self._layers[i])):
                lastLayerOutput.append(self._layers[i][j].output(currentInput))

            currentInput = lastLayerOutput

        return lastLayerOutput