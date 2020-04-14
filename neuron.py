# A single neuron with configurable activation function

import tensorflow as tf

class Neuron:
    _weights = None
    _bias = None

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def __init__(self, initialWeights, initialBias):
        self._weights = initialWeights
        self._bias = initialBias

    def predict(self, X):
        return tf.tensordot(self._weights, tf.transpose(X), 1) + self._bias

    def updateWeights(self, newWeights):
        self._weights = newWeights

    def updateBias(self, newBias):
        self._bias = newBias