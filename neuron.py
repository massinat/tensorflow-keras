# A single neuron with configurable activation function

class Neuron:
    _weights = None
    _bias = None
    _activationFunction = None

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def __init__(self, initialWeights, initialBias, activationFunction):
        self._weights = initialWeights
        self._bias = initialBias
        self._activationFunction = activationFunction

    def output(self, X):
        return self._activationFunction(self._weights @ X.T + self._bias)

    def updateWeights(self, newWeights):
        self._weights = newWeights