import numpy as np
import tensorflow as tf

class SoftMaxClassifier:
    _neuralNetwork = None
    _learningRate = None
    _iterations = None

    def __init__(self, neuralNetwork, learningRate, iterations):
        self._neuralNetwork = neuralNetwork
        self._learningRate = neuralNetwork
        self._iterations = iterations

    def gradientDescent(self, X, y):

        for i in range(self._iterations):
            with tf.GradientTape as tape:
                predictions = self._neuralNetwork.output(X)
                loss = self._crossEntropy(predictions, y)


    def _crossEntropy(self, predictions, y):
        return -1 * np.sum(np.log(y @ predictions.T))

    # Accuracy calculated as: [number of instances where max probability index is equal to correct class] / [number of instances]
    def _calculateAccuracy(self, predictions, y):
        return np.sum(np.argmax(predictions)==y) / np.size(y, 0)



