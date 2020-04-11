
import math
import numpy as np
import tensorflow as tf


class SoftMaxClassifier:
    _X = None
    _y = None
    _numberOfNeurons = 0

    def __init__(self, X, y, numberOfNeurons):
        self._X = X
        self._y = y
        self._numberOfNeurons = numberOfNeurons

    # Accuracy calculated as: [number of instances where max probability index is equal to correct class] / [number of instances]
    def _calculateAccuracy(self, predictions, y):
        return np.sum(np.argmax(predictions)==y) / np.size(y, 0)

    # Cross entropy loss calculated as: -[sum of log values for predicted classes]
    def _crossEntropy(self, predictions, y):
        return -1 * np.sum(np.log(y @ predictions.T))

    def _forwardPass(self, X, weights, bias):
        return tf.nn.softmax(weights @ X.T + bias)

    def gradientDescent(self):

    