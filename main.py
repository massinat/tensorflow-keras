import numpy as np
from dataset import Dataset
from neuralNetwork import NeuralNetwork

if __name__=="__main__":
    dataset = Dataset()
    neuralNetwork = neuralNetwork()

    neuralNetwork.addLayer()

# Accuracy calculated as: [number of instances where max probability index is equal to correct class] / [number of instances]
    def _calculateAccuracy(self, predictions, y):
        return np.sum(np.argmax(predictions)==y) / np.size(y, 0)

    # Cross entropy loss calculated as: -[sum of log values for predicted classes]
    def _crossEntropy(self, predictions, y):
        return -1 * np.sum(np.log(y @ predictions.T))