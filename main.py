import numpy as np
import tensorflow as tf
from dataset import Dataset
from neuralNetwork import NeuralNetwork
from softMaxClassifier import SoftMaxClassifier

if __name__=="__main__":
    dataset = Dataset()
    neuralNetwork = NeuralNetwork()
    neuralNetwork.addLayer(10, len(dataset.XTrain[0]), tf.nn.softmax)
    
    softMaxClassifier = SoftMaxClassifier(neuralNetwork, 0.1, 50)
    softMaxClassifier.gradientDescent(dataset.XTrain, dataset.yTrain)
    