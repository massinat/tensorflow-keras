import numpy as np
import tensorflow as tf
from dataset import Dataset
from neuralNetwork import NeuralNetwork

if __name__=="__main__":
    dataset = Dataset()
    neuralNetwork = NeuralNetwork()
    neuralNetwork.addLayer(10, len(dataset.XTrain[0]), tf.nn.softmax)



    print(neuralNetwork.output(dataset.XTrain))

    