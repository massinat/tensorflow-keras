import numpy as np
import tensorflow as tf
from keras.utils import np_utils

class Dataset:

    def __init__(self):
        loadedData = self._load()

        self._XTrain = np.float32(loadedData[0])
        self._yTrain = loadedData[1]
        self._XTest = np.float32(loadedData[2])
        self._yTest = loadedData[3]

    @property
    def XTrain(self):
        return self._XTrain

    @property
    def yTrain(self):
        return self._yTrain

    @property
    def XTest(self):
        return self._XTest

    @property
    def yTest(self):
        return self._yTest

    def _load(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        # load the training and test data    
        (tr_x, tr_y), (te_x, te_y) = fashion_mnist.load_data()

        # reshape the feature data
        tr_x = tr_x.reshape(tr_x.shape[0], 784)
        te_x = te_x.reshape(te_x.shape[0], 784)

        # normalise feature data
        tr_x = tr_x / 255.0
        te_x = te_x / 255.0

        print( "Shape of training features ", tr_x.shape)
        print( "Shape of test features ", te_x.shape)

        # one hot encode the training labels and get the transpose
        tr_y = np_utils.to_categorical(tr_y,10)
        tr_y = tr_y.T
        print ("Shape of training labels ", tr_y.shape)

        # one hot encode the test labels and get the transpose
        te_y = np_utils.to_categorical(te_y,10)
        te_y = te_y.T
        print ("Shape of testing labels ", te_y.shape)

        return tr_x, tr_y, te_x, te_y