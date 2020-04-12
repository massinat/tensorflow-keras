import unittest
import numpy as np
from neuralNetwork import NeuralNetwork

class NeuralNetworkTest(unittest.TestCase):

    def test_init(self):
        target = NeuralNetwork()

        self.assertEqual(target._layers, [])

    def test_addLayer(self):
        activationFunction1 = lambda x: x
        activationFunction2 = lambda x: x**2
        target = NeuralNetwork()

        target.addLayer(10, 20, activationFunction1)
        self.assertEqual(len(target._layers), 1)
        self.assertEqual(len(target._layers[0]), 10)
        np.testing.assert_array_equal(target._layers[0][0]._weights, np.zeros(20))
        self.assertEqual(target._layers[0][0]._bias, 0)
        self.assertEqual(target._layers[0][0]._activationFunction, activationFunction1)

        target.addLayer(5, 15, activationFunction2)
        self.assertEqual(len(target._layers), 2)
        self.assertEqual(len(target._layers[1]), 5)
        np.testing.assert_array_equal(target._layers[1][0]._weights, np.zeros(15))
        self.assertEqual(target._layers[1][0]._bias, 0)
        self.assertEqual(target._layers[1][0]._activationFunction, activationFunction2)

    def test_getWeightsForLayer(self):
        target = NeuralNetwork()
        target.addLayer(10, 20, lambda x: x**2)

        np.testing.assert_array_equal(target.getWeightsForLayer(0), np.zeros([10, 20]))

    def test_getBiasesForLayer(self):
        target = NeuralNetwork()
        target.addLayer(10, 20, lambda x: x**2)

        np.testing.assert_array_equal(target.getBiasesForLayer(0), np.zeros([10]))

    def test_predict(self,):
        target = NeuralNetwork()
        target.addLayer(10, 10, lambda x: x**2)

        prediction = target.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [np.zeros([10, 10])], [np.zeros([10])])

        np.testing.assert_array_equal(prediction, np.zeros([10]))

if __name__=="__main__":
    unittest.main()