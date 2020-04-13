import unittest
import tensorflow as tf
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
        tf.debugging.assert_equal(target._layers[0][0]._weights, tf.zeros(20))
        tf.debugging.assert_equal(target._layers[0][0]._bias, tf.Variable(0, dtype=tf.dtypes.float32))
        self.assertEqual(target._layers[0][0]._activationFunction, activationFunction1)

        target.addLayer(5, 15, activationFunction2)
        self.assertEqual(len(target._layers), 2)
        self.assertEqual(len(target._layers[1]), 5)
        tf.debugging.assert_equal(target._layers[1][0]._weights, tf.zeros(15))
        tf.debugging.assert_equal(target._layers[1][0]._bias, tf.Variable(0, dtype=tf.dtypes.float32))
        self.assertEqual(target._layers[1][0]._activationFunction, activationFunction2)

    def test_getWeightsForLayer(self):
        target = NeuralNetwork()
        target.addLayer(10, 20, lambda x: x**2)

        tf.debugging.assert_equal(target.getWeightsForLayer(0), tf.zeros([10, 20]))

    def test_getBiasesForLayer(self):
        target = NeuralNetwork()
        target.addLayer(10, 20, lambda x: x**2)

        tf.debugging.assert_equal(target.getBiasesForLayer(0), tf.zeros([10]))

    def test_predict(self,):
        target = NeuralNetwork()
        target.addLayer(10, 10, lambda x: x**2)

        prediction = target.predict(tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.dtypes.float32), [tf.zeros([10, 10])], [tf.zeros([10])])

        tf.debugging.assert_equal(prediction, tf.zeros(10))

if __name__=="__main__":
    unittest.main()