import unittest
import tensorflow as tf
from neuron import Neuron

class NeuronTest(unittest.TestCase):

    def test_init(self):
        weights = tf.constant([1, 2, 3])
        bias = tf.constant(4)

        target = Neuron(weights, bias)

        tf.debugging.assert_equal(target._weights, weights)
        tf.debugging.assert_equal(target.weights, weights)
        tf.debugging.assert_equal(target._bias, bias)
        tf.debugging.assert_equal(target.bias, bias)

    def test_predict(self):
        target = Neuron(tf.constant([1, 2, 3]), tf.constant(4))
       
        prediction = target.predict(tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))

        tf.debugging.assert_equal(prediction, tf.constant([18, 36, 54, 72]))

    def test_updateWeights(self):
        weights1 = tf.constant([4, 5, 6])

        target = Neuron(weights1, 7)
        tf.debugging.assert_equal(target._weights, weights1)

        weights2 = tf.constant([8, 9, 10])
        target.updateWeights(weights2)
        tf.debugging.assert_equal(target._weights, weights2)

    def test_updateBias(self):
        bias1 = tf.constant(7)
        target = Neuron([4, 5, 6], bias1)
        self.assertEqual(target._bias, bias1)

        bias2 = tf.constant(8)
        target.updateBias(bias2)
        self.assertEqual(target._bias, bias2)

if __name__=="__main__":
    unittest.main()
