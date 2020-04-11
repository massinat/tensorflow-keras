import unittest
import numpy as np
from neuron import Neuron

class NeuronTest(unittest.TestCase):

    def test_init(self):
        activationFunction = lambda x: x

        target = Neuron([1, 2, 3], 4, activationFunction)

        self.assertEqual(target._weights, [1, 2, 3])
        self.assertEqual(target._bias, 4)
        self.assertEqual(target._activationFunction, activationFunction)

    def test_output(self):
        target = Neuron([1, 2, 3], 4, lambda x: x**2)
       
        output = target.output(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))

        np.testing.assert_array_equal(output, np.array([324, 1296, 2916, 5184]))

    def test_updateWeights(self):
        target = Neuron([4, 5, 6], 7, lambda x: x)
        self.assertEqual(target._weights, [4, 5, 6])

        target.updateWeights([8, 9, 10])
        self.assertEqual(target._weights, [8, 9, 10])

if __name__=="__main__":
    unittest.main()
