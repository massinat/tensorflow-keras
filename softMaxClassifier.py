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
        optimazer = tf.keras.optimizers.Adam()
        weights = tf.Variable([self._neuralNetwork.getWeightsForLayer(0)])
        biases = tf.Variable([self._neuralNetwork.getBiasesForLayer(0)])
        
        for i in range(self._iterations):
            with tf.GradientTape() as tape:
                
                predictions = self._neuralNetwork.predict(X, weights, biases)
                loss = tf.Variable(self._crossEntropy(predictions, y))
                
                print(loss)

                print(f"Iteration {i} started.")
                print(weights)
                print(biases)

                gradients = tape.gradient(loss, [weights, biases])
                optimazer.apply_gradients(zip(gradients, [weights, biases]))

                # print(f"Iteration {i} completed.")
                # print(weights)
                # print(biases)

    def _crossEntropy(self, predictions, y):
        #return -1 * np.sum(np.log(y @ predictions.T))
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits (predictions, labels=y)
        loss = tf.reduce_mean(x_entropy)

        return loss

    # Accuracy calculated as: [number of instances where max probability index is equal to correct class] / [number of instances]
    def _calculateAccuracy(self, predictions, y):
        return np.sum(np.argmax(predictions)==y) / np.size(y, 0)



