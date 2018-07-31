# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 23:54:32 2018

@author: Shabaka
"""

import numpy as np


class Perceptron(object):
    def __init__(self, input_size, epochs=100):
        self.w = np.zeros(input_size)
        self.b = np.zeros(1)
        self.epochs = epochs

    def activation_fxn(self, z):
        return 1. if z >= 0 else 0.

    def predict(self, x):
        # produces a prediction of the network based on the bias
        z = self.w.T.dot(x) + self.b
        a = self.activation_fxn(z)
        return a

    def fit(self, X, y):    # rows X is example, y is true value
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                a = self.predict(X[i])
                e = y[i] - a
                # if neuron is exited or vice versa -error outputs
                self.w = self.w + e * X[i]
                self.b = self.b + e

if __name__ == '__main__':
    X =  np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]
                   ])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=4)
    perceptron.fit(X, y)   # this trains the perceptron

    accuracy = 0.    # we set training accuracy
    for i in range(y.shape[0]):
        accuracy += perceptron.predict(X[i]) == y[i]

    accuracy = accuracy / y.shape[0]

    print("{0:.4f}".format(accuracy))

    print(perceptron.w)
    print(perceptron.b)



