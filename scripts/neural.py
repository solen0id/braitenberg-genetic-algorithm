import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np


def random_weights(*args: int):
    weights = np.random.rand(*args)
    weights[weights < 0.5] *= -1  # randomly flip sign of weights

    return weights


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


@dataclass
class Fitness:
    x_start: float
    y_start: float
    distance: float = 0
    distance_from_start: float = 0
    distance_x: float = 0
    distance_y: float = 0
    collisions: int = 0

    def to_val(self):
        return self.distance + 3 * self.distance_from_start - self.collisions


@dataclass
class FollowLightFitness:
    light_intensity: float = 0

    def to_val(self):
        return self.light_intensity


class NeuralNetwork:
    def __init__(self, weights, biases, activation=tanh) -> None:
        self.weights = weights
        self.biases = biases
        self.activation = activation

    @classmethod
    def random(cls, n_inputs=4, n_hidden=3, n_outputs=2):
        return cls(
            weights=[
                random_weights(n_inputs, n_hidden),
                random_weights(n_hidden, n_outputs),
            ],
            biases=[random_weights(n_hidden), random_weights(n_outputs)],
        )

    @classmethod
    def from_file(cls, path):
        # load pickled neural network
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def to_file(self, path):
        # pickle neural network
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def think(self, inputs):
        # implement a simple feedforward neural network
        # with 1 hidden layer with 3 neurons
        # and  1 output layer with 2 neurons

        hidden_layer = self.activation(
            np.dot(np.array(inputs), self.weights[0]) + self.biases[0]
        )
        output_layer = np.dot(hidden_layer, self.weights[1]) + self.biases[1]

        return output_layer  # * 10  # scale output to [-10, 10] for motor control

    def mutate(self, rate=0.15, prob=0.3):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if np.random.rand() < prob:
                        self.weights[i][j][k] += np.random.uniform(-rate, rate)

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                if np.random.rand() < prob:
                    self.biases[i][j] += np.random.uniform(-rate, rate)

        return self

    def crossover(self, other):
        new_weights = []
        new_biases = []

        for i in range(len(self.weights)):
            new_weights.append(
                np.where(
                    np.random.rand(*self.weights[i].shape) < 0.5,
                    self.weights[i],
                    other.weights[i],
                )
            )

        for i in range(len(self.biases)):
            new_biases.append(
                np.where(
                    np.random.rand(*self.biases[i].shape) < 0.5,
                    self.biases[i],
                    other.biases[i],
                )
            )

        return NeuralNetwork(new_weights, new_biases, self.activation)
