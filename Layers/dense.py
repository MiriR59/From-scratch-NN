import numpy as np
from .base_layer import Layer

class Dense(Layer):
    def __init__(self, inputs, neurons, activation, initialisation, init_type):
        self.activation = activation
        self.w, self.b = initialisation.init(init_type, inputs, neurons)
    
    def forward(self, input):
        self.input = input
        self.z = self.w @ self.input + self.b
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, w_next, delta_next, regularization):
        self.delta = w_next.T @ delta_next * self.activation.backward()
        self.dw = self.delta @ self.input.T + regularization.backward(self.w)
        self.db = np.sum(self.delta, axis=1, keepdims=True)
        return self.dw, self.db, self.delta, self.w