import numpy as np
from .base_initialisation import Initialisation

class He(Initialisation):
    def init (self, type, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons

        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(neurons, inputs) - 0.5) * 2 * np.sqrt(6 / inputs)

        elif type == 'normal':
            w = (np.random.randn(neurons, inputs)) * np.sqrt(2 / inputs)

        else:
            print('Choose viable Initialisation type')

        return w, b