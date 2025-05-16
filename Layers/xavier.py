import numpy as np
from .base_initialisation import Initialisation

class Xavier(Initialisation):
    def init (self, type, inputs, neurons):
        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(neurons, inputs) - 0.5) * 2 * np.sqrt(6 / (inputs + neurons))

        elif type == 'normal':
            w = (np.random.randn(neurons, inputs)) * np.sqrt(2 / (inputs + neurons))

        else:
            print('Choose viable Initialisation type')

        return w, b

