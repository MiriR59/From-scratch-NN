import numpy as np
from .base_initialisation import Initialisation

class Random(Initialisation):
    def init(self, inputs, neurons):
        w = (np.random.rand(neurons, inputs) - 0.5) * 2
        b = (np.random.rand(neurons, 1) - 0.5) * 2
        return w, b 
