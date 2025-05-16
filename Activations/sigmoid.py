import numpy as np
from .base_activation import Activation

class sigmoid(Activation):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self):
        return self.out * (1 - self.out)