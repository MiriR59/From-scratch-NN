import numpy as np
from .base_activation import Activation

class SiLU(Activation):
    def forward(self, x):
        self.x = x
        self.out = x / (1 + np.exp(-x))
        return self.out
    
    def backward(self):
        return 1 / (1 + np.exp(-self.x)) + self.out * (1 - 1 / (1 + np.exp(-self.x)))