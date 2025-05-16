import numpy as np
from .base_activation import Activation

class ReLU(Activation):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self):
        return np.where(self.out > 0, 1, 0)