import numpy as np
from .base_activation import Activation

class tanh(Activation):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self):
        return 1 - self.out ** 2