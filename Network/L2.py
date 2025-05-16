import numpy as np
from .base_regularization import Regularization

class L2(Regularization):
    def __init__(self, lambd):
        self.lambd = lambd
        self.L2_loss = 0

    def forward(self, layers):
        for layer in layers:
            self.L2_loss += np.sum(layer.w ** 2)

        return self.L2_loss * self.lambd
    
    def backward(self, w):
        return 2 * self.lambd * w
