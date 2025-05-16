import numpy as np
from .base_loss import Loss

class MSE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        self.loss = np.mean((x_pred - x_true) ** 2)
        return self.loss
    
    def backward(self):
        return 2 * (self.x_pred - self.x_true) / len(self.x_pred)          # Check len(x_pred) logic for batching