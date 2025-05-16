import numpy as np
from .base_loss import Loss

class BCE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        x_pred = np.clip(x_pred, 1e-10, 1 - 1e-10)
        self.loss = np.mean(-(x_true * np.log(x_pred) + (1 - x_true) * np.log(1 - x_pred)))
        return self.loss
        
    def backward(self):
        return (self.x_pred - self.x_true) / (self.x_pred * (1 - self.x_pred))