import numpy as np

class Loss:
    def forward(self, x_pred, x_true):
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()
        
class MSE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        self.loss = np.mean((x_pred - x_true) ** 2)
        return self.loss
    
    def backward(self):
        return 2 * (self.x_pred - self.x_true) / len(self.x_pred)          # Check len(x_pred) logic for batching

class BCE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        x_pred = np.clip(x_pred, 1e-10, 1 - 1e-10)
        self.loss = np.mean(-(x_true * np.log(x_pred) + (1 - x_true) * np.log(1 - x_pred)))
        return self.loss
        
    def backward(self):
        return (self.x_pred - self.x_true) / (self.x_pred * (1 - self.x_pred))
    
class CCE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        x_pred = np.clip(x_pred, 1e-10, 1 - 1e-10)
        self.loss = - np.sum(x_true * np.log(x_pred))
        return self.loss
    
    def backward(self):
        return (self.x_pred - self.x_true) / (self.x_pred * (1 - self.x_pred))
