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
        exp = np.exp(x_pred - np.max(x_pred, axis=2, keepdims=True))
        self.prob = exp / np.sum(exp, axis=2, keepdims=True)
        self.x_true = x_true
        loss = -np.sum(x_true * np.log(self.prob + 1e-15)) / x_pred.shape[0]
        return loss

    def backward(self):
        return self.prob - self.x_true
