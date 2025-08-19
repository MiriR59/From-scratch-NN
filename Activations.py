import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()

class ReLU(Activation):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self):
        return np.where(self.out > 0, 1, 0)

class sigmoid(Activation):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self):
        return self.out * (1 - self.out)
    
class SiLU(Activation):
    def forward(self, x):
        self.x = x
        self.out = x / (1 + np.exp(-x))
        return self.out
    
    def backward(self):
        return 1 / (1 + np.exp(-self.x)) + self.out * (1 - 1 / (1 + np.exp(-self.x)))
    
class tanh(Activation):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self):
        return 1 - self.out ** 2
    
class one(Activation):
    def forward(self, x):
        return x
    
    def backward(self):
        return 1
