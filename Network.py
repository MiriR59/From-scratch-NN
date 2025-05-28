import numpy as np

class Neural_network:
    def __init__(self, loss_f, optimizer, *layers, regularization=None):
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.regularization = regularization if regularization else Null()
        self.layers = list(layers)
        
    def forward_pass(self, x):
        self.x = x
        for i in range(len(self.layers)):
            self.x = self.layers[i].forward(self.x)
        return self.x

    def loss(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        return self.loss_f.forward(self.x_pred, self.x_true) + self.regularization.forward(self.layers)
    
    def backpropagation(self):
        self.delta = self.loss_f.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].backward(np.ones((self.delta.shape[1], self.delta.shape[1])), self.delta, self.regularization)
                
            else:
                self.layers[i].backward(self.layers[i+1].w, self.layers[i+1].delta, self.regularization)

    def optimize(self):
        self.optimizer.optimize(self.layers)
        
class Regularization:
    def __init(self, *args):
        raise NotImplementedError()
        
    def forward(self, layers):
        raise NotImplementedError()
        
    def backward(self, *args):
        raise NotImplementedError()
        
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
    
class Null:
    def forward(self, layers):
        return 0
    
    def backward(self, w):
        return 0