import numpy as np
from Layers import Embedding

class ADAM:
    def __init__(self, alfa, beta1=0.9, beta2=0.999, epsilon=1e-9):
        self.alfa = alfa
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []

    def optimize(self, layers):
        i = 0
        for layer in layers:
            if isinstance(layer, Embedding):
                if self.t == 0:
                    self.m = np.zeros_like(layer.embedding_matrix)
                    self.v = np.zeros_like(layer.embedding_matrix)
                    
                self.m = self.beta1 * self.m + (1 - self.beta1) * layer.gradient_matrix
                self.v = self.beta2 * self.v + (1 - self.beta2) * (layer.gradient_matrix ** 2)
                
                if self.t > 0:
                    self.m_cor = self.m / (1 - self.beta1 ** self.t)
                    self.v_cor = self.v / (1 - self.beta2 ** self.t)
                else:
                    self.m_cor = self.m
                    self.v_cor = self.v
                layer.embedding_matrix -= self.alfa * self.m_cor / (np.sqrt(self.v_cor) + self.epsilon)
                
            else:
                if self.t == 0:
                    self.m_w.append(np.zeros_like(layer.w))
                    self.v_w.append(np.zeros_like(layer.w))
                    self.m_b.append(np.zeros_like(layer.b))
                    self.v_b.append(np.zeros_like(layer.b))
    
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.dw
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (layer.dw ** 2)
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)
    
                if self.t > 0:
                    self.m_w_cor = self.m_w[i] / (1 - self.beta1 ** self.t)
                    self.v_w_cor = self.v_w[i] / (1 - self.beta2 ** self.t)
                    self.m_b_cor = self.m_b[i] / (1 - self.beta1 ** self.t)
                    self.v_b_cor = self.v_b[i] / (1 - self.beta2 ** self.t)
                else:
                    self.m_w_cor = self.m_w[i]
                    self.v_w_cor = self.v_w[i]
                    self.m_b_cor = self.m_b[i]
                    self.v_b_cor = self.v_b[i]
    
                layer.w -= self.alfa * self.m_w_cor / (np.sqrt(self.v_w_cor) + self.epsilon)
                layer.b -= self.alfa * self.m_b_cor / (np.sqrt(self.v_b_cor) + self.epsilon)
                
                i +=1

        self.t += 1
        
class gradient_descent:
    def __init__(self, alfa):
        self.alfa = alfa
        
    def optimize(self, layers):
        for layer in layers:
            layer.w -= self.alfa * layer.dw
            layer.b -= self.alfa * layer.db

class Scheduler:
    def __init__(self, optimizer, **kwargs):
        self.t = 0
        self.optimizer = optimizer
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def step(self):
        self.update()
        self.t += 1
        
    def update(self):
        raise NotImplementedError()
        
class LR_exponential(Scheduler):
    def __init__(self, optimizer, decay_rate):
        super().__init__(optimizer, decay_rate=decay_rate, alfa=optimizer.alfa)

    def update(self):
        self.optimizer.alfa = self.alfa * (self.decay_rate ** self.t)
        
class LR_cosine_annealing(Scheduler):
    def __init__(self, optimizer, T, min_alfa, max_alfa):
        super().__init__(optimizer, T=T, min_alfa=min_alfa, max_alfa=max_alfa)

    def update(self):
        self.optimizer.alfa = self.min_alfa + (self.max_alfa - self.min_alfa) * (1 + np.cos(np.pi * self.t / self.T)) / 2
        
class LR_decay(Scheduler):
    def __init__(self, optimizer, decay_rate, interval):
        super().__init__(optimizer, decay_rate=decay_rate, interval=interval)

    def update(self):
        if self.t % self.interval == 0:
            self.optimizer.alfa *= self.decay_rate