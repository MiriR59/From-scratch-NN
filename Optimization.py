import numpy as np
from Layers import Embedding_block, Dense, Attention

class ADAM:
    def __init__(self, alfa, beta1=0.9, beta2=0.999, epsilon=1e-9):
        self.alfa = alfa
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []
        self.v = []
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []

    def optimize(self, layers):
        for layer in layers:
            if isinstance(layer, Embedding_block):
                for embedding in ([layer.embed, layer.positional_embed]):
                # m[0], v[0] = EMBEDDING, m[1], v[1] = POSITIONAL EMBEDDING

                    if self.t == 0:
                        embedding.m = np.zeros_like(embedding.embedding_matrix)
                        embedding.v = np.zeros_like(embedding.embedding_matrix)
                
                    embedding.m = self.beta1 * embedding.m + (1 - self.beta1) * embedding.gradient_matrix
                    embedding.v = self.beta2 * embedding.v + (1 - self.beta2) * (embedding.gradient_matrix ** 2)

                    if self.t > 0:
                        m_cor = embedding.m / (1 - self.beta1 ** self.t)
                        v_cor = embedding.v / (1 - self.beta2 ** self.t)
                    else:
                        m_cor = embedding.m
                        v_cor = embedding.v                            
                    embedding.embedding_matrix -= self.alfa * m_cor / (np.sqrt(v_cor) + self.epsilon)
            
            elif isinstance(layer, Attention):
                self.optimize_dense(layer.query, self.t)
                self.optimize_dense(layer.key, self.t)
                self.optimize_dense(layer.value, self.t)
            
            elif isinstance(layer, Dense):
                self.optimize_dense(layer, self.t)

        self.t += 1
    
    def optimize_dense(self, layer, t):
        if isinstance(layer, Dense):
            if self.t == 0:
                layer.m_w = (np.zeros_like(layer.w))
                layer.v_w = (np.zeros_like(layer.w))
                layer.m_b = (np.zeros_like(layer.b))
                layer.v_b = (np.zeros_like(layer.b))
        
            layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dw
            layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dw ** 2)
            layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.db
            layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.db ** 2)
        
            if self.t > 0:
                m_w_cor = layer.m_w / (1 - self.beta1 ** self.t)
                v_w_cor = layer.v_w / (1 - self.beta2 ** self.t)
                m_b_cor = layer.m_b / (1 - self.beta1 ** self.t)
                v_b_cor = layer.v_b / (1 - self.beta2 ** self.t)
            else:
                m_w_cor = layer.m_w
                v_w_cor = layer.v_w
                m_b_cor = layer.m_b
                v_b_cor = layer.v_b
        
            layer.w -= self.alfa * m_w_cor / (np.sqrt(v_w_cor) + self.epsilon)
            layer.b -= self.alfa * m_b_cor / (np.sqrt(v_b_cor) + self.epsilon)   

        else:
            raise RuntimeError('Layer isnt Dense, cant optimise')
        
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
