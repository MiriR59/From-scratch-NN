import numpy as np
from Tokenizer import tokenizer

class Layer:
    def __init__(self, *args):
        raise NotImplementedError()
            
    def forward(self, input):
        raise NotImplementedError()
        
    def backward(self, *args):
        raise NotImplementedError()
        
class Initialisation:
    def init(self, shape):
        raise NotImplementedError()
        
class Dense(Layer):
    def __init__(self, inputs, neurons, activation, initialisation, init_type):
        self.activation = activation
        self.w, self.b = initialisation.init(init_type, inputs, neurons)
    
    def forward(self, input):
        self.input = input
        self.z = self.input @ self.w + self.b
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, w_next, delta_next, regularization):
        self.delta = delta_next @ w_next.T * self.activation.backward()
        self.dw = self.input.T @ self.delta + regularization.backward(self.w)
        self.db = np.sum(self.delta, axis=0, keepdims=True)
        return self.dw, self.db, self.delta, self.w

class Embedding(Layer):
    def __init__(self, vocabulary_size, embedding_depth):
        self.vocabulary_size = vocabulary_size
        self.embedding_depth = embedding_depth
        self.embedding_matrix = np.random.rand(self.vocabulary_size, self.embedding_depth)
        
    def forward(self, ID_input):
        self.ID_input = ID_input
        self.embedded = np.zeros((1, self.embedding_depth))
        
        for token in self.ID_input:
            self.embedded = np.concatenate((self.embedded, self.embedding_matrix[token:token+1, :]), axis=0)
            
        return self.embedded[1:, :]
        
    def backward(self, w_next, delta_next, regularization):
        self.gradient_matrix = np.zeros_like(self.embedding_matrix)
        
        for i, token in enumerate (self.ID_input):
            self.embedded[token] += delta_next[i]               # CHECK LOGIC OF ADDING DELTA TO WHOLE ROW
            
        return self.gradient_matrix

class Xavier(Initialisation):
    def init (self, type, inputs, neurons):
        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(inputs, neurons) - 0.5) * 2 * np.sqrt(6 / (inputs + neurons))

        elif type == 'normal':
            w = (np.random.randn(inputs, neurons)) * np.sqrt(2 / (inputs + neurons))

        else:
            print('Choose viable Initialisation type')

        return w, b
    
class He(Initialisation):
    def init (self, type, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons

        'Type: uniform/normal'
        b = np.zeros((1, neurons))
        if type == 'uniform':
            w = (np.random.rand(inputs, neurons) - 0.5) * 2 * np.sqrt(6 / inputs)

        elif type == 'normal':
            w = (np.random.randn(inputs, neurons)) * np.sqrt(2 / inputs)

        else:
            print('Choose viable Initialisation type')

        return w, b
    
class Random(Initialisation):
    def init(self, inputs, neurons):
        w = (np.random.rand(inputs, neurons) - 0.5) * 2
        b = (np.random.rand(1, neurons) - 0.5) * 2
        return w, b 