import numpy as np
from Activations import one

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
    
    def backward(self, delta_next, w_next=None, regularization=None):
        if w_next is not None:
            self.delta = delta_next @ w_next.T * self.activation.backward()
        else:
            self.delta = delta_next * self.activation.backward()
        
        self.dw = np.swapaxes(self.input, -2, -1) @ self.delta + regularization.backward(self.w)
        self.dw = np.sum(self.dw, axis=0)
        self.db = np.sum(self.delta, axis=0)
        self.db = np.sum(self.db, axis=0, keepdims=True)
        return self.dw, self.db, self.delta, self.w

class Attention(Layer):
    def __init__(self, input_dim, attention_dim):
        self.query = Dense(input_dim, attention_dim, one(), He(), 'normal')
        self.key = Dense(input_dim, attention_dim, one(), He(), 'normal')
        self.value = Dense(input_dim, attention_dim, one(), He(), 'normal')

    def forward(self, input):
        self.Q = self.query.forward(input)
        self.K = self.key.forward(input)
        self.V = self.value.forward(input)
        self.score = (self.Q @ np.swapaxes(self.K, -2, -1)) / (np.sqrt(self.Q.shape[-1]))
        self.exp = np.exp(self.score - np.max(self.score, axis=1, keepdims=True))
        self.softmax = self.exp / np.sum(self.exp, axis=1, keepdims=True)
        return self.softmax @ self.V
        
    def backward(self, delta_next, w_next, regularization):
        if w_next is not None:
            delta_next = delta_next @ w_next.T
        
        delta_V = np.swapaxes(self.softmax, -2, -1) @ delta_next
        delta_softmax = delta_next @ np.swapaxes(self.V, -2, -1)
        delta_score = np.zeros_like(self.score)
        for i in range(self.softmax.shape[0]):
            for j in range(self.softmax.shape[1]):
                delta_score[i, j] = (np.diag(self.softmax[i, j]) - np.outer(self.softmax[i, j], self.softmax[i,j])) @ delta_softmax[i, j]

        delta_K = np.swapaxes(delta_score, -2, -1) @ self.Q / np.sqrt(self.Q.shape[-1])
        delta_Q = delta_score @ self.K / np.sqrt(self.Q.shape[-1])

        self.query.backward(delta_Q, w_next=None, regularization=regularization)
        self.key.backward(delta_K, w_next=None, regularization=regularization)
        self.value.backward(delta_V, w_next=None, regularization=regularization)
        self.delta = self.query.delta + self.key.delta + self.value.delta
        self.w = None

class MultiHeadAttention(Layer):
    def __init__(self, input_dim, attention_dim, number_heads):
        self.attenton_dim = attention_dim
        self.number_heads = number_heads
        self.heads = [Attention(input_dim, attention_dim) for _ in range(number_heads)]
        self.projection_layer = Dense(attention_dim*number_heads, input_dim, one(), He(), 'normal')
    
    def forward(self, input):
        x = [head.forward(input) for head in self.heads]
        concatenate = np.concatenate(x, axis=-1)
        return self.projection_layer.forward(concatenate)
    
    def backward(self, delta_next, w_next, regularization):
        _, _, delta_projection, _ = self.projection_layer.backward(delta_next, w_next, regularization=regularization)
        delta_projection = delta_projection @ self.projection_layer.w.T
        for head, delta in zip(self.heads, np.split(delta_projection, self.number_heads, axis=-1)):
            head.backward(delta, w_next=None, regularization=regularization)
        self.delta = sum(head.delta for head in self.heads)
        self.w = None            
    
class Embedding(Layer):
    def __init__(self, vocabulary_size, embedding_depth):
        self.vocabulary_size = vocabulary_size
        self.embedding_depth = embedding_depth
        self.embedding_matrix = np.random.rand(self.vocabulary_size, self.embedding_depth)
        
    def forward(self, ID_input):
        self.ID_input = ID_input
        self.embedded = self.embedding_matrix[self.ID_input] 
        return self.embedded
        
    def backward(self, delta_next, w_next):
        self.gradient_matrix = np.zeros_like(self.embedding_matrix)
        if w_next is not None:
            self.delta = delta_next @ w_next.T
        else:
            self.delta = delta_next 

        for i, token in enumerate (self.ID_input):
            self.gradient_matrix[token] += self.delta[i]               # CHECK LOGIC OF ADDING DELTA TO WHOLE ROW
            
        return

class PositionalEmbedding(Layer):
    def __init__(self, max_input_length, embedding_depth):
        self.max_input_length = max_input_length
        self.embedding_depth = embedding_depth
        self.embedding_matrix = np.random.rand(self.max_input_length, self.embedding_depth)

    def forward(self, ID_input):
        self.ID_input = ID_input
        if (ID_input.shape[1]) > self.max_input_length:
            raise ValueError("Input sentence is too long")
        else:
            self.embedded = self.embedding_matrix[:ID_input.shape[1]]
            self.embedded = self.embedded.reshape(1, self.embedded.shape[0], self.embedded.shape[1])
            self.embedded = np.repeat(self.embedded, ID_input.shape[0], axis=0)
            return self.embedded
    
    def backward(self, delta_next, w_next):
        self.gradient_matrix = np.zeros_like(self.embedding_matrix)
        if w_next is not None:
            self.delta = delta_next @ w_next.T
        else:
            self.delta = delta_next
        if self.delta.ndim == 3:
            self.delta = np.sum(self.delta, axis=0)

        for i in range(self.ID_input.shape[-1]):
            self.gradient_matrix[i] += self.delta[i]
        
        return

class Embedding_block:
    def __init__(self, embedding_depth, vocabulary_size, max_input_length):
        self.embed = Embedding(vocabulary_size, embedding_depth)
        self.positional_embed = PositionalEmbedding(max_input_length, embedding_depth)

    def forward(self, ID_input):
        self.embedded = self.embed.forward(ID_input) + self.positional_embed.forward(ID_input)
        return self.embedded
    
    def backward(self, delta_next, w_next, regularization=None):
        self.embed.backward(delta_next, w_next)
        self.positional_embed.backward(delta_next, w_next)

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
