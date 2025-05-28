import numpy as np

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
        self.z = self.w @ self.input + self.b
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, w_next, delta_next, regularization):
        self.delta = w_next.T @ delta_next * self.activation.backward()
        self.dw = self.delta @ self.input.T + regularization.backward(self.w)
        self.db = np.sum(self.delta, axis=1, keepdims=True)
        return self.dw, self.db, self.delta, self.w

class Xavier(Initialisation):
    def init (self, type, inputs, neurons):
        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(neurons, inputs) - 0.5) * 2 * np.sqrt(6 / (inputs + neurons))

        elif type == 'normal':
            w = (np.random.randn(neurons, inputs)) * np.sqrt(2 / (inputs + neurons))

        else:
            print('Choose viable Initialisation type')

        return w, b
    
class He(Initialisation):
    def init (self, type, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons

        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(neurons, inputs) - 0.5) * 2 * np.sqrt(6 / inputs)

        elif type == 'normal':
            w = (np.random.randn(neurons, inputs)) * np.sqrt(2 / inputs)

        else:
            print('Choose viable Initialisation type')

        return w, b
    
class Random(Initialisation):
    def init(self, inputs, neurons):
        w = (np.random.rand(neurons, inputs) - 0.5) * 2
        b = (np.random.rand(neurons, 1) - 0.5) * 2
        return w, b 