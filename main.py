import numpy as np

## --- NN from scratch --- ##
# --- Activation class definitions ---

class Activation:
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, x):
        raise NotImplementedError()
    
class sigmoid(Activation):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, x):
        return self.out * (1 - self.out)
    
class ReLU(Activation):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, x):
        return np.where(self.out > 0, 1, 0)
    
class tanh(Activation):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, x):
        return 1 - self.out ** 2

class Swish(Activation):
    def forward(self, x):
        self.out = x / (1 + np.exp(-x))
        return self.out
    
    def backward(self, x):
        return 1 / (1 + np.exp(-x)) + self.out * (1 - 1 / (1 + np.exp(-x)))

# --- Loss class definitions ---
class Loss:
    def forward(self, x_pred, x_true):
        raise NotImplementedError()
        
    def backward(self, x_pred, x_true):
        raise NotImplementedError()
        
class BCE(Loss):
    def forward(self, x_pred, x_true):
        fix = 1e-10
        x_pred = np.clip(x_pred, fix, 1 - fix)
        self.loss = np.mean(-(x_true * np.log(x_pred) + (1 - x_true) * np.log(1 - x_pred)))
        return self.loss
        
    def backward(self, x_pred, x_true):
        return (x_pred - x_true) / (x_pred * (1 - x_pred))
    
class MSE(Loss):
    def forward(self, x_pred, x_true):
        self.loss = np.mean((x_pred - x_true) ** 2)
        return self.loss
    
    def backward(self, x_pred, x_true):
        return 2 * (x_pred - x_true) / len(x_pred)          # Check len(x_pred) logic for batching
    
class CCE(Loss):
    def forward(self, x_pred, x_true):
        self.loss = - np.sum(x_true * np.log(x_pred))
        return self.loss
    def backward(self, x_pred, x_true):
        return (x_pred - x_true) / (x_pred * (1 - x_pred))
                
# --- Initialisation class definitions ---
class Initialisation:
    def init(self, shape):
        raise NotImplementedError()
    
class Random(Initialisation):
    def init(self, inputs, neurons):
        w = (np.random.rand(neurons, inputs) - 0.5) * 2
        b = (np.random.rand(neurons, 1) - 0.5) * 2
        return w, b 
 
# --- Layer class definitions ---
class Layer:
    def forward(self, input):
        raise NotImplementedError()

    def backward(self, *args):
        raise NotImplementedError()

class Dense(Layer):
    def __init__(self, inputs, neurons, activation, initialisation):
        self.activation = activation
        self.w, self.b = initialisation.init(inputs, neurons)
    
    def forward(self, input):
        self.input = input
        self.output = self.activation.forward(self.w @ self.input + self.b)
        return self.output
    
    def backward(self, w_next, delta_next):
        self.delta = w_next.T @ delta_next * self.activation.backward(self.output)
        self.dw = self.delta @ self.input.T                                # Check multiplication logic
        self.db = np.sum(self.delta, axis=1, keepdims=True)
        return self.dw, self.db, self.delta, self.w

# --- General NN definition ---
class Neural_network:
    def __init__(self, loss_f, *layers):
        self.layers = list(layers)
        self.loss_f = loss_f

    def forward_pass(self, x):
        self.x = x
        for i in range(len(self.layers)):
            self.x = self.layers[i].forward(self.x)
        return self.x

    def loss(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        return self.loss_f.forward(x_pred, x_true)
    
    def backpropagation(self):
        self.delta = self.loss_f.backward(self.x_pred, self.x_true)
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].backward(np.ones((self.delta.shape[0], self.delta.shape[0])), self.delta)
                
            else:
                self.layers[i].backward(self.layers[i+1].w, self.layers[i+1].delta)
                # self.delta =  (w[i].T @ self.delta) * self.loss_f.backward() * self.layer[i].activation.backward()
        return

    def gradient_descent(self):
        for i in range(len(self.layers)):
            self.layers[i].w -= alfa * self.layers[i].dw
            self.layers[i].b -= alfa * self.layers[i].db
        return
    
# --- Test loop ---
alfa = 1e-0
epochs = 100000
batch_size = 4
# --- XOR problem ---
dataset = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
true = np.array([[0], [1], [1], [0]])

initialize = Random()
loss_function = BCE()

hidden_1 = Dense(2, 10, sigmoid(), initialize)
hidden_2 = Dense(10, 10, sigmoid(), initialize)
output_layer = Dense(10, 1, sigmoid(), initialize)

nn = Neural_network(BCE(), hidden_1, hidden_2, output_layer)

losses = []
for epoch in range(epochs):
    total_loss = 0

    for l in range(dataset.shape[0] // batch_size):
        batch = dataset[(l):(l+batch_size), :].T
        batch_true = true[(l):(l+batch_size), :].T
        x_pred = nn.forward_pass(batch)

        loss_value = nn.loss(x_pred, batch_true)
        total_loss += loss_value

        nn.backpropagation()
        nn.gradient_descent()
    total_loss /= (dataset.shape[0] / batch_size)
    losses.append(total_loss)

    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')
    
# --- Results control ---
final = nn.forward_pass(dataset.T)
print(true.T)
print(final)