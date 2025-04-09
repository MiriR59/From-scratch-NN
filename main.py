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
    
    def backward(self, w_next, delta):
        self.delta = w_next.T @ delta * self.activation.backward(self.output)      # Get first delta_next from delta_next = loss.backward(output, truth) and fake w_next with array of ones or smthng, or resolve last layer outside class in for loop
        self.dw = self.delta @ self.input.T                                        # Check multiplication logic
        self.db = np.sum(self.delta, axis=1)
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
                self.layers[i].backward(np.ones_like(self.layers[i].b), self.delta)
                
            else:
                self.delta =  (w[i].T @ self.delta) * self.loss_f.backward() * self.layer[i].activation.backward()
        return

    # def update_weights

# --- Test loop ---
alfa = 1e-2
epochs = 200000
batch_size = 2
# --- XOR problem ---
dataset = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
true = np.array([[0], [1], [1], [0]])

initialize = Random()
loss_function = BCE()

input_layer = Dense(2, 10, sigmoid(), initialize)
hidden_1 = Dense(10, 10, sigmoid(), initialize)
output_layer = Dense(10, 1, sigmoid(), initialize)

nn = Neural_network(BCE(), input_layer, hidden_1, output_layer)

for epoch in range(epochs):
    total_loss = 0

    for l in range(dataset.shape[0] // batch_size):
        batch = dataset[(l):(l+batch_size), :].T
        batch_true = true[(l):(l+batch_size), :].T
        x_pred = nn.forward_pass(batch)

        loss_value = nn.loss(x_pred, batch_true)
        total_loss += loss_value

        nn.backpropagation()


losses = []
for epoch in range(epochs):
    total_loss = 0
    
    for l in range(dataset.shape[0] // batch_size):
        batch = dataset[(l):(l+batch_size), :].T
        batch_true = true[(l):(l+batch_size), :].T
        x_pred = forward_pass(batch)

        loss = loss_function.forward(x_pred ,batch_true)

        dw, db = backpropagation(batch_true, w, output, network)
    
        w, b = gradient_descent(alfa, w, b, dw, db)
    
        loss_c = loss(output[-1], batch_true)
        total_loss += loss_c
        
    losses.append(total_loss)
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')




def sigmoid(x):
    '''
    x = output before activation
    '''
    return 1 / (1 + np.exp(-x))

# --- Binary cross-entropy loss ---
def loss(x, y):
    '''
    x = predicted output
    y = correct answer
    '''
    fix = 1e-9
    x = np.clip(x, fix, 1 - fix)  # Ensure we dont get log 0
    return np.mean(-(y * np.log(x) + (1 - y) * np.log(1 - x)))

# --- Gradient descent in output---
def gradient_descent(alfa, w, b, dw, db):
    '''
    alfa - learning rate
    w - weight array
    b - bias array
    dw - weight gradient
    db - bias gradient
    '''
    for i in range(len(w)):
        w[i] -= alfa * dw[i]
        b[i] -= alfa * db[i]
        
    return w, b

# --- NN parameters initialization ---
def NN_init(network):
    '''
    network - architecture list defined beelow
    '''
    weights = []
    biases = []
    for i in range(len(network)-1):
        w = (np.random.rand(network[i+1], network[i]) - 0.5) * 2
        weights.append(w)
        b = (np.random.rand(network[i+1], 1) - 0.5) * 2
        biases.append(b)
    return weights, biases

# --- Forward pass ---
def forward_pass(batch, network, w, b):
    '''
    batch - input matrix with batch_size rows
    network - architecture defined below
    w - list of weights
    b - list of biases
    '''   
    output = [batch.T]
    
    for i in range(len(w)):
        out = sigmoid((w[i] @ output[i]) + b[i])
        output.append(out)
    return output

# --- Backpropagation ---
def backpropagation(true, w, output, network):
    '''
    true - ground truth
    w - list of weights
    output - list of activation outputs
    network - architecture defined below
    '''
    delta = []
    delta_w = []
    delta_b = []
    for i in range(len(network) -1):
        if i == 0:
            delta_h = (output[-1] - true) * output[-1] * (1 - output[-1]) / batch_size
        else:
            delta_h = (w[len(network) - 1 - i].T @ delta[i-1] ) * output[len(network) - 1 - i] * (1 - output[len(network) - 1 - i])
        delta.append(delta_h)
        delta_weight = delta_h @ output[len(network) - 2 - i].T
        delta_w.insert(0, delta_weight)
        delta_b.insert(0, delta_h.sum(axis=1, keepdims=True))
    return delta_w, delta_b

# --- Define architecture ---
network = [2, 10, 10, 1]    # Inputs, hidden layers, output
w, b = NN_init(network) # Initialize weights and biases
alfa = 1e-2             # Learning rate
epochs = 200000
batch_size = 4

# --- XOR problem ---
dataset = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])    # Dataset
true = np.array([[0], [1], [1], [0]])                   # Ground truth     

# --- Main loop ---
losses = []
for epoch in range(epochs):
    total_loss = 0
    
    for l in range(dataset.shape[0] // batch_size):
        batch = dataset[(l):(l+batch_size), :]
        batch_true = true[(l):(l+batch_size), :].T
    
        output = forward_pass(batch, network, w, b)
    
        dw, db = backpropagation(batch_true, w, output, network)
    
        w, b = gradient_descent(alfa, w, b, dw, db)
    
        loss_c = loss(output[-1], batch_true)
        total_loss += loss_c
        
    losses.append(total_loss)
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')

# --- Results control ---
final = []
for i in range(dataset.shape[0]):
    batch = dataset[i:i+1]
    final_post = forward_pass(batch, network, w, b)
    final.insert(0, final_post[-1][0, 0])

print(true.T)
print(final)