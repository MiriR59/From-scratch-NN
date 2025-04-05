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
                
                
def sigmoid(x):
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