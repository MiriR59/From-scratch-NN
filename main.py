import numpy as np

## --- NN from scratch --- ##
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
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))

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
network = [2, 10, 1]    # Inputs, hidden layers, output
w, b = NN_init(network) # Initialize weights and biases
alfa = 1e-2             # Learning rate
epochs = 100000
batch_size = 3

# --- XOR problem ---
input_array = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])    # Dataset
true = np.array([[0], [1], [1], [0]])                       # Ground truth     

# --- Test loop ---
for l in range(input_array.shape[0] // batch_size):
    batch = input_array[(l):(l+batch_size), :]
    batch_true = true[(l):(l+batch_size), :].T
    
    output = forward_pass(batch, network, w, b)
    
    dw, db = backpropagation(batch_true, w, output, network)
# --- Main loop ---
for epoch in range(epochs):
    total_loss = 0
    
    for l in range(input_array.shape[0] // batch_size):
        batch = input_array[(2*l):(2*l+2), :]
        
        out_post = forward_pass(batch, network, w, b)
        
        loss_c = loss(out_post[-1], true[l]) / input_array.shape[0]
        total_loss += loss_c

        dw, db = backpropagation(true[l], w, out_post, network)
        
        w, b = gradient_descent(alfa, w, b, dw, db)

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')         

# --- Results control ---
final = []
for i in range(input_array.shape[0]):
    batch = input_array[i:i+1]
    final_post = forward_pass(batch, network, w, b)
    final.insert(0, final_post[-1][0, 0])

print(true.T)
print(final)