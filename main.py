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

# --- Backpropagation ---
def backpropagation(x, y, z):
    '''
    x - predicted value
    y - correct target value
    z - output of previous layer after activation
    '''
    delta_out = (x - y) * x * (1 - x)
    dL_db = delta_out
    dL_dw = z * delta_out
    return dL_dw, dL_db

# --- Backpropagation in hidden ---
def backpropagation_hidden (x, y, w_out, n, X):
    '''
    x - predicted value
    y - correct value
    w_out - output weight array
    n - output of hidden layer after activation
    X - input data
    '''
    delta_out = (x - y) * x * (1 - x)
    delta_hidden = (delta_out * w_out) * (n * (1 - n))
  
    dL_dw_hid = (X @ delta_hidden.T).T
    dL_db_hid = np.sum(delta_hidden, axis=0)
    return dL_dw_hid, dL_db_hid

# --- Gradient descent in output---
def grad_desc(alfa, w, b, dL_dw, dL_db):
    '''
    alfa - learning rate
    w - weight array
    b - bias array
    dL_dw - weight gradient
    dL_db - bias gradient
    '''
    w_updated = w - alfa * dL_dw
    b_updated = b - alfa * dL_db
    return w_updated, b_updated

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

def forward_pass(batch, network, w, b):
    '''
    batch - matrix with single datapass in every row n_columns = network[0]
    network - architercture defined below
    w - list of weights
    b - list of biases
    '''   
    out_pre = []
    out_post = []
    for i in range(len(network)-1):
        output_pre = np.zeros([network[i+1], 1])
        output_post = np.zeros_like(output_pre)
        for k in range(network[i+1]):
            output_pre[k] += b[i][k, 0]
                
            for j in range(network[i]):
                if i == 0:
                    output_pre[k] += batch[0, j] * w[i][k, j]                   
                    
                else:
                    output_pre[k] += out_pre[i-1][j, 0] * w[i][k, j]
                    
            output_post[k] = sigmoid(output_pre[k])
        out_pre.append(output_pre)
        out_post.append(output_post)
    return out_pre, out_post

# --- Define architecture ---
network = [2, 10, 10, 1]    # 2 inputs, HL with 10 neurons, HL with 10 neurons, 1 output

w, b = NN_init(network)     # Initialize weights and biases

n_input = 2     # Input features
n_layers = 3    # Number of layers (inc output)
n_one = 10      # Neurons in 1st layer
n_two = 10      # Neurons in 2nd layer
n_out = 1       # Outputs
alfa = 1e-2     # Learning rate
epochs = 1 # Epochs

# NN definition
NN = np.zeros([n_layers, 1])

# XOR problem input
input_array = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
output_array = np.array([[0], [1], [1], [0]])
# Transpose inputs for matrix operations
X = input_array.T
y = output_array.T

# Layer output storing
n_one_out = np.zeros(n_one)


# --- TESTOVANI ---
for epoch in range(epochs):
    total_loss = 0
    
    for l in range(input_array.shape[0]):
        batch = input_array[l:l+1]
        
        # --- Forward pass ---
        outpre, outpost = forward_pass(batch, network, w, b)
        print('FORWARD PASS DONE')
        
        loss_c = loss(outpost[2], output_array[l])
        total_loss += loss_c

        # --- Backpropagation ---
        
        # --- Gradient descent ---
        
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')        




for epoch in range(epochs):
    total_loss = 0
    
    for i in range(input_array.shape[0]):
        x = input_array[i]
        y_correct = output_array[i]
        
        # --- Forward pass ---
        n_one_out = np.zeros((n_one, 1))
        for j in range(n_one):
            n_one_output = x[0] * w_one[j, 0] + x[1] * w_one[j, 1] + b_one[j]
            n_one_out[j, 0] = sigmoid(n_one_output)
    
        final_output = np.dot(w_out.T, n_one_out) + b_out
        final_out = sigmoid(final_output)
        
        loss_c = loss(final_out, y_correct)
        total_loss += loss_c
        
        delta_output_w, delta_output_b = backpropagation(final_out, y_correct, n_one_out)
        delta_hidden_w, delta_hidden_b = backpropagation_hidden(final_out, y_correct, w_out, n_one_out, X[:, i].reshape(-1, 1)) 
        
        w_out, b_out = grad_desc(alfa, w_out, b_out, delta_output_w, delta_output_b)
        w_one, b_one = grad_desc(alfa, w_one, b_one, delta_hidden_w, delta_hidden_b)
        
# --- Results control ---
final_out = np. zeros((1, 4))
for i in range(input_array.shape[0]):
    x = input_array[i]
    y_correct = output_array[i]

    # --- Forward pass ---
    n_one_out = np.zeros((n_one, 1))
    for j in range(n_one):
        n_one_output = x[0] * w_one[j, 0] + x[1] * w_one[j, 1] + b_one[j]
        n_one_out[j, 0] = sigmoid(n_one_output)

    final_output = np.dot(w_out.T, n_one_out) + b_out
    final_out[:, i] = sigmoid(final_output)

print(output_array.T)
print(final_out)