import numpy as np

# --- NN from zero ---
# 2 input - 10 neurons - 1 output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss
def loss(x, y):
    fix = 1e-9
    if x == 1:
        x -= fix
    if x == 0:
        x += fix
    else:
        x == x
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))

# --- Backpropagation ---
def backpropagation(x, y, z, w): # x - predicted value, y - correct value, z - weight array, w - output of previous layer
    dL_dw = np.zeros_like(z)
    dL_db = (x - y) * x * (1 - x)
    for k in range(len(z)):
        dL_dw[k] = (x - y) * x * (1 - x) * w[k]
    return dL_dw, dL_db

# --- Gradient descent ---
def grad_desc(alfa, w, b, dL_dw, dL_db): # alfa - learning rate, w - weight array, b - bias array, dL_dw - weight gradient, dL_db - bias gradient
    w_updated = w - alfa * dL_dw
    b_updated = b - alfa * dL_db
    return w_updated, b_updated

# inputs
n_input = 2
# neurons in 1st layer
n_one = 10
# outputs
n_out = 1
# learning rate
alfa = 1e-3

# XOR problem input
input_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Weights for 1st layer
w_one = (np.random.rand(n_one, n_input) - 0.5) * 2
# Weights for output layer
w_out = (np.random.random(n_one) - 0.5) * 2

# Biases for 1st layer
b_one = (np.random.random(n_one) - 0.5) * 2
# Biases for output layer
b_out = (np.random.random(n_out) - 0.5) * 2

# Layer output storing
n_one_out = np.zeros(n_one)

# --- Forward pass ---
for x, y in input_array:
    for i in range(n_one):
        n_one_output = x * w_one[i, 0] + y * w_one[i, 1] + b_one[i]
        n_one_out[i] = sigmoid(n_one_output)
    for j in range (n_out):
        final_output = np.dot(w_out, n_one_out) + b_out
        final_out = sigmoid(final_output)

# Output layer weight + bias update
dL_dw_out, dL_db_out = backpropagation(final_out, 1, w_out, n_one_out)
w_out_t, b_out_u = grad_desc(alfa, w_out, b_out, dL_dw_out, dL_db_out)

# Hidden layer weight + bias update
dL_dw_one, dL_db_one = backpropagation_hidden(final_out, 1, z, w)

def backpropagation_hidden (x, y, w, b, h): # x - predicted value, y - correct value, w - weight array, b - bias array, h - output array of hidden layer
    delta_hidden = np.zeros_like(w)
    dL_dw = delta_hidden
    dL_dw = np.zeros_like(b)
    for l in range(len(b)):
        delta_out = (x - y) * x * (1 - x)
        delta_hidden[l] = delta_out * w[l] * h[l] * (1 - h[l])
        
    