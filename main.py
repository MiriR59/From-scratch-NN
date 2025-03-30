import numpy as np

# --- NN from zero ---
# 2 input - 10 neurons - 1 output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss
def loss(x, y):
    '''
    x = predicted output
    y = correct answer
    '''
    fix = 1e-9
    x = np.clip(x, fix, 1 - fix)  # Ensure we never get log 0
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


# inputs
n_input = 2     # Input features
n_one = 10      # neurons in 1st layer
n_out = 1       # outputs
alfa = 1e-2     # learning rate
epochs = 150000

# XOR problem input
input_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_array = np.array([[0], [1], [1], [0]])

# Transpose inputs for matrix operations
X = input_array.T
y = output_array.T

# Weights for 1st layer
w_one = (np.random.rand(n_one, n_input) - 0.5) * 2
# Weights for output layer
w_out = (np.random.random((n_one, n_out)) - 0.5) * 2

# Biases for 1st layer
b_one = (np.random.random(n_one) - 0.5) * 2
# Biases for output layer
b_out = (np.random.random(n_out) - 0.5) * 2

# Layer output storing
n_one_out = np.zeros(n_one)

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
        
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')
        
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