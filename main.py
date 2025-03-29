import numpy as np

# --- NN from zero ---
# 2 input - 10 neurons - 1 output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# inputs
n_input = 2
# neurons in 1st layer
n_one = 10
# outputs
n_out = 1

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

for x, y in input_array:
    for i in range(n_one):
        n_one_output = x * w_one[i, 0] + y * w_one[i, 1] + b_one[i]
        n_one_out[i] = sigmoid(n_one_output)
    for j in range (n_out):
        final_output = np.dot(w_out, n_one_out) + b_out
        final_out = sigmoid(final_output)
        