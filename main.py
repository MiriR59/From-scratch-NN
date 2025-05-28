import numpy as np
from Losses import BCE, MSE, CCE
from Activations import ReLU, sigmoid, SiLU, tanh
from Layers import Dense, Embedding, He, Random, Xavier
from Network import Neural_network, L2, Null
from Optimization import ADAM, gradient_descent, LR_cosine_annealing, LR_decay, LR_exponential
from Tokenizer import tokenizer

# --- Test loop ---
alfa = 1e-2
epochs = 10000
batch_size = 3

# --- XOR problem ---
dataset = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
true = np.array([[0], [1], [1], [0]])

loss_f = BCE()
optimizer = ADAM(alfa)
scheduler = LR_cosine_annealing(optimizer, 10000, 0.00001, 0.01)

hidden_1 = Dense(2, 10, sigmoid(), He(), 'normal')
hidden_2 = Dense(10, 10, sigmoid(), He(), 'normal')
output_layer = Dense(10, 1, sigmoid(), He(), 'normal')

nn = Neural_network(loss_f, optimizer, hidden_1, hidden_2, output_layer)

# --- LM test loop ---
lay_1 = Embedding(27, 16)
lay_2 = Dense(16, 32, sigmoid(), He(), 'normal')
nn_embed = Neural_network(loss_f, optimizer, lay_1, lay_2)
tokenize = tokenizer()
sentence = 'hello'

ID = tokenize.encoder(sentence)
prediction = nn_embed.forward_pass(ID)


losses = []
for epoch in range(epochs):
    total_loss = 0

    for l in range(0, dataset.shape[0], batch_size):
        batch = dataset[(l):(l+batch_size), :]
        batch_true = true[(l):(l+batch_size), :]
        x_pred = nn.forward_pass(batch)

        loss_value = nn.loss(x_pred, batch_true)
        total_loss += loss_value

        nn.backpropagation()
        nn.optimize()
        
    total_loss /= (dataset.shape[0] / batch_size)
    losses.append(total_loss)
    scheduler.step()

    if epoch % 1 == 0:
        print(f'Epoch: {epoch}, Loss: {total_loss}, LR: {optimizer.alfa}')
    
# --- Results control ---
final = nn.forward_pass(dataset)
print(true)
print(final)