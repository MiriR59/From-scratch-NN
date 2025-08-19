import numpy as np
from Losses import BCE, MSE, CCE
from Activations import ReLU, sigmoid, SiLU, tanh, one
from Layers import Dense, Attention, MultiHeadAttention, Embedding, PositionalEmbedding, Embedding_block, He, Random, Xavier
from Network import Neural_network, L2, Null
from Optimization import ADAM, gradient_descent, LR_cosine_annealing, LR_decay, LR_exponential
from Tokenizer import tokenizer

def softmax(input):
    e = np.exp(input - np.max(input, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

alfa = 5e-4
epochs = 5000
batch_size = 10
loss_f = CCE()
optimizer = ADAM(alfa)
scheduler = LR_decay(optimizer, 0.9, 25)
# --- LM test loop ---
vocabulary = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
tokenize = tokenizer(vocabulary)
embedding = Embedding_block(64, len(vocabulary), 5)
attention = MultiHeadAttention(64, 64, 4)
dense_1 = Dense(64, 128, SiLU(), He(), 'normal')
dense_2 = Dense(128, 64, SiLU(), He(), 'normal')
dense_3 = Dense(64, len(vocabulary), one(), He(), 'normal')
nn_embed = Neural_network(loss_f, optimizer, embedding, attention, dense_1, dense_2, dense_3)

with open('words.txt', 'r') as file:
    dataset = [line.strip() for line in file if line.strip()]
# dataset = ['hello', 'apple', 'mouse', 'chill', 'penny', 'white']
losses = []
tokens = 3 # number of predicted letters

for epoch in range(epochs):
    epoch_loss = 0
    
    for j in range(len(dataset)//batch_size):
        data = dataset[j*batch_size:(j+1)*batch_size]
        enco = tokenize.encoder(data)
        ID = enco[:, :4]
        true = np.zeros((batch_size, 4, len(vocabulary)))
        for i in range(4):
            for k in range(batch_size):
                true[k, i, enco[k, i+1]] = 1

        pred = nn_embed.forward_pass(ID)
        loss = nn_embed.loss(pred, true)

        nn_embed.backpropagation()
        nn_embed.optimize()

        epoch_loss += loss

    # scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/j}')


data_numpy = np.array(dataset)
sample_size = 20
random = np.random.choice(len(data_numpy), size=sample_size, replace=False)
sample = data_numpy[random].tolist()
encoded = tokenize.encoder(sample)
finish = np.zeros([sample_size, embedding.positional_embed.max_input_length]).astype(int)
finish[:, :5-tokens] = encoded[: , :5-tokens]

for l in reversed(range(tokens)):
    start = finish[:, :5-l-1]
    pred = nn_embed.forward_pass(start)
    pred = softmax(pred)
    for m in range(pred.shape[0]):
        finish[m, -l-1] = int(np.argmax(pred[m, -1]))

finish = tokenize.decoder(finish)
for n in range(len(finish)):
    print('From:', sample[n][:5-tokens], 'Generated:', finish[n], 'Correct:', sample[n])