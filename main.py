import numpy as np

## --- NN from scratch --- ##
# --- Activation class definitions ---

class Activation:
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()
    
class sigmoid(Activation):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self):
        return self.out * (1 - self.out)
    
class ReLU(Activation):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self):
        return np.where(self.out > 0, 1, 0)
    
class tanh(Activation):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self):
        return 1 - self.out ** 2

class Swish(Activation):
    def forward(self, x):
        self.x = x
        self.out = x / (1 + np.exp(-x))
        return self.out
    
    def backward(self):
        return 1 / (1 + np.exp(-self.x)) + self.out * (1 - 1 / (1 + np.exp(-self.x)))

# --- Loss definitions ---
class Loss:
    def forward(self, x_pred, x_true):
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()
        
class BCE(Loss):
    def forward(self, x_pred, x_true):
        fix = 1e-10
        self.x_pred = x_pred
        self.x_true = x_true
        x_pred = np.clip(x_pred, fix, 1 - fix)
        self.loss = np.mean(-(x_true * np.log(x_pred) + (1 - x_true) * np.log(1 - x_pred)))
        return self.loss
        
    def backward(self):
        return (self.x_pred - self.x_true) / (self.x_pred * (1 - self.x_pred))
    
class MSE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        self.loss = np.mean((x_pred - x_true) ** 2)
        return self.loss
    
    def backward(self):
        return 2 * (self.x_pred - self.x_true) / len(self.x_pred)          # Check len(x_pred) logic for batching
    
class CCE(Loss):
    def forward(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        self.loss = - np.sum(x_true * np.log(x_pred))
        return self.loss
    
    def backward(self):
        return (self.x_pred - self.x_true) / (self.x_pred * (1 - self.x_pred))

# --- Initialisation class definitions ---
class Initialisation:
    def init(self, shape):
        raise NotImplementedError()

class Xavier(Initialisation):
    def init (self, type, inputs, neurons):
        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(neurons, inputs) - 0.5) * 2 * np.sqrt(6 / (inputs + neurons))

        elif type == 'normal':
            w = (np.random.randn(neurons, inputs)) * np.sqrt(2 / (inputs + neurons))

        else:
            print('Choose viable Initialisation type')

        return w, b
    
class He(Initialisation):
    def init (self, type, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons

        'Type: uniform/normal'
        b = np.zeros((neurons, 1))
        if type == 'uniform':
            w = (np.random.rand(neurons, inputs) - 0.5) * 2 * np.sqrt(6 / inputs)

        elif type == 'normal':
            w = (np.random.randn(neurons, inputs)) * np.sqrt(2 / inputs)

        else:
            print('Choose viable Initialisation type')

        return w, b
    
class Random(Initialisation):
    def init(self, inputs, neurons):
        w = (np.random.rand(neurons, inputs) - 0.5) * 2
        b = (np.random.rand(neurons, 1) - 0.5) * 2
        return w, b 

# --- Optimization algorithms definitions ---
class Gradient_descent:
    def __init__(self, alfa):
        self.alfa = alfa
        
    def optimize(self, layers):
        for layer in layers:
            layer.w -= self.alfa * layer.dw
            layer.b -= self.alfa * layer.db

class ADAM:
    def __init__(self, alfa, beta1=0.9, beta2=0.999, epsilon=1e-9):
        self.alfa = alfa
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []

    def optimize(self, layers):
        for i, layer in enumerate(layers):
            if self.t == 0:
                self.m_w.append(np.zeros_like(layer.w))
                self.v_w.append(np.zeros_like(layer.w))
                self.m_b.append(np.zeros_like(layer.b))
                self.v_b.append(np.zeros_like(layer.b))

            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.dw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (layer.dw ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)

            if self.t > 0:
                self.m_w_cor = self.m_w[i] / (1 - self.beta1 ** self.t)
                self.v_w_cor = self.v_w[i] / (1 - self.beta2 ** self.t)
                self.m_b_cor = self.m_b[i] / (1 - self.beta1 ** self.t)
                self.v_b_cor = self.v_b[i] / (1 - self.beta2 ** self.t)
            else:
                self.m_w_cor = self.m_w[i]
                self.v_w_cor = self.v_w[i]
                self.m_b_cor = self.m_b[i]
                self.v_b_cor = self.v_b[i]

            layer.w -= self.alfa * self.m_w_cor / (np.sqrt(self.v_w_cor) + self.epsilon)
            layer.b -= self.alfa * self.m_b_cor / (np.sqrt(self.v_b_cor) + self.epsilon)

        self.t += 1


# --- Schedulers definitions ---
class Scheduler:
    def __init__(self, optimizer, **kwargs):
        self.t = 0
        self.optimizer = optimizer
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def step(self):
        raise NotImplementedError()
    
class LR_decay(Scheduler):
    def __init__(self, optimizer, decay_rate, interval):
        super().__init__(optimizer, decay_rate=decay_rate, interval=interval)

    def step(self):
        self.t += 1

        if self.t % self.interval == 0:
            self.optimizer.alfa *= self.decay_rate

class LR_exponential(Scheduler):
    def __init__(self, optimizer, decay_rate):
        super().__init__(optimizer, decay_rate=decay_rate, alfa=optimizer.alfa)

    def step(self):
        self.optimizer.alfa = self.alfa * (self.decay_rate ** self.t)
        self.t += 1
    
class LR_cosine_annealing(Scheduler):
    def __init__(self, optimizer, T, min_alfa, max_alfa):
        super().__init__(optimizer, T=T, min_alfa=min_alfa, max_alfa=max_alfa)

    def step(self):
        self.optimizer.alfa = self.min_alfa + (self.max_alfa - self.min_alfa) * (1 + np.cos(np.pi * self.t / self.T)) / 2
        self.t += 1

# --- Layer class definitions ---
class Layer:
    def forward(self, input):
        raise NotImplementedError()
        
    def backward(self, *args):
        raise NotImplementedError()

    def backward(self, *args):
        raise NotImplementedError()

class Dense(Layer):
    def __init__(self, inputs, neurons, activation, initialisation, init_type):
        self.activation = activation
        self.w, self.b = initialisation.init(init_type, inputs, neurons)
    
    def forward(self, input):
        self.input = input
        self.z = self.w @ self.input + self.b
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, w_next, delta_next, regularization):
        self.delta = w_next.T @ delta_next * self.activation.backward()
        self.dw = self.delta @ self.input.T + regularization.backward(self.w)
        self.db = np.sum(self.delta, axis=1, keepdims=True)
        return self.dw, self.db, self.delta, self.w

class L2:
    def __init__(self, lambd):
        self.lambd = lambd
        self.L2_loss = 0

    def forward(self, layers):
        for layer in layers:
            self.L2_loss += np.sum(layer.w ** 2)

        return self.L2_loss * self.lambd
    
    def backward(self, w):
        return 2 * self.lambd * w

class Null:
    def forward(self, layers):
        return 0
    
    def backward(self, w):
        return 0
        
# --- General NN definition ---
class Neural_network:
    def __init__(self, loss_f, optimizer, *layers, regularization=None):
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.regularization = regularization if regularization else Null()
        self.layers = list(layers)
        
    def forward_pass(self, x):
        self.x = x
        for i in range(len(self.layers)):
            self.x = self.layers[i].forward(self.x)
        return self.x

    def loss(self, x_pred, x_true):
        self.x_pred = x_pred
        self.x_true = x_true
        return self.loss_f.forward(self.x_pred, self.x_true) + self.regularization.forward(self.layers)
    
    def backpropagation(self):
        self.delta = self.loss_f.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].backward(np.ones((self.delta.shape[0], self.delta.shape[0])), self.delta, self.regularization)
                
            else:
                self.layers[i].backward(self.layers[i+1].w, self.layers[i+1].delta, self.regularization)

    def optimize(self):
        self.optimizer.optimize(self.layers)
    
# --- Test loop ---
alfa = 1e-2
epochs = 10000
batch_size = 2
# --- XOR problem ---
dataset = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
true = np.array([[0], [1], [1], [0]])

loss_f = BCE()
optimizer = ADAM(alfa)
scheduler = LR_cosine_annealing(optimizer, 10000, 0.00001, 0.01)

hidden_1 = Dense(2, 10, sigmoid(), He(), 'normal')
hidden_2 = Dense(10, 10, sigmoid(), He(), 'normal')
output_layer = Dense(10, 1, sigmoid(), He(), 'normal')

nn = Neural_network(loss_f, optimizer, hidden_1, hidden_2, output_layer, regularization=L2(lambd=0.000001))

losses = []
for epoch in range(epochs):
    total_loss = 0

    for l in range(0, dataset.shape[0], batch_size):
        batch = dataset[(l):(l+batch_size), :].T
        batch_true = true[(l):(l+batch_size), :].T
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
final = nn.forward_pass(dataset.T)
print(true.T)
print(final)