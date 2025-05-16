import numpy as np

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