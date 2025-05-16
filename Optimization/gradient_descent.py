class gradient_descent:
    def __init__(self, alfa):
        self.alfa = alfa
        
    def optimize(self, layers):
        for layer in layers:
            layer.w -= self.alfa * layer.dw
            layer.b -= self.alfa * layer.db