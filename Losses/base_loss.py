class Loss:
    def forward(self, x_pred, x_true):
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()