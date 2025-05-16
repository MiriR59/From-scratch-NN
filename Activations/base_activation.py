class Activation:
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()