class Layer:
    def __init__(self, *args):
        raise NotImplementedError()
            
    def forward(self, input):
        raise NotImplementedError()
        
    def backward(self, *args):
        raise NotImplementedError()