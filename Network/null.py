class Null:
    def forward(self, layers):
        return 0
    
    def backward(self, w):
        return 0

