class Scheduler:
    def __init__(self, optimizer, **kwargs):
        self.t = 0
        self.optimizer = optimizer
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def step(self):
        self.update()
        self.t += 1
        
    def update(self):
        raise NotImplementedError()