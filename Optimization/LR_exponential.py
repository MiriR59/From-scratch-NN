from .base_scheduler import Scheduler

class LR_exponential(Scheduler):
    def __init__(self, optimizer, decay_rate):
        super().__init__(optimizer, decay_rate=decay_rate, alfa=optimizer.alfa)

    def update(self):
        self.optimizer.alfa = self.alfa * (self.decay_rate ** self.t)