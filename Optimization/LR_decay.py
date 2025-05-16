from .base_scheduler import Scheduler

class LR_decay(Scheduler):
    def __init__(self, optimizer, decay_rate, interval):
        super().__init__(optimizer, decay_rate=decay_rate, interval=interval)

    def update(self):
        if self.t % self.interval == 0:
            self.optimizer.alfa *= self.decay_rate